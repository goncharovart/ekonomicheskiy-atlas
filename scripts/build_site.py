"""Данные сайта: site/data/*.json и site/data/atlas.js (тот же набор одним скриптом,
чтобы страница открывалась и с GitHub Pages, и прямо из файла, где fetch запрещён).

Входы (результаты модели, пишут другие части):
  outputs/labels/final.parquet  итоговая типология (KEFRiN, K = 7): territory_id, month, label (int)
  outputs/indices.csv           индексы: method, name, family, variant, period, K, SW … MQ, z_*
  outputs/method_ranking.csv    места: пороговое агрегирование и Борда, с AVU и без
  outputs/types.csv             необязательно: label, name, description
  outputs/interpret/profiles.csv необязательно: траты на жителя по типам (spend_ratio, spend_rub_geo)
ATLAS.layout — картограммы для сайта в км проекции geo.json: c (центроиды), eq (равные клетки:
шестиугольная решётка поверх Дорлинга), pop (кружки Дорлинга площадью по населению 2024), r_eq, r_pop.
Пока нет types.csv, названия типов — заглушки «Тип A…», в meta.stub стоит «названия типов»,
и сайт помечает их как черновые. Всё остальное — настоящие расчёты.

Запуск: .venv/Scripts/python.exe -X utf8 scripts/build_site.py
Геометрию (site/data/geo.json) готовит scripts/prepare_geo.py.
"""
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site/data"
LABELS = ROOT / "outputs/labels/final.parquet"
TYPES = ROOT / "outputs/types.csv"
PROFILES = ROOT / "outputs/interpret/profiles.csv"  # траты на жителя по типам (spend_ratio, spend_rub_geo)
INDICES = ROOT / "outputs/indices.csv"
RANKING = ROOT / "outputs/method_ranking.csv"
FINAL_METHOD = "kefrin"  # outputs/final_choice.md: итоговая типология — KEFRiN, K = 7
REPO = "https://github.com/goncharovart/ekonomicheskiy-atlas"
CATS = ["Продовольствие", "Маркетплейсы", "Общественное питание", "Здоровье", "Транспорт"]
CAT_SHORT = {"Продовольствие": "продукты", "Маркетплейсы": "маркетплейсы",
             "Общественное питание": "кафе и рестораны", "Здоровье": "здоровье", "Транспорт": "транспорт"}
# Семь цветов типов. Минимальный ΔE2000 по всем парам и против серого «нет данных» (#e4e1da,
# #d9d6cf): обычное зрение 23,1; протан 10,1; дейтан 13,2; тритан 14,6 (Machado 2009, полная
# тяжесть). Седьмой — коричневый, не серый: серый занят МО без данных.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e8c13a", "#b8327a", "#6e4a1a"]
INDEX_META = [  # ключ, название, лучше, на чём считается, по чему ставится оценка в рейтинге
    ("SW", "Силуэт", "max", "признаки", "SW"), ("CH", "Калински–Харабаш", "max", "признаки", "CH"),
    ("S_Dbw", "S_Dbw", "min", "признаки", "S_Dbw"), ("AVI", "Изолированность AVI", "max", "граф", "z_AVI"),
    ("AVU", "Слитость AVU", "min", "граф", "z_AVU"), ("MQ", "Модулярность Q", "max", "граф", "MQ"),
]
FAMILY = {"baseline": "базовый", "jury": "метод члена жюри"}


def load():
    mo = pd.read_parquet(ROOT / "data/processed/mo.parquet")
    mo = mo[mo["balanced"]]
    attrs = pd.read_parquet(ROOT / "data/processed/mo_attributes.parquet")
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    panel = panel[panel["territory_id"].isin(mo.index)]
    return mo, attrs, panel


def mode_row(a):
    c = np.bincount(a)
    return int(np.flatnonzero(c == c.max())[-1]) if len(a) else -1


def methods_table():
    """Основной вариант сравнения (общее K, variant=common) по двум периодам: значения из indices.csv,
    места и оценки из method_ranking.csv (сайт мест не пересчитывает)."""
    ind = pd.read_csv(INDICES)
    rk = pd.read_csv(RANKING)
    t = ind[ind["variant"] == "common"].merge(rk, on=["variant", "period", "method"], suffixes=("", "_r"), validate="1:1")
    assert len(t) == (ind["variant"] == "common").sum(), "method_ranking.csv не покрывает indices.csv"
    keys = [k for k, *_ in INDEX_META]
    out = {}
    for period, g in t.groupby("period"):
        out[period] = [{
            "key": r["method"], "name": r["name"], "family": FAMILY.get(r["family"], r["family"]), "K": int(r["K"]),
            **{k: _r(r[k]) for k in keys}, "z": {k: _r(r[f"z_{k}"], 1) for k in ("AVI", "AVU", "MQ")},
            "grade": {k: int(r[f"grade_{rk_by}"]) for k, *_, rk_by in INDEX_META},
            "place": int(r["threshold_place"]), "place_noavu": int(r["threshold_place_without_AVU"]),
            "borda": int(r["borda_place"]), "borda_noavu": int(r["borda_place_without_AVU"]),
            "borda_points": _r(r["borda_points"], 1), "final": r["method"] == FINAL_METHOD,
        } for _, r in g.iterrows()]
    return out


def dorling(c, r, iters=500, seed=0):
    """Кружки радиуса r без наложений, каждый как можно ближе к своему центроиду c (картограмма Дорлинга)."""
    p = c + np.random.default_rng(seed).normal(0, 1e-3, c.shape) * r[:, None]
    for it in range(iters):
        pairs = cKDTree(p).query_pairs(2 * r.max(), output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        d = p[j] - p[i]
        dist = np.hypot(d[:, 0], d[:, 1]) + 1e-9
        ov = r[i] + r[j] - dist
        m = ov > 1e-3 * r.min()
        if not m.any():
            break
        mv = d[m] / dist[m, None] * (ov[m] / 2)[:, None]
        delta = np.zeros_like(p)
        np.add.at(delta, i[m], -mv)
        np.add.at(delta, j[m], mv)
        p += 0.6 * delta + 0.02 * (1 - it / iters) * (c - p)
    return p


def hexsnap(p, r):
    """Каждому МО — своя клетка шестиугольной решётки (шаг 2r), ближайшая к его месту на картограмме."""
    dy = np.sqrt(3) * r
    cell = lambda row, col: np.array([(col + (row % 2) / 2) * 2 * r, row * dy])
    taken, out = set(), np.zeros_like(p)
    for i in np.lexsort((p[:, 0], -p[:, 1])):
        row0 = int(round(p[i, 1] / dy)); col0 = int(round(p[i, 0] / (2 * r) - (row0 % 2) / 2))
        for k in range(1, 60):
            cand = [(np.hypot(*(cell(a, b) - p[i])), a, b) for a in range(row0 - k, row0 + k + 1)
                    for b in range(col0 - k, col0 + k + 1) if (a, b) not in taken]
            if cand:
                _, a, b = min(cand)
                taken.add((a, b)); out[i] = cell(a, b)
                break
    return out


def layouts(ids, pop):
    """Центроиды МО и две картограммы в км той же проекции, что geo.json: равные клетки и кружки по населению."""
    geo = json.loads((OUT / "geo.json").read_text(encoding="utf-8"))
    cen = {f["id"]: shapely.geometry.shape(f["geometry"]).centroid for f in geo["features"]}
    c = np.array([[cen[t].x, cen[t].y] for t in ids])
    area = sum(shapely.geometry.shape(f["geometry"]).area for f in geo["features"])
    r_eq = np.sqrt(0.6 * area / len(ids) / np.pi)  # клетки вместе — 60 % площади страны: запад раздвигается, восток не пустеет
    eq = hexsnap(dorling(c, np.full(len(ids), r_eq)), r_eq)
    pp = np.nan_to_num(pop, nan=np.nanmedian(pop))
    r_pop = np.sqrt(pp / pp.sum() * 0.45 * area / np.pi)  # площадь кружка ∝ населению, всего 45 % площади
    r_pop = np.maximum(r_pop, 0.25 * r_eq)
    dp = dorling(c, r_pop)
    rd = lambda a: np.rint(a).astype(int).tolist()
    return {"r_eq": round(float(r_eq), 2), "c": [rd(c[:, 0]), rd(c[:, 1])], "eq": [rd(eq[:, 0]), rd(eq[:, 1])],
            "pop": [rd(dp[:, 0]), rd(dp[:, 1])], "r_pop": np.round(r_pop, 1).tolist(),
            "population": [None if np.isnan(x) else int(x) for x in pop]}


def main():
    mo, attrs, panel = load()
    stub = []
    months = sorted(panel["month"].unique())
    mkeys = [pd.Timestamp(m).strftime("%Y-%m") for m in months]
    ids = sorted(mo.index)
    idx = {t: i for i, t in enumerate(ids)}

    # Доли пяти категорий: [МО][месяц][категория], настоящие.
    p = panel[panel["category"].isin(CATS)].pivot_table(index=["territory_id", "month"], columns="category", values="share")
    p = p.reindex(pd.MultiIndex.from_product([ids, months]))[CATS].fillna(0)
    shares = p.to_numpy().reshape(len(ids), len(months), len(CATS))

    # Типы по месяцам.
    L = pd.read_parquet(LABELS)
    L["month"] = pd.to_datetime(L["month"]).dt.strftime("%Y-%m")
    lab = L.pivot(index="territory_id", columns="month", values="label").reindex(index=ids, columns=mkeys)
    lab = lab.fillna(-1).astype(int).to_numpy()
    K = int(lab.max()) + 1
    assert K <= len(COLORS), f"типов {K}, проверенных цветов {len(COLORS)}: подобрать и проверить ещё цвета"

    names = {}
    if TYPES.exists():
        t = pd.read_csv(TYPES)
        desc = t["description"] if "description" in t else pd.Series("", index=t.index)
        names = {int(l): (str(n).strip(), d.strip() if isinstance(d, str) else "")
                 for l, n, d in zip(t["label"], t["name"], desc) if isinstance(n, str) and n.strip()}
    if len(names) < K:
        stub.append("названия типов")

    modal = np.array([mode_row(r[r >= 0]) for r in lab])
    stability = np.array([(r == m).mean() for r, m in zip(lab, modal)])
    switches = (np.diff(lab, axis=1) != 0).sum(1)

    # Темпы: рост трат 2024 к 2023 по каждому МО (всего и по категориям), настоящие.
    v = panel.assign(y=panel["month"].dt.year).groupby(["territory_id", "category", "y"])["value"].sum().unstack("y")
    growth = (v[2024] / v[2023] - 1).unstack("category").reindex(ids)

    X = shares.reshape(-1, len(CATS))
    labf = lab.ravel()
    ru = X.mean(0)
    at = attrs.reindex(ids)
    pop = at["population_2024"].fillna(at["population_2023"])
    prof = pd.read_csv(PROFILES).set_index("type") if PROFILES.exists() else pd.DataFrame()
    types = []
    for k in range(K):
        sel = labf == k
        inmod = modal == k
        mean = X[sel].mean(0) if sel.any() else np.zeros(len(CATS))
        rel = mean / ru  # во сколько раз доля выше средней: по разности три сельских типа выходили одинаковыми
        x = lambda j: f"×{rel[j]:.2f}".replace(".", ",")
        auto = f"Выше среднего: {CAT_SHORT[CATS[rel.argmax()]]} {x(rel.argmax())}; ниже: {CAT_SHORT[CATS[rel.argmin()]]} {x(rel.argmin())}"
        name, desc = names.get(k, (f"Тип {chr(65 + k)}", ""))
        desc = desc or auto
        ex = pd.DataFrame({"id": ids, "pop": pop.to_numpy(), "stab": stability})[inmod]
        ex = ex[ex["stab"] >= ex["stab"].quantile(0.5)].nlargest(4, "pop") if len(ex) else ex
        g = growth[inmod]
        types.append({
            "id": k, "name": name, "desc": desc, "color": COLORS[k],
            "n_mo": int(inmod.sum()), "share_mo_months": round(float(sel.mean()), 4),
            "shares": [round(float(x), 4) for x in mean],
            "growth_total": _r(g["Все категории"].median()),
            "growth": [_r(g[c].median()) for c in CATS],
            "wage_2023": _r(at.loc[inmod, "wage_2023"].median(), 0),
            "population_2024": _r(pop[inmod].median(), 0),
            # из профилей, как в описании типа: иначе на одной карточке две разные медианы
            "market_access_2024": _r(prof.at[str(k), "market_access_median"], 3) if str(k) in prof.index else None,
            "stability": _r(stability[inmod].mean(), 3),
            "spend_ratio": _r(prof.at[str(k), "spend_ratio"], 3) if str(k) in prof.index else None,
            "spend_rub": _r(prof.at[str(k), "spend_rub_geo"], 0) if str(k) in prof.index else None,
            "examples": [{"id": int(r.id), "n": mo.at[r.id, "name"], "r": mo.at[r.id, "region_name"]} for r in ex.itertuples()],
        })

    # Переходы по кварталам: тип квартала — самый частый из трёх месяцев.
    quarters = [f"{m[:4]} К{(int(m[5:]) - 1) // 3 + 1}" for m in mkeys[::3]]
    ql = np.array([[mode_row(r[q * 3:q * 3 + 3][r[q * 3:q * 3 + 3] >= 0]) for q in range(len(quarters))] for r in lab])
    links = []
    for q in range(len(quarters) - 1):
        pairs = pd.Series(list(zip(ql[:, q], ql[:, q + 1]))).value_counts()
        links += [{"q": q, "s": int(s), "t": int(t), "n": int(n)} for (s, t), n in pairs.items() if s >= 0 and t >= 0]
    nodes = [[int((ql[:, q] == k).sum()) for k in range(K)] for q in range(len(quarters))]

    # В заголовке — разрыв темпов роста трат между типами и устойчивость. Число МО по типам между
    # кварталами не годится: у соседних типов (города 3 и 5) оно скачет туда-обратно на ~110 МО за месяц.
    gt = [t["growth_total"] or 0 for t in types]
    hi, lo = int(np.argmax(gt)), int(np.argmin(gt))
    nm = lambda k: types[k]["name"].split(":")[0].strip()
    pc = lambda x, f=".1%": format(x, f).replace(".", ",").replace("%", " %")
    headline = (f"Траты по картам быстрее всего растут в типе «{nm(hi)}» (+{pc(gt[hi])} за 2024 год к 2023-му), "
                f"медленнее всего — в типе «{nm(lo)}» (+{pc(gt[lo])}). Муниципалитет проводит в своём основном "
                f"типе в среднем {pc(stability.mean(), '.0%')} месяцев.")

    meta = {
        "stub": stub, "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "months": mkeys, "categories": CATS, "n_mo": len(ids), "n_regions": int(mo["region_name"].nunique()),
        "types": types, "headline": headline,
        "ru_shares": [round(float(x), 4) for x in ru], "ru_growth_total": _r(growth["Все категории"].median()),
        "links": {"repo": REPO, "report": REPO + "/blob/main/reports/ОТЧЁТ.md", "pdf": "landing.pdf"},
    }
    data = {
        "meta": meta,
        "labels": {"ids": ids, "labels": lab.tolist(), "stability": np.round(stability, 3).tolist(),
                   "switches": switches.tolist()},
        "shares": {"ids": ids, "scale": 1000, "values": np.rint(shares * 1000).astype(int).tolist()},
        "transitions": {"quarters": quarters, "nodes": nodes, "links": links},
        "layout": layouts(ids, pop.to_numpy(dtype=float)),
        "methods": {"indices": [{"key": k, "name": n, "better": b, "on": o, "by": by} for k, n, b, o, by in INDEX_META],
                    "periods": methods_table()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    js = ["window.ATLAS = {};"]
    for name, obj in data.items():
        txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        (OUT / f"{name}.json").write_text(txt, encoding="utf-8")
        js.append(f"ATLAS.{name} = {txt};")
    geo = OUT / "geo.json"
    js.append(f"ATLAS.geo = {geo.read_text(encoding='utf-8')};" if geo.exists() else "ATLAS.geo = null;")
    (OUT / "atlas.js").write_text("\n".join(js), encoding="utf-8")
    size = (OUT / "atlas.js").stat().st_size / 1e6
    print(f"site/data: {len(ids)} МО, {K} типов, {len(links)} связей, atlas.js {size:.2f} МБ, заглушки: {stub or 'нет'}")
    print(headline)


def _r(x, nd=4):
    return None if pd.isna(x) else round(float(x), nd)


if __name__ == "__main__":
    main()
