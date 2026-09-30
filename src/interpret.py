"""Интерпретация итоговой типологии (KEFRiN, K = 7, outputs/labels/final.parquet).

Тип МО — модальный тип за 24 месяца; при равенстве (25 МО) — больший номер, как на карте сайта
(scripts/build_site.py, mode_row) и в таблице outputs/final_choice.md.
  mo_table   — одна строка на МО: сквозные признаки в исходных единицах, траты на жителя, рост 2024/2023,
               население, регион, тип МО;
  profiles   — профиль типа: доли категорий и их отношение к среднему по 2 016 МО, уровень трат, рост,
               доступность рынков, население, регионы-лидеры, состав по типам МО;
  examples   — 5 МО ближе всего к центру типа и 3 пограничных (меньше всего разница расстояний до своего
               и до ближайшего чужого центра) в пространстве 15 сквозных z-признаков;
  mirkin     — профиль по Миркину: отклонение центра в σ и вклад B_kv = |S_k|(c_kv − ȳ_v)² в разброс T;
  FCA        — порядковая шкала по квартилям, понятия (A, B) с порождающими множествами до 3 признаков,
               точность и покрытие для типа, устойчивость Кузнецова: точно перебором (малые объёмы) и
               оценки Бузмакова–Кузнецова–Наполи 1 − Σ 2^−Δm ≤ σ ≤ 1 − 2^−Δ для больших;
  rosstat_check — Краскел–Уоллис по типам и η²_H = (H − k + 1)/(n − k) для показателей Росстата.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from src import features as F

ROOT = F.ROOT
SH = ["sh_grocery", "sh_market", "sh_food_out", "sh_transport", "sh_health", "sh_other"]
RU = {"sh_grocery": "продукты", "sh_market": "маркетплейсы", "sh_food_out": "общепит",
      "sh_transport": "транспорт", "sh_health": "здоровье", "sh_other": "прочее"}
GEN = {"sh_grocery": "продуктов", "sh_market": "маркетплейсов", "sh_food_out": "общепита",
       "sh_transport": "транспорта", "sh_health": "здоровья", "sh_other": "прочего"}
FEAT_RU = {"mean_log_level": "уровень трат", "trend_log_level": "тренд уровня трат",
           "market_access": "доступность рынков",
           **{f"mean_{s}": f"доля {g}" for s, g in GEN.items()},
           **{f"trend_{s}": f"тренд доли {g}" for s, g in GEN.items()}}


def feat_ru(c):
    return FEAT_RU.get(c, c)


# ---------- типы и таблица МО ----------

def modal_type(lab):
    """lab: territory_id, month, label → DataFrame(type, stability) по territory_id."""
    wide = lab.pivot(index="territory_id", columns="month", values="label").sort_index()
    a = wide.to_numpy()
    modal = np.array([np.flatnonzero((c := np.bincount(r)) == c.max())[-1] for r in a])
    return pd.DataFrame({"type": modal, "stability": (a == modal[:, None]).mean(1)}, index=wide.index)


def mo_table(root=ROOT):
    w = F.load_wide(root)
    t = F.through_raw(w)
    tot = w[F.TOTAL].unstack("month")
    yr = tot.columns.year
    t["spend_rub"] = np.exp(t["mean_log_level"])
    t["growth_24_23"] = tot.loc[:, yr == 2024].mean(axis=1) / tot.loc[:, yr == 2023].mean(axis=1) - 1
    mo = pd.read_parquet(root / "data/processed/mo.parquet").reindex(t.index)
    t["population"] = mo.population_2024.fillna(mo.population_2023)
    for c in ["name", "region_name", "mo_type", "status", "lat", "lon"]:
        t[c] = mo[c]
    t.index.name = "territory_id"
    return t


def km(lat, lon, lat0, lon0):
    """Расстояние по дуге большого круга, км."""
    la, lo, la0, lo0 = map(np.radians, (lat, lon, lat0, lon0))
    return 6371 * 2 * np.arcsin(np.sqrt(np.sin((la - la0) / 2) ** 2 + np.cos(la) * np.cos(la0) * np.sin((lo - lo0) / 2) ** 2))


# ---------- профили ----------

def profiles(t, types, top_regions=3):
    ty = types.reindex(t.index)
    rows = []
    avg = t[[f"mean_{s}" for s in SH]].mean()
    reg_n = t.region_name.value_counts()
    for k in sorted(ty.unique()):
        g = t[ty == k]
        r = {"type": k, "n_mo": len(g)}
        for s in SH:
            r[f"{s}_pct"] = 100 * g[f"mean_{s}"].mean()
            r[f"{s}_ratio"] = g[f"mean_{s}"].mean() / avg[f"mean_{s}"]
        r["spend_rub_geo"] = float(np.exp(g.mean_log_level.mean()))
        r["spend_ratio"] = r["spend_rub_geo"] / float(np.exp(t.mean_log_level.mean()))
        r["growth_median_pct"] = 100 * g.growth_24_23.median()
        r["market_access_median"] = g.market_access.median()
        r["population_median"] = g.population.median()
        vc = g.region_name.value_counts().head(top_regions)
        r["regions"] = "; ".join(f"{n} ({c} из {reg_n[n]})" for n, c in vc.items())
        mix = g.mo_type.value_counts(normalize=True)
        r["mo_type_mix"] = "; ".join(f"{n} {100 * c:.0f} %" for n, c in mix.items())
        r["capitals"] = int((g.status == "административный_центр_субъекта").sum())
        r["east60_pct"] = 100 * (g.lon >= 60).mean()
        r["lat_median"], r["lon_median"] = g.lat.median(), g.lon.median()
        r["north60_pct"] = 100 * (g.lat >= 60).mean()
        rows.append(r)
    out = pd.DataFrame(rows).set_index("type")
    out.loc["all"] = {**{f"{s}_pct": 100 * avg[f"mean_{s}"] for s in SH}, **{f"{s}_ratio": 1.0 for s in SH},
                      "n_mo": len(t), "spend_rub_geo": float(np.exp(t.mean_log_level.mean())), "spend_ratio": 1.0,
                      "growth_median_pct": 100 * t.growth_24_23.median(),
                      "market_access_median": t.market_access.median(), "population_median": t.population.median(),
                      "lat_median": t.lat.median(), "lon_median": t.lon.median(),
                      "north60_pct": 100 * (t.lat >= 60).mean(), "east60_pct": 100 * (t.lon >= 60).mean(),
                      "capitals": int((t.status == "административный_центр_субъекта").sum())}
    return out


def examples(Z, types, info, n_typical=5, n_border=3):
    """Z: z-признаки МО (индекс territory_id); info: name, region_name для подписи."""
    ty = types.reindex(Z.index).to_numpy()
    ks = np.unique(ty)
    C = np.stack([Z.to_numpy()[ty == k].mean(0) for k in ks])
    D = np.linalg.norm(Z.to_numpy()[:, None, :] - C[None], axis=2)
    own = D[np.arange(len(ty)), np.searchsorted(ks, ty)]
    Do = D.copy()
    Do[np.arange(len(ty)), np.searchsorted(ks, ty)] = np.inf
    other, d_other = ks[Do.argmin(1)], Do.min(1)
    rows = []
    for k in ks:
        idx = np.flatnonzero(ty == k)
        for role, order in (("типичное", idx[np.argsort(own[idx])][:n_typical]),
                            ("пограничное", idx[np.argsort((d_other - own)[idx])][:n_border])):
            for i in order:
                tid = Z.index[i]
                rows.append({"type": k, "role": role, "territory_id": tid, "name": info.at[tid, "name"],
                             "region": info.at[tid, "region_name"], "dist_own": own[i],
                             "nearest_other_type": other[i], "margin": d_other[i] - own[i]})
    return pd.DataFrame(rows)


def mirkin(Z, types):
    """Профиль по Миркину на z-признаках: dev_sigma = (c_kv − ȳ_v)/s_v, contrib = |S_k|(c_kv − ȳ_v)²/T."""
    ty = types.reindex(Z.index)
    ybar, sd = Z.mean(), Z.std(ddof=0)
    T = float(((Z - ybar) ** 2).to_numpy().sum())
    rows = []
    for k in sorted(ty.unique()):
        g = Z[ty == k]
        d = g.mean() - ybar
        for v in Z.columns:
            rows.append({"type": k, "feature": v, "feature_ru": feat_ru(v), "dev_sigma": d[v] / sd[v],
                         "contrib": len(g) * d[v] ** 2 / T})
    return pd.DataFrame(rows), T


# ---------- FCA ----------

LEVELS = ["нижняя четверть", "ниже медианы", "выше медианы", "верхняя четверть"]


def scale_quartiles(df):
    """Порядковая шкала: для каждого признака 4 атрибута (≤Q1, ≤Me, >Me, >Q3). Возвращает bool-матрицу и имена."""
    cols, names = [], []
    for c in df.columns:
        q1, me, q3 = df[c].quantile([0.25, 0.5, 0.75])
        for lev, m in zip(LEVELS, [df[c] <= q1, df[c] <= me, df[c] > me, df[c] > q3]):
            cols.append(m.to_numpy())
            names.append(f"{feat_ru(c)} — {lev}")
    return np.column_stack(cols), names


def extent(I, B):
    return I[:, list(B)].all(1) if len(B) else np.ones(I.shape[0], bool)


def intent(I, A):
    return tuple(np.flatnonzero(I[A].all(0))) if A.any() else tuple(range(I.shape[1]))


def all_concepts(I):
    """Все понятия малого контекста перебором подмножеств атрибутов: {intent: extent}."""
    out = {}
    for r in range(I.shape[1] + 1):
        for B in combinations(range(I.shape[1]), r):
            A = extent(I, B)
            out.setdefault(intent(I, A), tuple(np.flatnonzero(A)))
    return out


def stability_exact(I, A, B):
    """σ(A,B) = |{C ⊆ A : C' = B}| / 2^|A| — перебором, только для |A| ≤ 20."""
    A = np.flatnonzero(A) if np.asarray(A).dtype == bool else np.asarray(A)
    assert len(A) <= 20, "перебор 2^|A|"
    B, hit = tuple(B), 0
    for mask in range(1 << len(A)):
        C = np.zeros(I.shape[0], bool)
        C[[a for j, a in enumerate(A) if mask >> j & 1]] = True
        hit += intent(I, C) == B
    return hit / (1 << len(A))


def stability_bounds(I, A, B):
    """Оценки устойчивости через Δ_m = |A \\ m'| по атрибутам m ∉ B (Buzmakov, Kuznetsov, Napoli 2014).
    Подмножество C ⊆ A меняет описание, только если C ⊆ m' для какого-то m ∉ B."""
    rest = np.setdiff1d(np.arange(I.shape[1]), B)
    delta = (~I[np.ix_(A, rest)]).sum(0) if len(rest) else np.array([A.sum()])
    lo = max(0.0, 1 - float(np.sum(np.exp2(-delta.astype(float)))))
    return int(delta.min()), lo, 1 - 2.0 ** -int(delta.min())


def concepts_from_generators(I, max_len=3):
    """Замыкания всех наборов до max_len атрибутов. Уникальные понятия: список (intent, extent, generator)."""
    seen = {}
    for r in range(1, max_len + 1):
        for g in combinations(range(I.shape[1]), r):
            A = extent(I, g)
            if not A.any():
                continue
            B = intent(I, A)
            if B not in seen or len(g) < len(seen[B][1]):
                seen[B] = (A, g)
    return [(B, A, g) for B, (A, g) in seen.items()]


def readable(B, names):
    """Убрать из содержания атрибуты, следующие из шкалы: «верхняя четверть» ⇒ «выше медианы»."""
    s = [names[b] for b in B]
    drop = {x.replace("верхняя четверть", "выше медианы") for x in s if x.endswith("верхняя четверть")}
    drop |= {x.replace("нижняя четверть", "ниже медианы") for x in s if x.endswith("нижняя четверть")}
    return [x for x in s if x not in drop]


def fca_rules(I, names, y, max_len=3, min_precision=0.5, min_sigma=0.99, top=3):
    """Для каждого типа — понятия с точностью ≥ min_precision и нижней оценкой σ ≥ min_sigma,
    лучшие по F1 (точность × покрытие). y — типы МО (np.array)."""
    cs = concepts_from_generators(I, max_len)
    E = np.stack([A for _, A, _ in cs])
    ks = np.unique(y)
    Y = y[:, None] == ks[None]
    hit, size, nk = E.astype(int) @ Y.astype(int), E.sum(1), Y.sum(0)
    rows = []
    for j, k in enumerate(ks):
        prec, cov = hit[:, j] / size, hit[:, j] / nk[j]
        f1 = 2 * prec * cov / np.maximum(prec + cov, 1e-12)
        cand = [i for i in np.argsort(-f1) if prec[i] >= min_precision][:200]
        got = 0
        for i in cand:
            B, A, g = cs[i]
            d, lo, hi = stability_bounds(I, A, B)
            if lo < min_sigma:
                continue
            rows.append({"type": k, "rule": " И ".join(readable(B, names)),
                         "generator": " И ".join(names[x] for x in g), "n_extent": int(size[i]),
                         "n_in_type": int(hit[i, j]), "precision": prec[i], "coverage": cov[i], "f1": f1[i],
                         "lift": prec[i] / (nk[j] / len(y)), "delta": d, "sigma_low": lo, "sigma_up": hi,
                         "extent_by_type": "; ".join(f"{kk}: {hit[i, jj]}" for jj, kk in enumerate(ks)
                                                     if hit[i, jj])})
            got += 1
            if got == top:
                break
    return pd.DataFrame(rows), len(cs)


# ---------- внешняя проверка ----------

def eta2_h(H, k, n):
    return (H - k + 1) / (n - k)


def rosstat_check(X, types):
    """X: показатели по МО (индекс territory_id). Краскел–Уоллис по типам, η²_H, медианы по типам."""
    ty = types.reindex(X.index)
    kw, med = [], {}
    for c in X.columns:
        s = X[c].dropna()
        groups = [s[ty.reindex(s.index) == k].to_numpy() for k in sorted(ty.unique())]
        groups = [g for g in groups if len(g)]
        H, p = kruskal(*groups)
        kw.append({"indicator": c, "n": len(s), "H": H, "p": p, "eta2_H": eta2_h(H, len(groups), len(s))})
        med[c] = s.groupby(ty.reindex(s.index)).median()
    med = pd.DataFrame(med)
    med.loc["all"] = X.median()
    return pd.DataFrame(kw), med
