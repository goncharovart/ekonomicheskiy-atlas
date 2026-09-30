"""Интерпретация итоговой типологии одной командой:

    .venv/Scripts/python.exe -X utf8 scripts/run_interpret.py

Вход: outputs/labels/final.parquet (KEFRiN, K = 7), data/processed/*.parquet.
Выход: outputs/types.csv (label, name, description) и outputs/interpret/:
  mo_types.csv       модальный тип каждого МО и доля месяцев в нём;
  profiles.csv       профиль типа (доли, траты, рост, доступность, население, регионы, состав по типам МО);
  examples.csv       5 типичных и 3 пограничных МО на тип;
  mirkin.csv         профиль по Миркину (σ-отклонение и вклад в разброс);
  fca_rules.csv      правила «тип = набор признаков» из понятий FCA с точностью, покрытием и устойчивостью;
  fca_scale.csv      пороги квартилей шкалы;
  rosstat_kw.csv, rosstat_medians.csv   внешняя проверка Росстатом (Краскел–Уоллис, η²_H, медианы);
  TABLES.md          те же таблицы в виде для отчёта.
Росстат в кластеризации не участвовал.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import features as F  # noqa: E402
from src import interpret as P  # noqa: E402

OUT = ROOT / "outputs/interpret"
MOSCOW = (55.7558, 37.6173)  # центр Москвы, широта и долгота
FCA_COLS = ["mean_sh_grocery", "mean_sh_market", "mean_sh_food_out", "mean_sh_transport", "mean_sh_health",
            "mean_sh_other", "mean_log_level", "market_access", "trend_sh_market", "trend_log_level"]
SECTIONS = {"A": "сельское и лесное хозяйство", "B": "добыча", "C": "обрабатывающие производства",
            "F": "строительство", "G": "торговля", "H": "транспорт и хранение", "I": "гостиницы и общепит",
            "J": "информация и связь", "K": "финансы и страхование", "M": "профессиональная и научная",
            "O": "госуправление", "P": "образование", "Q": "здравоохранение"}
ROSSTAT_RU = {"wage_2023": "зарплата 2023, ₽", "wage_2024": "зарплата 2024, ₽",
              "wage_rel_region_2023": "зарплата 2023 к медиане МО региона",
              "employees_per_capita_2023": "работники на жителя 2023", "population": "население",
              **{f"emp_share_2023_{s}": f"доля занятых: {n}" for s, n in SECTIONS.items()}}

# Названия даны после чисел (TABLES.md). Проверки ниже падают, если прогон с другими метками
# перестанет им соответствовать.
NAMES = {
    0: ("Сельские районы с самыми низкими тратами: продукты выше среднего",
        "Муниципальные районы и округа с самыми низкими безналичными тратами на жителя ({spend} среднего). "
        "Продукты {groc} среднего, общепит {food}; доля маркетплейсов ниже средней, но растёт быстрее, чем в других типах."),
    1: ("Районы и малые города: транспорт и здоровье ниже среднего",
        "Районы и малые города, чаще северо-запада, центра и Урала, с самыми низкими долями транспорта ({transp} среднего) "
        "и здоровья ({health}); продукты и маркетплейсы выше среднего, траты {spend} среднего."),
    2: ("Сельские районы с самой высокой долей маркетплейсов",
        "Районы и округа Поволжья, Центра и юга, где маркетплейсы занимают больше всего ({mkt} среднего). "
        "Траты ниже средних ({spend}), общепит {food} среднего."),
    3: ("Москва, Подмосковье и соседние областные центры: самые высокие траты и доступность рынков",
        "Москва, почти всё Подмосковье, {capitals} областных центров вокруг Москвы (Тула, Рязань, Калуга, Тверь и др.) "
        "и пригородные районы крупных городов. Общее — доступность рынков в верхней четверти ({ma} при медиане "
        "{ma_all}); траты {spend} среднего, общепит {food}, продукты {groc}."),
    4: ("Север и Восток: высокие траты при худшей доступности рынков",
        "Территории Севера, Сибири и Дальнего Востока с худшей доступностью рынков ({ma} при медиане {ma_all}). "
        "Траты высокие ({spend} среднего), маркетплейсов меньше всего ({mkt}), «прочего» больше ({other})."),
    5: ("Петербург и крупные города: общепит и транспорт выше среднего",
        "Петербург, большинство региональных столиц ({capitals} из {capitals_all}) и их города-спутники: общепит {food} среднего, "
        "транспорт {transp}, траты {spend}. От типа Москвы отличается уровнем трат и доступностью рынков."),
    6: ("Районы Сибири: транспорт выше среднего, доступность рынков низкая",
        "Районы, в основном в Сибири ({east} % МО восточнее 60° в. д.): самая высокая доля транспорта "
        "({transp} среднего) при доступности рынков {ma} против {ma_all}. Самый слабо выраженный тип."),
}


def check_names(prof):
    """Каждое утверждение названия — проверка по профилю: другой прогон с другими номерами типов упадёт здесь."""
    p = prof.drop(index="all")
    assert p.spend_rub_geo.idxmin() == 0 and p.spend_rub_geo.idxmax() == 3
    assert p.sh_grocery_ratio[0] > 1.05
    assert p.sh_market_ratio.idxmax() == 2
    assert p.sh_transport_ratio.idxmin() == 1 and p.sh_health_ratio[1] < 0.9
    assert p.sh_food_out_ratio.idxmax() == 3 and p.sh_food_out_ratio[3] > 1.8
    assert p.regions[3].startswith("Москва") and p.regions[5].startswith("Санкт-Петербург")
    assert p.market_access_median.idxmax() == 3 and p.capitals[3] >= 5
    assert p.market_access_median.idxmin() == 4 and p.spend_ratio[4] > 1.1
    assert p.sh_food_out_ratio[5] > 1.5 and p.sh_transport_ratio[5] > 1.1 and p.capitals.idxmax() == 5
    assert p.sh_transport_ratio.idxmax() == 6 and p.market_access_median[6] < prof.market_access_median["all"]
    assert p.east60_pct[6] > 50 and p.east60_pct[4] > 50


def describe(prof, k):
    r, a = prof.loc[k], prof.loc["all"]
    c = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")  # noqa: E731
    v = dict(spend=c(r.spend_ratio), groc=c(r.sh_grocery_ratio), food=c(r.sh_food_out_ratio),
             mkt=c(r.sh_market_ratio), transp=c(r.sh_transport_ratio), health=c(r.sh_health_ratio),
             other=c(r.sh_other_ratio), ma=f"{r.market_access_median:.0f}", ma_all=f"{a.market_access_median:.0f}",
             capitals=f"{r.capitals:.0f}", capitals_all=f"{a.capitals:.0f}", east=f"{r.east60_pct:.0f}")
    return NAMES[k][1].format(**v)


def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "—" for c in cols) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(ROOT / "outputs/labels/final.parquet")
    mt = P.modal_type(lab)
    t = P.mo_table(ROOT)
    assert set(t.index) == set(mt.index), "метки и признаки о разных МО"
    types = mt["type"].reindex(t.index)
    t.join(mt)[["name", "region_name", "mo_type", "type", "stability"]].to_csv(OUT / "mo_types.csv")

    # 1. профили и примеры
    prof = P.profiles(t, types)
    prof.to_csv(OUT / "profiles.csv")
    Z = F.through_features(F.load_wide(ROOT))
    ex = P.examples(Z, types, t)
    ex.to_csv(OUT / "examples.csv", index=False)

    # 2. Миркин
    mk, T = P.mirkin(Z, types)
    mk.to_csv(OUT / "mirkin.csv", index=False)

    # 3. FCA
    X = t[FCA_COLS]
    I, names = P.scale_quartiles(X)
    X.quantile([0.25, 0.5, 0.75]).T.rename(columns=lambda q: f"q{int(q * 100)}").to_csv(OUT / "fca_scale.csv")
    rules, n_concepts = P.fca_rules(I, names, types.to_numpy())
    rules.to_csv(OUT / "fca_rules.csv", index=False)

    # 4. Росстат
    at = pd.read_parquet(ROOT / "data/processed/mo_attributes.parquet").reindex(t.index)
    R = pd.DataFrame({"wage_2023": at.wage_2023, "wage_2024": at.wage_2024,
                      "employees_per_capita_2023": at.employees_per_capita_2023,
                      "population": at.population_2024.fillna(at.population_2023)})
    R["wage_rel_region_2023"] = at.wage_2023 / at.wage_2023.groupby(t.region_name).transform("median")
    for s in SECTIONS:
        R[f"emp_share_2023_{s}"] = at[f"emp_share_2023_{s}"]
    kw, med = P.rosstat_check(R, types)
    kw.insert(1, "indicator_ru", kw.indicator.map(ROSSTAT_RU))
    kw.to_csv(OUT / "rosstat_kw.csv", index=False)
    med.to_csv(OUT / "rosstat_medians.csv")

    # столицы регионов: где они и чем различаются столицы разных типов
    cap = t[t.status == "административный_центр_субъекта"].assign(type=types, wage_2023=at.wage_2023)
    cap["km_to_moscow"] = P.km(cap.lat, cap.lon, *MOSCOW)
    cap[["name", "region_name", "type", "spend_rub", "wage_2023", "market_access", "population", "km_to_moscow",
         "mean_sh_food_out"]].sort_values(["type", "km_to_moscow"]).to_csv(OUT / "capitals.csv")

    # 5. названия
    check_names(prof)
    pd.DataFrame([{"label": k, "name": NAMES[k][0], "description": describe(prof, k)} for k in NAMES]).to_csv(
        ROOT / "outputs/types.csv", index=False)

    # TABLES.md
    name = {k: v[0] for k, v in NAMES.items()}
    p = prof.reset_index()
    p["тип"] = p["type"].map(lambda k: f"{k} {name.get(k, 'все МО')}".strip() if k != "all" else "все 2 016 МО")
    cols = ["тип", "n_mo"] + [f"{s}_pct" for s in P.SH] + ["spend_rub_geo", "growth_median_pct",
                                                          "market_access_median", "population_median"]
    f1 = {f"{s}_pct": "{:.1f}" for s in P.SH} | {"spend_rub_geo": "{:,.0f}", "growth_median_pct": "{:+.1f}",
                                                 "market_access_median": "{:.0f}", "population_median": "{:,.0f}",
                                                 "n_mo": "{:.0f}"}
    ratio = p[["тип"] + [f"{s}_ratio" for s in P.SH] + ["spend_ratio"]]
    parts = [f"# Таблицы интерпретации (scripts/run_interpret.py, {time.strftime('%d.%m.%Y')})",
             "## Профили (доли в %, траты — геом. среднее ₽/мес на жителя, рост — медиана 2024/2023, %)",
             md(p[cols], f1).replace(",", " "),
             "## Отношение к среднему по 2 016 МО", md(ratio, {c: "{:.2f}" for c in ratio.columns[1:]}),
             "## Регионы-лидеры и состав", md(p[["тип", "regions", "mo_type_mix", "capitals", "north60_pct", "east60_pct", "lat_median",
                                                 "lon_median"]].iloc[:-1], {"north60_pct": "{:.0f}", "east60_pct": "{:.0f}", "capitals": "{:.0f}",
                                                                            "lat_median": "{:.1f}",
                                                                            "lon_median": "{:.1f}"}),
             "## Примеры МО", md(ex, {"dist_own": "{:.2f}", "margin": "{:.2f}"})]
    top = mk.assign(a=mk.dev_sigma.abs()).sort_values(["type", "contrib"], ascending=[True, False])
    top = top.groupby("type").head(4)
    tot = mk.groupby("type").contrib.sum()
    parts += [f"## Миркин: 4 признака с наибольшим вкладом (T = {T:.0f}; объяснено всего {tot.sum():.3f})",
              md(top[["type", "feature_ru", "dev_sigma", "contrib"]], {"dev_sigma": "{:+.2f}", "contrib": "{:.4f}"}),
              "Вклад типа целиком: " + ", ".join(f"{k}: {v:.3f}" for k, v in tot.items()),
              f"## FCA: {I.shape[1]} атрибутов, {n_concepts} различных понятий из порождающих наборов до 3 атрибутов",
              md(rules.drop(columns="generator"), {c: "{:.3f}" for c in ["precision", "coverage", "f1", "sigma_low", "sigma_up"]}
                 | {"lift": "{:.2f}"}),
              "## Росстат: Краскел–Уоллис", md(kw, {"H": "{:.1f}", "p": "{:.1e}", "eta2_H": "{:.3f}"}),
              "## Росстат: медианы по типам", md(med.T.reset_index(), {c: "{:.3f}" for c in med.index})]
    (OUT / "TABLES.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    json.dump({"n_mo": len(t), "type_sizes": types.value_counts().sort_index().tolist(), "T_mirkin": T,
               "explained_share_mirkin": float(tot.sum()), "fca_attributes": I.shape[1],
               "fca_concepts": n_concepts,
               "spearman_spend_wage2023_mo": float(t.spend_rub.corr(at.wage_2023, method="spearman")),
               "spearman_spend_wage2023_types": float(prof.drop(index="all").spend_rub_geo.astype(float).corr(
                   med.drop(index="all").wage_2023, method="spearman")),
               "seconds": round(time.time() - t0, 1)},
              open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("ok", round(time.time() - t0, 1), "с")


if __name__ == "__main__":
    main()
