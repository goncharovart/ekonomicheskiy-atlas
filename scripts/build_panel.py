"""Собирает панель и справочник МО из data/raw в data/processed и data/geo.

Запуск: .venv/Scripts/python.exe -X utf8 scripts/build_panel.py

Выход:
  data/processed/panel.parquet          МО × месяц × категория (long): value, share
  data/processed/mo.parquet             справочник МО + население + покрытие панели + флаги границ
  data/processed/mo_attributes.parquet  атрибуты МО: доступность рынков, занятость и зарплаты по ОКВЭД2
  data/processed/rosstat_long.parquet   Росстат, привязанный к territory_id (год × показатель × раздел ОКВЭД2)
  data/geo/mo_simplified.geojson        упрощённые полигоны для карты (МО из панели)
  data/processed/build_summary.json     цифры для reports/DATA.md

Решение по границам (подробно — reports/DATA.md):
  territory_id СберИндекса — «МО в постоянных границах»: смена имени, типа и ОКТМО (район → округ)
  его не меняет, ряд не рвётся. Новый territory_id появляется только при объединении или разделении.
  Такие МО не склеиваем (средний чек на жителя без весов не суммируется), а помечаем флагом
  boundary_change_2023_2024; сбалансированная панель (balanced=True) — только МО со всеми 24 месяцами.
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, GEO = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "data" / "geo"
TOTAL, OTHER = "Все категории", "Прочее (расчёт)"
YEARS = (2023, 2024)
# В кодах разделов ОКВЭД2 у Росстата встречаются кириллические буквы (А, В, Е, Н) — приводим к латинице.
CYR2LAT = str.maketrans("АВЕНСКМОРТ", "ABEHCKMOPT")


def panel():
    c = pd.read_parquet(RAW / "hackathon/hackathonlicence/consumption.parquet")
    c = c.rename(columns={"date": "month"})
    c["month"] = pd.to_datetime(c["month"] + "-01")
    w = c.pivot_table(index=["territory_id", "month"], columns="category", values="value", aggfunc="first")
    parts = [x for x in w.columns if x != TOTAL]
    w[OTHER] = w[TOTAL] - w[parts].sum(axis=1)  # «Все категории» включает и иные категории, их остаток
    p = pd.concat({"value": w.stack(), "share": w.div(w[TOTAL], axis=0).stack()}, axis=1).reset_index()
    p["territory_id"] = p["territory_id"].astype("int32")
    return p.sort_values(["territory_id", "month", "category"]).reset_index(drop=True), parts


def directory(p):
    d = pd.read_excel(RAW / "borders/t_dict_municipal_districts.xlsx", dtype={"oktmo": str})
    d["oktmo8"] = d["oktmo"].str.replace("-", "").str[:8]
    ids = d.groupby("territory_id").agg(year_from=("year_from", "min"), year_to=("year_to", "max"),
                                        n_versions=("oktmo", "size"), oktmo_all=("oktmo8", lambda s: ",".join(sorted(set(s)))))
    last = d.sort_values("year_from").groupby("territory_id").tail(1).set_index("territory_id")  # последняя версия
    cols = ["oktmo8", "municipal_district_name", "municipal_district_name_short", "municipal_district_type",
            "municipal_district_status", "region_code", "region_name", "municipal_district_center",
            "municipal_district_center_lat", "municipal_district_center_lon", "change_id_from", "change_id_to"]
    mo = ids.join(last[cols]).rename(columns={"oktmo8": "oktmo", "municipal_district_name": "name",
                                              "municipal_district_name_short": "name_short",
                                              "municipal_district_type": "mo_type", "municipal_district_status": "status",
                                              "municipal_district_center": "center", "municipal_district_center_lat": "lat",
                                              "municipal_district_center_lon": "lon"})
    cov = p[p.category == TOTAL].groupby("territory_id")["month"].agg(n_months="nunique", first_month="min", last_month="max")
    mo = mo.join(cov)
    mo["in_panel"] = mo["n_months"].notna()
    mo["n_months"] = mo["n_months"].fillna(0).astype(int)
    mo["balanced"] = mo["n_months"] == p["month"].nunique()
    # объединение/разделение внутри окна данных: МО появилось в 2024 или перестало существовать до 2024
    mo["boundary_change_2023_2024"] = (mo["year_from"] == 2024) | (mo["year_to"] == 2024)
    mo.index = mo.index.astype("int32")
    return mo, d


def rosstat(d):
    """Росстат по ОКТМО → territory_id через версию справочника, действующую в этом году."""
    frames = []
    for code, name in [("Y48112027", "population"), ("Y48423005", "employees"), ("Y48423007", "wage")]:
        r = pd.read_csv(RAW / f"rosstat/{code}.csv.gz", sep=";", dtype=str)
        r = r[r["mun_level"] == "Муниципальное образование верхнего уровня"]
        if name == "population":
            r = r[(r["indicator_period"] == "На 1 января") & (r["mest"] == "Все население")].assign(okved2="Всего")
        else:
            r = r[r["indicator_period"] == "Январь-декабрь"]
        r = r.assign(indicator=name, year=r["year"].astype(int), value=pd.to_numeric(r["indicator_value"], errors="coerce"))
        frames.append(r[["oktmo", "oktmo_stable", "year", "indicator", "okved2", "value"]].dropna(subset=["value"]))
    r = pd.concat(frames)
    sec = r["okved2"].str.extract(r"^Раздел\s+(\S)")[0].str.translate(CYR2LAT)
    r["okved_section"] = sec.fillna("TOTAL")
    # точное совпадение по году; иначе — любая версия с этим ОКТМО (или oktmo_stable Росстата), если она однозначна
    v = d[["oktmo8", "territory_id", "year_from", "year_to"]]
    m = r.merge(v, left_on="oktmo", right_on="oktmo8", how="left")
    exact = m[(m.year >= m.year_from) & (m.year < m.year_to)]
    rest = r.loc[~r.set_index(["oktmo", "year", "indicator", "okved2"]).index.isin(
        exact.set_index(["oktmo", "year", "indicator", "okved2"]).index)]
    uniq = v.groupby("oktmo8")["territory_id"].nunique()
    one = v.drop_duplicates("oktmo8").loc[lambda x: x.oktmo8.isin(uniq[uniq == 1].index)].set_index("oktmo8")["territory_id"]
    # ОКТМО, сменившиеся в 2024 (после среза справочника), ловим по «стабильному» ОКТМО самого Росстата
    tid = rest["oktmo"].map(one).fillna(rest["oktmo_stable"].map(one))
    fb = rest.assign(territory_id=tid).dropna(subset=["territory_id"])
    out = pd.concat([exact.assign(match="year"), fb.assign(match="any_version")])
    stats = {"rows_total": len(r), "rows_matched_year": len(exact), "rows_matched_fallback": len(fb),
             "rows_unmatched": len(r) - len(exact) - len(fb),
             "oktmo_unmatched": sorted(set(r.oktmo) - set(out.oktmo))[:30]}
    out = out[["territory_id", "year", "indicator", "okved_section", "okved2", "value", "oktmo", "match"]]
    dup = out.duplicated(["territory_id", "year", "indicator", "okved_section"], keep=False)
    stats["territory_year_collisions"] = int(dup.sum())
    # ponytail: коллизии (два ОКТМО на один territory_id в году) редки — берём строку с точным совпадением года
    out = out.sort_values("match", key=lambda s: s != "year").drop_duplicates(["territory_id", "year", "indicator", "okved_section"])
    out["territory_id"] = out["territory_id"].astype("int32")
    return out.reset_index(drop=True), stats


def attributes(rl, mo):
    a = pd.DataFrame(index=mo.index)
    ma = pd.read_parquet(RAW / "hackathon/hackathonlicence/market_access.parquet").set_index("territory_id")
    a["market_access_2024"] = ma["market_access"]
    for y in YEARS:
        t = rl[(rl.year == y) & (rl.okved_section == "TOTAL")].pivot_table(index="territory_id", columns="indicator", values="value")
        for ind in ("population", "employees", "wage"):
            if ind in t:
                a[f"{ind}_{y}"] = t[ind]
    y = 2023  # последний год с полным охватом отраслей (2024 у Росстата заметно беднее)
    e = rl[(rl.year == y) & (rl.indicator == "employees") & (rl.okved_section != "TOTAL")]
    e = e.pivot_table(index="territory_id", columns="okved_section", values="value")
    a = a.join(e.div(a[f"employees_{y}"], axis=0).add_prefix(f"emp_share_{y}_"))
    wg = rl[(rl.year == y) & (rl.indicator == "wage") & (rl.okved_section != "TOTAL")]
    a = a.join(wg.pivot_table(index="territory_id", columns="okved_section", values="value").add_prefix(f"wage_{y}_"))
    a["employees_per_capita_2023"] = a["employees_2023"] / a["population_2023"]
    a.index.name = "territory_id"
    return a


def geo(mo):
    g = gpd.read_file(RAW / "borders/t_dict_municipal_districts_poly.gpkg")
    g["territory_id"] = g["territory_id"].astype(int)
    g = g[g.territory_id.isin(mo.index[mo.in_panel])]
    g["geometry"] = g.geometry.simplify(0.01, preserve_topology=True)  # ~1 км, для карты хватает
    g = g.merge(mo[["name", "region_name", "mo_type"]], left_on="territory_id", right_index=True)
    g[["territory_id", "name", "region_name", "mo_type", "geometry"]].to_file(GEO / "mo_simplified.geojson", driver="GeoJSON")
    return len(g)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GEO.mkdir(parents=True, exist_ok=True)
    p, parts = panel()
    mo, d = directory(p)
    rl, rstats = rosstat(d)
    a = attributes(rl, mo)
    mo = mo.join(a[["population_2023", "population_2024"]])
    p.to_parquet(OUT / "panel.parquet", index=False)
    mo.to_parquet(OUT / "mo.parquet")
    a.to_parquet(OUT / "mo_attributes.parquet")
    rl.to_parquet(OUT / "rosstat_long.parquet", index=False)
    n_geo = geo(mo)
    inp = mo[mo.in_panel]
    summary = {
        "panel_rows": len(p), "mo_in_panel": int(inp.shape[0]), "mo_balanced": int(inp.balanced.sum()),
        "months": p.month.nunique(), "period": [str(p.month.min().date()), str(p.month.max().date())],
        "categories": sorted(p.category.unique()), "base_categories": parts,
        "n_months_hist": inp.n_months.value_counts().sort_index().to_dict(),
        "mo_in_dictionary": int(len(mo)), "dictionary_versions": int(len(d)),
        "boundary_change_in_panel": inp.index[inp.boundary_change_2023_2024].tolist(),
        "mo_multi_version_in_panel": int((inp.n_versions > 1).sum()),
        "other_share": p[p.category == OTHER].share.describe().round(4).to_dict(),
        "other_negative": int((p[p.category == OTHER].value < 0).sum()),
        "mo_type_in_panel": inp.mo_type.value_counts().to_dict(),
        "regions_in_panel": int(inp.region_code.nunique()),
        "population_2024_missing_in_panel": int(inp.population_2024.isna().sum()),
        "attr_missing_in_panel": a.loc[inp.index].isna().sum().loc[lambda s: s > 0].to_dict(),
        "market_access_missing_in_panel": int(a.loc[inp.index, "market_access_2024"].isna().sum()),
        "geo_features": n_geo, "rosstat": rstats,
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "attr_missing_in_panel"}, ensure_ascii=False, default=str, indent=1))


if __name__ == "__main__":
    main()
