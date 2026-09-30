"""Индекс мобильности СберИндекса как внешняя проверка типов (в кластеризацию он не входит):

    .venv/Scripts/python.exe -X utf8 scripts/check_mobility.py

Вход: data/raw/api/indeks-mobilnosti.parquet (скачивает scripts/download_data.py), типы МО —
outputs/interpret/mo_types.csv (scripts/run_interpret.py).
Выгрузка API — одно годовое значение в км на МО за 2024 и 2025 годы, и только по МО Северо-Западного
федерального округа. Кода МО в ней нет, только название, поэтому название сопоставляется со справочником
внутри регионов СЗФО: по всей стране названия вроде «Кировский муниципальный район» неоднозначны.
Выход: outputs/interpret/mobility_check.csv — медианы по типам на пересечении с 2 016 МО типологии и
критерий Краскела–Уоллиса с η²_H (строка group = all), как у проверки Росстатом; то же за 2024 год
без внутригородских территорий Петербурга.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import interpret as P  # noqa: E402

SZFO = ["Архангельская область", "Ненецкий автономный округ", "Вологодская область", "Калининградская область",
        "Ленинградская область", "Мурманская область", "Новгородская область", "Псковская область",
        "Республика Карелия", "Республика Коми", "Санкт-Петербург"]
YEARS = {"2024-12-31": "km_2024", "2025-11-01": "km_2025"}


def main():
    raw = pd.read_parquet(ROOT / "data/raw/api/indeks-mobilnosti.parquet")
    assert set(raw.unit_measure) == {"км"} and set(raw.period) == set(YEARS), "выгрузка API поменялась"
    raw["name"] = raw.ref_area.str.split().str.join(" ")
    mo = pd.read_parquet(ROOT / "data/processed/mo.parquet")
    mo = mo[mo.region_name.isin(SZFO)].assign(name=lambda d: d.name.str.split().str.join(" "))
    assert mo.name.is_unique, "названия МО внутри СЗФО неоднозначны"
    tid = pd.Series(mo.index, index=mo.name)
    missing = sorted(set(raw.name) - set(tid.index))
    assert len(missing) <= 1, f"не нашлись в справочнике: {missing}"  # 30.09: одно имя, в справочнике такого МО нет
    raw = raw[raw.name.isin(tid.index)]
    X = raw.assign(territory_id=raw.name.map(tid), col=raw.period.map(YEARS)).pivot(
        index="territory_id", columns="col", values="value")

    types = pd.read_csv(ROOT / "outputs/interpret/mo_types.csv", index_col="territory_id")["type"]
    X = X[X.index.isin(types.index)]
    # треть пересечения (97 из 274) — внутригородские территории Петербурга; без них проверка строже
    X["km_2024_without_spb"] = X.km_2024.where(mo.region_name.reindex(X.index) != "Санкт-Петербург")
    kw, med = P.rosstat_check(X, types)
    n = X.notna().groupby(types.reindex(X.index)).sum()
    n.loc["all"] = X.notna().sum()
    out = med.rename_axis("group").reset_index()
    for c in X.columns:
        out.insert(out.columns.get_loc(c), f"n_{c}", out.group.map(n[c]))
    for _, r in kw.iterrows():
        out.loc[out.group == "all", [f"H_{r.indicator}", f"p_{r.indicator}", f"eta2_H_{r.indicator}"]] = [r.H, r.p, r.eta2_H]
    out.to_csv(ROOT / "outputs/interpret/mobility_check.csv", index=False)
    print(f"МО в выгрузке {raw.ref_area.nunique() + len(missing)}, нет в справочнике: {missing}; "
          f"в типологии {len(X)} из {len(types)}")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
