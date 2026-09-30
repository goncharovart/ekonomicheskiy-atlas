"""Геометрия МО для сайта: site/data/geo.json.

Берёт data/geo/mo_simplified.geojson (его пишет build_panel.py): сбалансированные МО из
data/processed/mo.parquet (2 016 штук) помечает b=1, остальные МО панели (без полных
24 месяцев, на карте серым фоном) — b=0. Проецирует в
равновеликую коническую проекцию для России (км), упрощает и округляет до 0,1 км.

Координаты уже плоские, поэтому на сайте d3.geoIdentity без сферической
геометрии: нет проблем с порядком обхода колец и с Чукоткой за 180-м меридианом.

Запуск: .venv/Scripts/python.exe -X utf8 scripts/prepare_geo.py
"""
from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/geo/mo_simplified.geojson"
OUT = ROOT / "site/data/geo.json"
# Альберс для России: стандартные параллели 52 и 64, центральный меридиан 100 в.д.
CRS = "+proj=aea +lat_1=52 +lat_2=64 +lat_0=56 +lon_0=100 +datum=WGS84 +units=km +no_defs"


def main():
    mo = pd.read_parquet(ROOT / "data/processed/mo.parquet")
    keep = set(mo.index[mo["balanced"]])
    g = gpd.read_file(SRC)
    g = g.to_crs(CRS)
    # Мелкие МО (районы Москвы и Петербурга) упрощаем мягче, крупные сильнее.
    tol = (g.area ** 0.5 * 0.04).clip(upper=2.0)
    g["geometry"] = [shapely.simplify(geom, t, preserve_topology=True) for geom, t in zip(g.geometry, tol)]
    g["geometry"] = shapely.set_precision(g.geometry.values, 0.1)
    g = g[~g.geometry.is_empty]

    feats = []
    for r in g.itertuples():
        geom = json.loads(shapely.to_geojson(r.geometry))
        feats.append({"type": "Feature", "id": int(r.territory_id),
                      "properties": {"n": r.name, "r": r.region_name, "b": int(r.territory_id in keep)}, "geometry": geom})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps({"type": "FeatureCollection", "crs_proj4": CRS, "features": feats},
                     ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(txt, encoding="utf-8")
    missing = keep - {f["id"] for f in feats}
    print(f"geo.json: {len(feats)} МО (сбалансированных {len(keep) - len(missing)}), {len(txt) / 1e6:.2f} МБ, без геометрии: {sorted(missing)[:20]}")
    assert len(keep) - len(missing) >= 0.98 * len(keep), "потеряли геометрию у заметной доли МО"


if __name__ == "__main__":
    main()
