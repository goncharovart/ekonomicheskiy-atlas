"""Весь конвейер одной командой, по шагам:

    .venv/Scripts/python.exe -X utf8 scripts/run_all.py [--skip-download] [--quick] [--fresh]

download_data → build_panel → fetch_external → run_baseline → run_jury → run_compare →
run_interpret → run_dynamics (если скрипт есть) → check_mobility → prepare_geo → build_site.

После каждого шага проверяется, что он записал свои файлы: они есть и обновлены за время шага
(у download_data и fetch_external — только что есть: уже скачанное они не трогают). Если шаг
упал или файлов нет, конвейер останавливается с кодом 1. В конце — время каждого шага.
Перед тяжёлыми шагами (воркеры joblib, n_jobs в configs/*.yaml) ждёт, пока свободно ≥ 2 ГБ памяти.

--skip-download  не качать, data/raw уже на месте (проверяется, что нужные файлы есть);
--quick          проверка, что всё запускается (минуты вместо ~20): run_baseline, run_jury и
                 run_compare идут в укороченном режиме и пишут в outputs/quick/, полные
                 результаты в outputs/ не трогают; остальные шаги идут на полных результатах;
--fresh          пересчитать KEFRiN и CANUS в run_compare и бутстрэп в run_dynamics, не читая кэш.
"""
import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = "data/raw/hackathon/hackathonlicence/"

# скрипт, файлы, которые он обязан записать, пишет ли их каждый раз заново
STEPS = [
    ("download_data", [H + "consumption.parquet", H + "market_access.parquet", H + "connection.parquet",
                       "data/raw/borders/t_dict_municipal_districts.xlsx",
                       "data/raw/borders/t_dict_municipal_districts_poly.gpkg",
                       "data/raw/rosstat/Y48112027.csv.gz", "data/raw/rosstat/Y48423005.csv.gz",
                       "data/raw/rosstat/Y48423007.csv.gz"], False),
    ("build_panel", ["data/processed/panel.parquet", "data/processed/mo.parquet",
                     "data/processed/mo_attributes.parquet", "data/processed/rosstat_long.parquet",
                     "data/processed/build_summary.json", "data/geo/mo_simplified.geojson"], True),
    ("fetch_external", ["external/KEFRiN/kefrin.py", "external/CANUS/canus.py"], False),
    ("run_baseline", ["outputs/labels/kmeans_byK.parquet", "outputs/labels/leiden_byK.parquet",
                      "outputs/baseline_icvi.csv", "outputs/network_ablation.csv",
                      "outputs/baseline_best_k.json"], True),
    ("run_jury", ["outputs/labels/kefrin.parquet", "outputs/labels/canus.parquet",
                  "outputs/labels/pattern.parquet", "outputs/jury/jury_meta.json"], True),
    ("run_compare", ["outputs/labels/final.parquet", "outputs/labels/compare_byK.parquet",
                     "outputs/indices.csv", "outputs/method_ranking.csv", "outputs/compare_meta.json"], True),
    ("run_interpret", ["outputs/types.csv", "outputs/interpret/profiles.csv",
                       "outputs/interpret/summary.json"], True),
    ("run_dynamics", ["outputs/dynamics/summary.json", "outputs/dynamics/transitions_quarter.csv"], True),
    ("check_mobility", ["outputs/interpret/mobility_check.csv"], True),   # внешняя проверка индексом мобильности (СЗФО)
    ("prepare_geo", ["site/data/geo.json"], True),
    ("build_site", ["site/data/atlas.js", "site/data/meta.json"], True),
]
QUICK = {"run_baseline", "run_jury", "run_compare"}  # у них есть --quick с выходом в outputs/quick/
FRESH = {"run_compare", "run_dynamics"}              # у них есть --fresh
OPTIONAL = {"run_dynamics"}                          # нет скрипта — шаг пропускается
HEAVY = {"run_baseline", "run_jury", "run_compare", "run_dynamics"}  # перед ними ждём свободную память
MIN_FREE_GB = 2


def free_gb():
    """Доступная физическая память, ГБ; None — ОС не даёт узнать (тогда не ждём)."""
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("total", "avail", "tpage", "apage", "tvirt", "avirt", "aext")]
        st = Status(length=ctypes.sizeof(Status))
        return st.avail / 2**30 if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)) else None
    try:
        with open("/proc/meminfo") as f:
            return next(int(x.split()[1]) / 2**20 for x in f if x.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return None


def wait_memory(name):
    while (gb := free_gb()) is not None and gb < MIN_FREE_GB:
        print(f"== {name}: свободно {gb:.1f} ГБ памяти (нужно {MIN_FREE_GB}), жду 30 с", flush=True)
        time.sleep(30)


def check(name, files, since):
    """Пустой список — всё на месте; иначе — чего нет или что не обновилось."""
    bad = []
    for f in files:
        p = ROOT / f
        if not p.exists():
            bad.append(f"{f}: нет файла")
        elif since is not None and p.stat().st_mtime < since:
            bad.append(f"{f}: не обновлён шагом {name}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true", help="не качать, данные уже в data/raw")
    ap.add_argument("--quick", action="store_true", help="укороченный прогон тяжёлых шагов в outputs/quick/")
    ap.add_argument("--fresh", action="store_true", help="пересчитать кэши run_compare и run_dynamics")
    args = ap.parse_args()
    times, t_all = [], time.time()

    for name, files, rewrites in STEPS:
        script = ROOT / "scripts" / f"{name}.py"
        if name == "download_data" and args.skip_download:
            bad = check(name, files, None)
            if bad:
                sys.exit("--skip-download, но данных нет:\n  " + "\n  ".join(bad))
            print(f"== {name}: пропущен (--skip-download), данные на месте", flush=True)
            continue
        if name in OPTIONAL and not script.exists():
            print(f"== {name}: пропущен — нет scripts/{name}.py", flush=True)
            continue
        extra = []
        if args.quick and name in QUICK:
            extra.append("--quick")
            files = [f"outputs/quick/{Path(f).name}" for f in files]
        if args.fresh and name in FRESH:
            extra.append("--fresh")

        if name in HEAVY:
            wait_memory(name)
        print(f"\n== {name} {' '.join(extra)}".rstrip(), flush=True)
        t = time.time()
        rc = subprocess.run([sys.executable, "-X", "utf8", str(script), *extra], cwd=ROOT).returncode
        dt = time.time() - t
        times.append((name, dt))
        if rc:
            sys.exit(f"шаг {name} упал с кодом {rc}")
        bad = check(name, files, t - 1 if rewrites else None)  # секунда запаса на точность mtime
        if bad:
            sys.exit(f"шаг {name} не записал свои файлы:\n  " + "\n  ".join(bad))
        print(f"== {name}: {dt:.0f} с, файлы на месте", flush=True)

    print("\nВремя шагов:")
    for name, dt in times:
        print(f"  {name:<15} {dt:7.0f} с")
    print(f"  {'всего':<15} {time.time() - t_all:7.0f} с")


if __name__ == "__main__":
    main()
