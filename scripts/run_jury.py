"""Методы жюри одной командой: KEFRiN, CANUS, паттерн-кластеризация.

.venv/Scripts/python.exe -X utf8 scripts/run_jury.py [--config configs/jury.yaml] [--quick]
Метки: outputs/labels/{kefrin,canus,pattern}.parquet — (mo_id, month 'YYYY-MM' или 'all', cluster).
Прочее: outputs/jury/ (словарь паттернов, устойчивость МО, время и число кластеров).
--quick: 2 месяца, n_init 2, 10 эпох CANUS — проверка, что всё запускается; пишет в outputs/quick/.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import jury_methods as jm  # noqa: E402

LABELS, OUT = ROOT / "outputs" / "labels", ROOT / "outputs" / "jury"


def frame(ids, month, labels):
    return pd.DataFrame({"mo_id": ids, "month": month, "cluster": np.asarray(labels, int)})


def run_graph_methods(X, cfg, months):
    """KEFRiN и CANUS по каждому месяцу и по средним за период ('all')."""
    snaps = [(m.strftime("%Y-%m"), X.xs(m, level="month")) for m in months]
    snaps.append(("all", X.groupby(level="territory_id").mean()))
    out, secs = {"kefrin": [], "canus": []}, {"kefrin": 0.0, "canus": 0.0}
    canus_cfg = dict(cfg["canus"])
    fast = canus_cfg.pop("fast_grad_norms", True)
    for name, S in snaps:
        Y, ids = S.to_numpy(), S.index.to_numpy()
        A = jm.knn_graph(Y, cfg["knn_k"])
        t = time.time()
        out["kefrin"].append(frame(ids, name, jm.kefrin(Y, A, seed=cfg["seed"], **cfg["kefrin"])))
        secs["kefrin"] += time.time() - t
        t = time.time()
        out["canus"].append(frame(ids, name, jm.canus(jm.zscore(Y), A, fast_grad_norms=fast, seed=cfg["seed"], **canus_cfg)))
        secs["canus"] += time.time() - t
        print(f"{name}: kefrin {np.bincount(out['kefrin'][-1].cluster)}, canus {np.bincount(out['canus'][-1].cluster)}", flush=True)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}, secs


def run_pattern(X, cfg, mode):
    pc = cfg["pattern"]
    P = X.rank(pct=True) if pc["normalize"] == "percentile" else X
    mean = P.groupby(level="territory_id").mean()
    codes = jm.pattern_codes(np.vstack([P.to_numpy(), mean.to_numpy()]), mode, pc["eps"])
    is_month = np.r_[np.ones(len(P)), np.zeros(len(mean))]
    lab, uniq = jm.pattern_clusters(codes, count_mask=is_month)
    ids = np.r_[P.index.get_level_values("territory_id"), mean.index]
    month = np.r_[P.index.get_level_values("month").strftime("%Y-%m"), ["all"] * len(mean)]
    labels = frame(ids, month, lab)
    monthly = labels[labels.month != "all"]
    names = list(X.columns)
    dictionary = pd.DataFrame({
        "cluster": range(len(uniq)),
        "pattern": [jm.pattern_text(c, names, mode) for c in uniq],
        "code": [" ".join(map(str, c)) for c in uniq],
        "n_mo_months": np.bincount(monthly.cluster, minlength=len(uniq)),
    })
    return labels, dictionary, jm.pattern_stability(monthly)


def commit(name):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT / "external" / name,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "jury.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    labels_dir, out_dir = LABELS, OUT
    if args.quick:
        cfg["months"], cfg["kefrin"]["n_init"], cfg["canus"]["epochs"] = 2, 2, 10
        labels_dir = out_dir = ROOT / "outputs" / "quick"
    labels_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["seed"])

    X = jm.load_shares(cfg["categories"])
    months = X.index.get_level_values("month").unique().sort_values()
    if cfg.get("months", "all") != "all":
        months = months[: int(cfg["months"])]
        X = X[X.index.get_level_values("month").isin(months)]
    meta = {"n_mo": int(X.index.get_level_values("territory_id").nunique()), "n_months": len(months),
            "external_commits": {n: commit(n) for n in ("KEFRiN", "CANUS")}, "seconds": {}, "n_clusters": {}}

    for i, mode in enumerate(cfg["pattern"]["modes"]):
        t = time.time()
        labels, dictionary, stab = run_pattern(X, cfg, mode)
        key = "pattern" if i == 0 else f"pattern_{mode}"
        labels.to_parquet(labels_dir / f"{key}.parquet", index=False)
        dictionary.to_csv(out_dir / f"{key}_dictionary.csv", index=False, encoding="utf-8")
        stab.to_parquet(out_dir / f"{key}_stability.parquet", index=False)
        meta["seconds"][key] = round(time.time() - t, 1)
        nm = labels[labels.month != "all"]
        meta["n_clusters"][key] = {"distinct_patterns": int(nm.cluster.nunique()),
                                   "per_month_median": float(nm.groupby("month").cluster.nunique().median()),
                                   "top8_coverage": float(dictionary.n_mo_months.head(8).sum() / len(nm))}
        meta[f"{key}_stability"] = {"never_changed": int((stab.n_changes == 0).sum()),
                                    "median_changes": float(stab.n_changes.median()),
                                    "median_modal_share": float(stab.modal_share.median())}
        print(key, meta["n_clusters"][key], meta[f"{key}_stability"], flush=True)

    res, secs = run_graph_methods(X, cfg, months)
    for key, labels in res.items():
        labels.to_parquet(labels_dir / f"{key}.parquet", index=False)
        meta["seconds"][key] = round(secs[key], 1)
        nm = labels[labels.month != "all"]
        meta["n_clusters"][key] = {"k": cfg[key]["n_clusters"],
                                   "nonempty_per_month_min": int(nm.groupby("month").cluster.nunique().min()),
                                   "min_cluster_size": int(nm.groupby(["month", "cluster"]).size().min())}
    (out_dir / "jury_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
