"""Базовые кластеры одной командой:

    .venv/Scripts/python.exe -X utf8 scripts/run_baseline.py [--config configs/baseline.yaml] [--quick]

Что делает:
  1. признаки (src/features.py): помесячные и сквозные;
  2. графы (src/network.py): kNN по косинусу долей — помесячно и сквозной;
  3. k-means и Leiden для K из k_range, все индексы ICVI и z (src/baseline.py, src/icvi.py);
  4. выбор K (Борда), метки → outputs/labels/{kmeans,leiden}.parquet (mo_id, month, cluster),
     все K → outputs/labels/{kmeans,leiden}_byK.parquet (mo_id, month, K, cluster);
  5. сравнение способов строить сеть на сквозном периоде → outputs/network_ablation.csv.
--quick: 3 месяца и мало перестановок, для проверки, что всё запускается; пишет в outputs/quick/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import features as F  # noqa: E402
from src import network as N  # noqa: E402
from src.baseline import (align, evaluate, kmeans_labels, leiden_labels, select_k,  # noqa: E402
                          sweep_period)
from src.icvi import pairwise_dist  # noqa: E402

SH = ["sh_health", "sh_market", "sh_food_out", "sh_grocery", "sh_transport", "sh_other"]
SHOW = ["SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "Q", "z_SW", "z_S_Dbw", "z_AVI", "z_AVU", "z_Q"]


def edge_jaccard(A, B):
    a, b = set(zip(*N.sp.triu(A, 1).nonzero())), set(zip(*N.sp.triu(B, 1).nonzero()))
    return len(a & b) / max(len(a | b), 1)


def network_ablation(cfg, w, X_all, shares_all, P, km_all, K_ld, K_km, n_perm):
    """Сравнение сетей на сквозном периоде: статистика графа, Leiden при γ=1 и при лучшем K,
    индексы k-means (лучшее K) на этом графе, согласие Leiden с k-means (ARI)."""
    nc, seed, variant = cfg["network"], cfg["seed"], cfg["icvi"]["s_dbw_variant"]
    k, top = nc["knn_k"], nc["threshold_top"]
    S_cos = N.cosine_sim(shares_all)
    S_corr, _ = N.lagcorr_sim(F.log_series(w)[0], nc["corr_max_lag"])
    S_combo = np.maximum(S_cos, 0) * P
    graphs = {"cosine_knn": N.knn_graph(S_cos, k), "cosine_thr": N.threshold_graph(S_cos, top),
              "lagcorr_knn": N.knn_graph(S_corr, k), "lagcorr_thr": N.threshold_graph(S_corr, top),
              "distance_knn": N.knn_graph(P, k), "combo_knn": N.knn_graph(S_combo, k),
              "combo_thr": N.threshold_graph(S_combo, top)}
    X = np.asarray(X_all)
    D = pairwise_dist(X)
    rows = []
    for name, A in graphs.items():
        st = N.graph_stats(A)
        ld1, _ = leiden_labels(A, None, seed, n_iterations=cfg["clustering"]["leiden"]["n_iterations"])
        row = dict(graph=name, **st, jaccard_vs_cosine_knn=edge_jaccard(A, graphs["cosine_knn"]),
                   leiden_g1_K=int(ld1.max() + 1), leiden_g1_Q=evaluate(None, A, ld1, None, 0, seed, variant)["Q"])
        if st["components"] <= K_ld:  # у порогового графа изоляты: каждый — свой кластер, K не набрать
            ldk, _ = leiden_labels(A, K_ld, seed, cfg["clustering"]["leiden"]["gamma_range"],
                                   cfg["clustering"]["leiden"]["bisect_steps"])
            e_ld = evaluate(X, A, ldk, D, n_perm, seed, variant)
            row.update(leiden_K=int(ldk.max() + 1), ARI_leiden_kmeans=adjusted_rand_score(km_all, ldk),
                       **{f"leiden_{c}": e_ld[c] for c in ("SW", "CH", "S_Dbw", "AVI", "AVU", "Q", "z_AVI", "z_AVU", "z_Q")})
        e_km = evaluate(X, A, km_all, D, n_perm, seed, variant)
        row.update(kmeans_K=K_km, **{f"kmeans_{c}": e_km[c] for c in ("AVI", "AVU", "Q", "z_AVI", "z_AVU", "z_Q")})
        rows.append(row)
    # DTW — на подвыборке МО (сравнение с корреляцией на той же подвыборке)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(X), min(nc["dtw"]["n_sub"], len(X)), replace=False))
    Dd = N.dtw_dist(F.log_series(w)[0][sub, -1, :], nc["dtw"]["window"])
    S_dtw = np.exp(-Dd / np.median(Dd[np.triu_indices(len(sub), 1)]))
    np.fill_diagonal(S_dtw, 0)
    A_dtw, A_corr_sub = N.knn_graph(S_dtw, k), N.knn_graph(S_corr[np.ix_(sub, sub)], k)
    ld1, _ = leiden_labels(A_dtw, None, seed)
    e_km = evaluate(X[sub], A_dtw, km_all[sub], None, n_perm, seed, variant)
    rows.append(dict(graph=f"dtw_knn_sub{len(sub)}", **N.graph_stats(A_dtw),
                     jaccard_vs_lagcorr_knn_sub=edge_jaccard(A_dtw, A_corr_sub),
                     leiden_g1_K=int(ld1.max() + 1), leiden_g1_Q=evaluate(None, A_dtw, ld1, None, 0, seed, variant)["Q"],
                     kmeans_K=K_km, **{f"kmeans_{c}": e_km[c] for c in ("AVI", "AVU", "Q", "z_AVI", "z_AVU", "z_Q")}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    nc, oc = cfg["network"], cfg["outputs"]
    t0 = time.time()

    w = F.load_wide(ROOT)
    clip = cfg["features"]["clip_z"]
    mf = F.monthly_features(w, clip)
    X_all = F.through_features(w, clip)
    ids = X_all.index
    months = sorted(mf)
    if args.quick:
        months = months[:3]
        cfg["icvi"]["n_perm_all"] = cfg["icvi"]["n_perm_month"] = 5
        # полные результаты не затираем: всё в outputs/quick/, как у run_compare.py
        oc = {k: "outputs/quick" if k == "labels_dir" else f"outputs/quick/{Path(v).name}" for k, v in oc.items()}
    assert all((mf[m].index == ids).all() for m in months)

    Dist = N.road_distance(ids, *N.mo_coords(ids, ROOT), nc["detour"])
    P, sigma = N.proximity(Dist, nc["knn_k"], nc["proximity_sigma"])

    def graph(shares):
        S = N.cosine_sim(shares)
        return N.knn_graph(np.maximum(S, 0) * P if nc["main"] == "combo" else S, nc["knn_k"])

    shares_all = np.hstack([mf[m][SH].to_numpy() for m in sorted(mf)])
    periods = [("all", X_all, graph(shares_all), cfg["icvi"]["n_perm_all"])]
    periods += [(m.strftime("%Y-%m"), mf[m][cfg["features"]["monthly_cols"]], graph(mf[m][SH].to_numpy()),
                 cfg["icvi"]["n_perm_month"]) for m in months]
    print(f"признаки и графы: {time.time() - t0:.0f} с; σ близости = {sigma:.0f} км", flush=True)

    res = Parallel(n_jobs=cfg["n_jobs"], max_nbytes="1M", mmap_mode="r")(delayed(sweep_period)(p, X, A, cfg, n) for p, X, A, n in periods)
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    labels = {p: lab for (p, *_), (_, lab) in zip(periods, res)}
    print(f"перебор K: {time.time() - t0:.0f} с", flush=True)

    best = select_k(df, cfg["icvi"]["k_select"])
    out_lab = ROOT / oc["labels_dir"]
    out_lab.mkdir(parents=True, exist_ok=True)
    for method in ("kmeans", "leiden"):
        ref = labels["all"][(method, best[method]["all"])]
        main_rows, byk = [], []
        for p, lab in labels.items():
            y = ref if p == "all" else align(lab[(method, best[method]["all"])], ref)  # K из 'all' во всех месяцах
            main_rows.append(pd.DataFrame({"mo_id": ids, "month": p, "cluster": y}))
            byk += [pd.DataFrame({"mo_id": ids, "month": p, "K": K, "cluster": v})
                    for (m, K), v in lab.items() if m == method]
        pd.concat(main_rows).to_parquet(out_lab / f"{method}.parquet", index=False)
        pd.concat(byk).to_parquet(out_lab / f"{method}_byK.parquet", index=False)

    df.to_csv(ROOT / oc["icvi_csv"], index=False)
    abl = network_ablation(cfg, w, X_all, shares_all, P, labels["all"][("kmeans", best["kmeans"]["all"])],
                           best["leiden"]["all"], best["kmeans"]["all"], cfg["icvi"]["n_perm_all"])
    abl.to_csv(ROOT / oc["network_csv"], index=False)
    json.dump({"best_k": best, "proximity_sigma_km": sigma, "k_select": cfg["icvi"]["k_select"],
               "network_main": nc["main"], "seconds": round(time.time() - t0)},
              open(ROOT / oc["best_k_json"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    a = df[df.period == "all"].set_index(["method", "K_target"])
    print("\n== Сквозные признаки ('all'): индексы по методам и K ==")
    print(a[["K", "min_size"] + SHOW].round(3).to_string())
    print("\n== Помесячно: медиана за месяцы ==")
    print(df[df.period != "all"].groupby(["method", "K_target"])[SHOW].median().round(3).to_string())
    print("\n== Лучшее K (Борда по", cfg["icvi"]["k_select"], ") ==", best)
    print("\n== Сети (сквозной период) ==")
    print(abl.round(3).T.to_string())
    print(f"\nготово за {time.time() - t0:.0f} с → {oc['icvi_csv']}, {oc['network_csv']}, {oc['labels_dir']}/")


if __name__ == "__main__":
    main()
