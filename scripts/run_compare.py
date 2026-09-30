"""Сравнение восьми методов на общих признаках и общем графе одной командой:

    .venv/Scripts/python.exe -X utf8 scripts/run_compare.py [--config configs/compare.yaml] [--quick] [--fresh]

  1. общие входы: признаки src/features.py (сквозные и помесячные), граф src/network.py
     (kNN по корреляции рядов, один на весь период);
  2. метки: k-means — из outputs/labels/kmeans_byK.parquet (те же признаки, run_baseline.py);
     Ward и GMM — заново на тех же признаках (дёшево, без кэша); Leiden и спектральная — заново
     на общем графе; KEFRiN и CANUS — заново на общих признаках и графе
     (кэш outputs/labels/jury_common_byK.parquet, --fresh пересчитывает); паттерны — слияние до K;
  3. шесть индексов + z против перестановок: сквозные для K из k_range, помесячно для k_common
     и лучшего K метода (Борда по k_select на сквозных, как в run_baseline.py);
  4. outputs/indices.csv, outputs/method_ranking.csv (пороговое агрегирование и Борда),
     устойчивость по месяцам, итоговая типология outputs/labels/final.parquet (номера месяцев
     согласованы под модальный тип МО, src/dynamics.consensus_align).
--quick: 2 месяца, K из k_common, мало перестановок и эпох, без кэша; пишет в outputs/quick/.
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
from src import compare as C  # noqa: E402
from src import features as F  # noqa: E402
from src import jury_methods as jm  # noqa: E402
from src import network as N  # noqa: E402
from src.baseline import align, borda_best_k, evaluate, gmm_labels, leiden_labels, spectral_labels, ward_labels  # noqa: E402
from src.dynamics import consensus_align  # noqa: E402
from src.icvi import pairwise_dist  # noqa: E402

METHODS = ["kmeans", "leiden", "ward", "gmm", "spectral", "kefrin", "canus", "patterns"]
FAMILY = {m: "baseline" for m in ("kmeans", "leiden", "ward", "gmm", "spectral")} |          {m: "jury" for m in ("kefrin", "canus", "patterns")}
NAME = {"kmeans": "k-means по признакам", "leiden": "Leiden по графу",
        "ward": "Ward (агломеративная) по признакам", "gmm": "GMM (диагональная ковариация) по признакам",
        "spectral": "Спектральная по графу",
        "kefrin": "KEFRiN (Шалилех, Миркин)", "canus": "CANUS (Шалилех)",
        "patterns": "Паттерны (Алескеров, Мячин), слияние до K"}
FEAT = {"ward": ward_labels, "gmm": gmm_labels}  # на тех же признаках, что k-means; каждый запуск < 1 с
ALIGN = {"kmeans", "ward", "gmm", "kefrin", "canus"}  # номера по месяцам случайны; у графовых и паттернов общие
IDX = ["SW", "CH", "S_Dbw", "AVI", "AVU", "Q"]
ZIDX = [f"z_{c}" for c in IDX]
AVU_NOTE = ("AVU участвует с оговоркой: в формуле лаборатории он зависит только от формы разреза, "
            "не от его доли в весе графа (z у хороших разбиений бывает «хуже случайного»); "
            "место без AVU — threshold_place_without_AVU")


def joint_labels(method, Y, A, K, jcfg, seed):
    import torch
    torch.set_num_threads(1)
    if method == "kefrin":
        p = {k: v for k, v in jcfg["kefrin"].items() if k != "n_clusters"}
        return jm.kefrin(Y, A, n_clusters=K, seed=seed, **p)
    p = {k: v for k, v in jcfg["canus"].items() if k not in ("n_clusters", "fast_grad_norms")}
    return jm.canus(Y, A, n_clusters=K, fast_grad_norms=jcfg["canus"].get("fast_grad_norms", True), seed=seed, **p)


def long(labels, ids):
    return pd.concat([pd.DataFrame({"method": m, "period": p, "K": K, "mo_id": ids, "cluster": np.asarray(v, int)})
                      for (m, p, K), v in labels.items()], ignore_index=True)


def unlong(df, ids):
    return {(m, p, int(K)): g.set_index("mo_id").cluster.reindex(ids).to_numpy()
            for (m, p, K), g in df.groupby(["method", "period", "K"])}


def ensure_joint(L, need, X, Ad, jcfg, seed, n_jobs):
    todo = sorted(k for k in need if k not in L)
    if todo:
        t = time.time()
        res = Parallel(n_jobs=n_jobs, max_nbytes="1M", mmap_mode="r")(delayed(joint_labels)(m, X[p], Ad, K, jcfg, seed) for m, p, K in todo)
        L.update(zip(todo, res))
        print(f"KEFRiN/CANUS: {len(todo)} запусков за {time.time() - t:.0f} с", flush=True)
    return L


def ensure_feat(L, need, X, seed, n_jobs):
    todo = sorted(k for k in need if k not in L)
    L.update(zip(todo, Parallel(n_jobs=n_jobs)(delayed(FEAT[m])(X[p], K, seed) for m, p, K in todo)))


def pattern_labels(jcfg, ids, Ks):
    """Паттерны полного порядка (как run_jury.py), слитые до каждого K; номера общие для месяцев."""
    pc = jcfg["pattern"]
    assert pc["normalize"] == "percentile"
    S = jm.load_shares(jcfg["categories"])
    P = S.rank(pct=True)
    mean = P.groupby(level="territory_id").mean()
    Z = np.vstack([P.to_numpy(), mean.to_numpy()])
    lab, uniq = jm.pattern_clusters(jm.pattern_codes(Z, "pairs", pc["eps"]),
                                    count_mask=np.r_[np.ones(len(P)), np.zeros(len(mean))])
    maps = C.merge_patterns(uniq, Z, lab, min(Ks))
    nm = len(P)
    idx = pd.MultiIndex.from_arrays([P.index.get_level_values("territory_id"),
                                     P.index.get_level_values("month").strftime("%Y-%m")])
    out = {}
    for K in Ks:
        wide = pd.Series(maps[K][lab[:nm]], index=idx).unstack().reindex(ids)
        out.update({("patterns", p, K): wide[p].to_numpy() for p in wide.columns})
        out[("patterns", "all", K)] = pd.Series(maps[K][lab[nm:]], index=mean.index).reindex(ids).to_numpy()
    return out, len(uniq)


def eval_job(key, X, A, lab, n_perm, seed, variant):
    m, p, K = key
    X = np.asarray(X)
    r = evaluate(X, A, lab, pairwise_dist(X), n_perm, seed, variant)
    return dict(method=m, family=FAMILY[m], period=p, K_target=K, K=int(len(np.unique(lab))),
                min_size=int(np.unique(lab, return_counts=True)[1].min()), **r)


def aligned_wide(L, method, K, months):
    ref = L[(method, "all", K)]
    cols = [align(L[(method, p, K)], ref) if method in ALIGN else L[(method, p, K)] for p in months]
    return np.stack(cols, 1)


def ranking(ind, cols, n_grades):
    rows = []
    for (variant, period), g in ind.groupby(["variant", "period"], sort=False):
        g = g.set_index("method")
        G = C.grades(g, cols, n_grades)
        no_avu = [c for c in cols if c.removeprefix("z_") != "AVU"]
        pts, bplace = C.borda(g, cols)
        out = pd.DataFrame({"variant": variant, "period": period, "method": g.index, "name": g["name"],
                            "family": g["family"], "K": g["K"]})
        for c in cols:
            out[f"grade_{c}"] = G[c]
        for k in range(1, n_grades + 1):
            out[f"n_grade{k}"] = (G == k).sum(axis=1)
        out["threshold_place"] = C.threshold_places(G, n_grades)
        out["threshold_place_without_AVU"] = C.threshold_places(G[no_avu], n_grades)
        out["borda_points"] = pts
        out["borda_place"] = bplace
        out["borda_place_without_AVU"] = C.borda(g, no_avu)[1]
        for c in ("threshold_place", "borda_place"):  # тот же ли лидер блока без AVU
            out[f"{c}_leader_same_without_AVU"] = set(out.index[out[c] == 1]) == set(out.index[out[f"{c}_without_AVU"] == 1])
        out["note"] = AVU_NOTE
        rows.append(out.sort_values(["threshold_place", "borda_place"]))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/compare.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="пересчитать KEFRiN и CANUS, не читая кэш")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    bcfg = yaml.safe_load(open(ROOT / cfg["baseline_config"], encoding="utf-8"))
    jcfg = yaml.safe_load(open(ROOT / cfg["jury_config"], encoding="utf-8"))
    seed, nj, oc = cfg["seed"], cfg["n_jobs"], cfg["outputs"]
    ic, nc, lc = bcfg["icvi"], bcfg["network"], bcfg["clustering"]["leiden"]
    variant, n_perm_all, n_perm_month = ic["s_dbw_variant"], ic["n_perm_all"], ic["n_perm_month"]
    k_common, ksel = cfg["k_common"], cfg["k_select"]
    Ks_all = list(range(bcfg["clustering"]["k_range"][0], bcfg["clustering"]["k_range"][1] + 1))
    out = (lambda p: ROOT / "outputs/quick" / Path(p).name) if args.quick else (lambda p: ROOT / p)
    if args.quick:
        (ROOT / "outputs/quick").mkdir(exist_ok=True)
    t0 = time.time()

    w = F.load_wide(ROOT)
    clip = bcfg["features"]["clip_z"]
    mf = F.monthly_features(w, clip)
    X_all = F.through_features(w, clip)
    ids = X_all.index.to_numpy()
    months = sorted(mf)[:2] if args.quick else sorted(mf)
    if args.quick:
        Ks_all, n_perm_all, n_perm_month = list(k_common), 3, 3
        jcfg["kefrin"]["n_init"], jcfg["canus"]["epochs"] = 2, 10
    X = {"all": X_all.to_numpy()}
    X.update({m.strftime("%Y-%m"): mf[m].loc[ids, bcfg["features"]["monthly_cols"]].to_numpy() for m in months})
    mkeys = list(X)[1:]
    A = C.common_graph(w, nc["knn_k"], nc["corr_max_lag"])
    Ad = A.toarray()
    gstats = N.graph_stats(A)
    print(f"входы: {time.time() - t0:.0f} с; граф {cfg['graph']}: {gstats}", flush=True)

    # почему growth не в помесячных признаках (baseline.yaml): темп соседних месяцев в рангах
    g = F.monthly_raw(w)["growth"].unstack("month").iloc[:, 1:]  # у 2023-01 прошлого месяца нет
    rho = pd.Series([g.iloc[:, t].corr(g.iloc[:, t + 1], method="spearman") for t in range(g.shape[1] - 1)],
                    index=[m.strftime("%Y-%m") for m in g.columns[1:]])
    diag = {"growth_rank_corr_adjacent_months": {
        "what": "Спирмен между Δln(трат на жителя) месяца t−1 и t по 2 016 МО, пары 2023-02/03 … 2024-11/12",
        "mean": round(float(rho.mean()), 3), "median": round(float(rho.median()), 3),
        "min": round(float(rho.min()), 3), "max": round(float(rho.max()), 3), "by_month": rho.round(3).to_dict()},
        "common_graph": dict(name=cfg["graph"], **gstats)}
    json.dump(diag, open(out(oc["diagnostics"]), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("темп соседних месяцев, Спирмен:", diag["growth_rank_corr_adjacent_months"]["mean"], flush=True)

    # --- метки ---
    km = pd.read_parquet(ROOT / bcfg["outputs"]["labels_dir"] / "kmeans_byK.parquet")
    km = km[km.month.isin(list(X)) & km.K.isin(set(Ks_all) | set(k_common))].rename(columns={"month": "period"})
    L = unlong(km.assign(method="kmeans"), ids)
    ld = Parallel(n_jobs=nj, max_nbytes="1M", mmap_mode="r")(delayed(leiden_labels)(A, K, seed, lc["gamma_range"], lc["bisect_steps"],
                                                     lc["n_iterations"]) for K in Ks_all)
    gammas = {}
    for K, (lab, gamma) in zip(Ks_all, ld):
        gammas[K] = gamma
        L.update({("leiden", p, K): lab for p in X})  # граф один на весь период — разбиение тоже
    sp = Parallel(n_jobs=nj)(delayed(spectral_labels)(A, K, seed) for K in Ks_all)
    for K, lab in zip(Ks_all, sp):
        L.update({("spectral", p, K): lab for p in X})
    ensure_feat(L, {(m, "all", K) for m in FEAT for K in Ks_all} | {(m, p, K) for m in FEAT for p in mkeys for K in k_common},
                X, seed, nj)
    pl, n_patterns = pattern_labels(jcfg, ids, Ks_all)
    L.update({k: v for k, v in pl.items() if k[1] in X})
    cache = None if args.quick else ROOT / oc["labels_cache"]
    J = unlong(pd.read_parquet(cache), ids) if cache and cache.exists() and not args.fresh else {}
    need = {(m, "all", K) for m in ("kefrin", "canus") for K in Ks_all}
    need |= {(m, p, K) for m in ("kefrin", "canus") for p in mkeys for K in k_common}
    J = ensure_joint(J, need, X, Ad, jcfg, seed, nj)
    if cache:
        long(J, ids).to_parquet(cache, index=False)
    L.update({k: v for k, v in J.items() if k[1] in X})
    print(f"метки: {time.time() - t0:.0f} с", flush=True)

    # --- индексы на сквозных, выбор лучшего K ---
    jobs = [(m, "all", K) for m in METHODS for K in Ks_all]
    rows = Parallel(n_jobs=nj, max_nbytes="1M", mmap_mode="r")(delayed(eval_job)(k, X["all"], A, L[k], n_perm_all, seed, variant) for k in jobs)
    df_all = pd.DataFrame(rows)
    def pick(cols):
        return {m: (borda_best_k(d.set_index("K_target"), cols)[0] if len(d) > 1 else int(d.K_target.iloc[0]))
                for m, d in df_all.groupby("method")}
    best, best_noavu = pick(ksel), pick([c for c in ksel if c.removeprefix("z_") != "AVU"])
    print("лучшее K (Борда на сквозных):", best, "| без AVU:", best_noavu, flush=True)

    # --- помесячно: общие K и лучшее K метода ---
    Km = {m: sorted(set(k_common) | {best[m]}) for m in METHODS}
    J = ensure_joint(J, {(m, p, best[m]) for m in ("kefrin", "canus") for p in mkeys}, X, Ad, jcfg, seed, nj)
    if cache:
        long(J, ids).to_parquet(cache, index=False)
    L.update({k: v for k, v in J.items() if k[1] in X})
    ensure_feat(L, {(m, p, best[m]) for m in FEAT for p in mkeys}, X, seed, nj)
    jobs = [(m, p, K) for m in METHODS for K in Km[m] for p in mkeys]
    rows = Parallel(n_jobs=nj, max_nbytes="1M", mmap_mode="r")(delayed(eval_job)(k, X[k[1]], A, L[k], n_perm_month, seed, variant) for k in jobs)
    df = pd.concat([df_all, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(out(oc["indices_full"]), index=False)
    keep = {(m, p, K) for m in METHODS for p in X for K in (Ks_all if p == "all" else Km[m])}
    long({k: L[k] for k in sorted(keep)}, ids).to_parquet(out(oc["labels_all"]), index=False)
    print(f"индексы: {time.time() - t0:.0f} с", flush=True)

    # --- сводная таблица и рейтинг ---
    num = ["K"] + IDX + ZIDX
    ind = []
    for var, Kof in (("common", lambda m: k_common[0]), ("best", lambda m: best[m])):
        for m in METHODS:
            d = df[(df.method == m) & (df.K_target == Kof(m))]
            a, mm = d[d.period == "all"].iloc[0], d[d.period != "all"][num].median()
            base = dict(method=m, name=NAME[m], family=FAMILY[m], variant=var)
            ind += [dict(base, period="all", **a[num]), dict(base, period="month_median", **mm)]
    ind = pd.DataFrame(ind)
    ind["K"] = ind.K.round().astype(int)
    rank = ranking(ind, ksel, cfg["n_grades"])
    rank["best_K"], rank["best_K_without_AVU"] = rank.method.map(best), rank.method.map(best_noavu)
    ren = {"Q": "MQ", "z_Q": "z_MQ"}
    ind.rename(columns=ren).to_csv(out(oc["indices"]), index=False)
    rank.rename(columns={"grade_Q": "grade_MQ"}).to_csv(out(oc["ranking"]), index=False)

    # --- устойчивость по месяцам ---
    st = []
    for m in METHODS:
        for K in Km[m]:
            wide = aligned_wide(L, m, K, mkeys)
            ref = L[(m, "all", K)]
            st.append(dict(method=m, K=K, **C.stability(wide),
                           ari_month_vs_all=float(np.mean([adjusted_rand_score(ref, wide[:, t]) for t in range(wide.shape[1])])),
                           min_types_in_month=int(min(len(np.unique(wide[:, t])) for t in range(wide.shape[1])))))
    st = pd.DataFrame(st)
    st.to_csv(out(oc["stability"]), index=False)

    # --- итоговая типология ---
    fm, fK = cfg["final"]["method"], cfg["final"]["K"]
    # номера месяцев — под модальный тип МО (иначе близкие сельские типы меняются номерами по кругу,
    # см. src/dynamics.py), затем 0 — самый массовый тип; состав кластеров каждого месяца не меняется
    lab = C.by_size(consensus_align(aligned_wide(L, fm, fK, mkeys)))
    fin = pd.DataFrame({"territory_id": np.repeat(ids, len(mkeys)), "month": np.tile(mkeys, len(ids)),
                        "label": lab.ravel().astype(int)})
    assert fin.label.min() == 0 and fin.label.nunique() == fin.label.max() + 1
    fin.to_parquet(out(oc["final_labels"]), index=False)

    meta = dict(graph=cfg["graph"], graph_stats=gstats, best_k=best, best_k_without_AVU=best_noavu, k_common=k_common, k_select=ksel,
                leiden_gamma=gammas, n_patterns=n_patterns, final=cfg["final"],
                final_sizes=np.bincount(lab.ravel()).tolist(), seconds=round(time.time() - t0))
    json.dump(meta, open(out("outputs/compare_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    show = ["method", "variant", "period", "K"] + IDX + ["z_AVI", "z_AVU", "z_Q"]
    print("\n== indices.csv ==\n", ind[show].round(3).to_string(index=False))
    print("\n== рейтинг ==\n", rank.drop(columns=["note", "name", "family"]).to_string(index=False))
    print("\n== устойчивость ==\n", st.round(3).to_string(index=False))
    print(f"\nитог: {fm}, K={fK}, размеры типов {meta['final_sizes']}; готово за {meta['seconds']} с")


if __name__ == "__main__":
    main()
