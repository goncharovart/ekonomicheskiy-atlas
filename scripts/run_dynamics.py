"""Динамика итоговой типологии (KEFRiN, K = 7) одной командой:

    .venv/Scripts/python.exe -X utf8 scripts/run_dynamics.py [--boot 100] [--fresh]

Вход: outputs/labels/final.parquet (номера типов согласованы между месяцами), названия МО —
data/processed/mo.parquet, метки других методов — outputs/labels/compare_byK.parquet.
Номера типов по месяцам сначала согласуются под модальный тип МО (src/dynamics.consensus_align,
почему — там же); всё ниже считается на них, сводка по номерам final.parquet как есть — в
summary.json → as_given. Выход — outputs/dynamics/: labels_consensus.parquet, stability_mo.csv, transitions_quarter.csv, transitions_year.csv,
monic_events.csv, seasonal.csv, bootstrap.csv, summary.json. Текст — outputs/dynamics/DYNAMICS.md.

Бутстрэп — подвыборки 80 % МО без возвращения (с возвращением у графового метода узлы
дублировались бы в графе) и заново KEFRiN K = 7 на тех же признаках и подграфе. Дорогой шаг
кэшируется в bootstrap.csv; --fresh пересчитывает.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score as ari

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import dynamics as D  # noqa: E402

OUT = ROOT / "outputs/dynamics"
TAU, TAU_SPLIT = 0.5, 0.25
BOOT_MONTHS = ["2023-01", "2023-07", "2024-01", "2024-07"]
SUMMER, DEC = ("06", "07", "08"), ("12",)


def region_top(ids, mo, n=3):
    return "; ".join(f"{r} ({c})" for r, c in mo.loc[ids, "region_name"].value_counts().head(n).items())


def examples(ids, mo, n=3):
    top = mo.loc[ids].sort_values("population_2024", ascending=False).head(n)
    return "; ".join(f"{a} ({b})" for a, b in zip(top["name_short"].fillna(top["name"]), top["region_name"]))


def _kefrin_job(Y, A, K, params, seed, idx):
    logging.getLogger("kefrin").setLevel(logging.WARNING)
    from src import jury_methods as jm
    return idx, jm.kefrin(Y[idx], A[np.ix_(idx, idx)], n_clusters=K, seed=seed, **params)


def bootstrap(ids, W, months, B, frac, seed):
    """Подвыборки 80 % МО, KEFRiN K=7 заново: сквозной период (B раз) и 4 месяца (B // 10 раз)."""
    from src import compare as C
    from src import features as F
    from src.baseline import align
    bcfg = yaml.safe_load(open(ROOT / "configs/baseline.yaml", encoding="utf-8"))
    jcfg = yaml.safe_load(open(ROOT / "configs/jury.yaml", encoding="utf-8"))
    params = {k: v for k, v in jcfg["kefrin"].items() if k != "n_clusters"}
    clip = bcfg["features"]["clip_z"]
    w = F.load_wide(ROOT)
    A = C.common_graph(w, bcfg["network"]["knn_k"], bcfg["network"]["corr_max_lag"]).toarray()
    X = {"all": F.through_features(w, clip).loc[ids].to_numpy()}
    mf = {m.strftime("%Y-%m"): v for m, v in F.monthly_features(w, clip).items()}
    X.update({m: mf[m].loc[ids, bcfg["features"]["monthly_cols"]].to_numpy() for m in BOOT_MONTHS})
    # эталоны в номерах final: сквозное разбиение KEFRiN K=7 перенумеровано под модальный тип МО
    c = pd.read_parquet(ROOT / "outputs/labels/compare_byK.parquet")
    ref_all = c[(c.method == "kefrin") & (c.period == "all") & (c.K == 7)].set_index("mo_id").cluster.reindex(ids)
    modal = np.array([D.mode_row(r) for r in W])
    ref = {"all": align(ref_all.to_numpy(), modal)}
    ref.update({m: W[:, months.index(m)] for m in BOOT_MONTHS})
    rng = np.random.default_rng(seed)
    jobs = [(p, b, np.sort(rng.choice(len(ids), int(frac * len(ids)), replace=False)))
            for p in X for b in range(B if p == "all" else max(B // 10, 5))]
    res = Parallel(n_jobs=bcfg["n_jobs"], max_nbytes="1M", mmap_mode="r")(delayed(_kefrin_job)(X[p], A, 7, params, seed + b, idx) for p, b, idx in jobs)
    rows = []
    for (p, b, _), (idx, lab) in zip(jobs, res):
        r = ref[p][idx]
        rows.append(dict(period=p, run=b, ari_vs_full=ari(r, lab),
                         **{f"jaccard_type{k}": v for k, v in D.cluster_jaccard(r, lab).items()}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    fin = pd.read_parquet(ROOT / "outputs/labels/final.parquet")
    Wdf = D.to_wide(fin)
    ids, months, W0 = Wdf.index.to_numpy(), list(Wdf.columns), Wdf.to_numpy().astype(int)
    W = D.consensus_align(W0)
    pd.DataFrame(W, index=ids, columns=months).rename_axis("territory_id").reset_index().melt(
        "territory_id", var_name="month", value_name="label").to_parquet(OUT / "labels_consensus.parquet", index=False)
    relabeled = {months[t]: {int(a): int(np.bincount(W[W0[:, t] == a, t]).argmax()) for a in np.unique(W0[:, t])
                             if np.bincount(W[W0[:, t] == a, t]).argmax() != a} for t in range(W.shape[1])}
    N, T, K = W.shape[0], W.shape[1], int(W.max()) + 1
    mo = pd.read_parquet(ROOT / "data/processed/mo.parquet")
    quarters = [f"{m[:4]}-Q{(int(m[5:]) - 1) // 3 + 1}" for m in months[::3]]
    Q = D.period_modes(W, [list(range(q * 3, q * 3 + 3)) for q in range(len(quarters))])
    Y = D.period_modes(W, [[i for i, m in enumerate(months) if m.startswith(y)] for y in ("2023", "2024")])
    modal = np.array([D.mode_row(r) for r in W])
    q_ties = int(sum(len(set(r[q * 3:q * 3 + 3])) == 3 for r in W for q in range(len(quarters))))

    # --- 1. настоящие смены и траектории МО ---
    raw = D.raw_switches(W)
    h2 = np.array([D.confirmed_switches(r, 2) for r in W])
    h3 = np.array([D.confirmed_switches(r, 3) for r in W])
    traj = np.array([D.trajectory_class(r) for r in Q])
    st = pd.DataFrame({
        "territory_id": ids, "name": mo.loc[ids, "name"].to_numpy(), "region": mo.loc[ids, "region_name"].to_numpy(),
        "modal_type": modal, "modal_share": (W == modal[:, None]).mean(1).round(3),
        "raw_switches": raw, "switches_h2": h2, "switches_h3": h3,
        "mode_2023": Y[:, 0], "mode_2024": Y[:, 1], "quarter_modes": ["-".join(map(str, r)) for r in Q],
        "trajectory": traj})
    st.to_csv(OUT / "stability_mo.csv", index=False)

    # --- 2. матрицы переходов ---
    rows = []
    for q in range(len(quarters) - 1):
        M = D.transition_matrix(Q[:, q], Q[:, q + 1], K)
        rows += [dict(from_q=quarters[q], to_q=quarters[q + 1], from_type=a, to_type=b, n_mo=int(M[a, b]))
                 for a in range(K) for b in range(K) if M[a, b]]
    tq = pd.DataFrame(rows)
    tq.to_csv(OUT / "transitions_quarter.csv", index=False)
    q_stay = [float((Q[:, q] == Q[:, q + 1]).mean()) for q in range(len(quarters) - 1)]

    M = D.transition_matrix(Y[:, 0], Y[:, 1], K)
    rows = []
    for a in range(K):
        for b in range(K):
            if not M[a, b]:
                continue
            sel = ids[(Y[:, 0] == a) & (Y[:, 1] == b)]
            moved = ids[(Y[:, 0] == a) & (Y[:, 1] == b) & (traj == "moved")]
            rows.append(dict(from_type=a, to_type=b, n_mo=int(M[a, b]), share_of_from=round(M[a, b] / M[a].sum(), 3),
                             n_moved_for_good=len(moved), top_regions=region_top(sel, mo) if a != b else "",
                             examples=examples(moved if len(moved) else sel, mo) if a != b else ""))
    ty = pd.DataFrame(rows)
    ty.to_csv(OUT / "transitions_year.csv", index=False)
    flows = ty[ty.from_type != ty.to_type].sort_values("n_mo", ascending=False)

    # --- 3. MONIC ---
    ev_rows, sens = [], {}
    for level, L, names in (("quarter", Q, quarters), ("month", W, months)):
        for t in range(L.shape[1] - 1):
            for e in D.monic(L[:, t], L[:, t + 1], TAU, TAU_SPLIT):
                ev_rows.append(dict(level=level, from_period=names[t], to_period=names[t + 1], event=e["event"],
                                    source=" ".join(map(str, e["source"])), target=" ".join(map(str, e["target"])),
                                    overlap=" ".join(map(str, e["overlap"])), size_ratio=e["size_ratio"],
                                    same_number=(e["event"] == "survive" and e["source"] == e["target"])))
    ev = pd.DataFrame(ev_rows)
    ev.to_csv(OUT / "monic_events.csv", index=False)
    for tau in (0.4, 0.5, 0.6, 0.7, 0.8):
        c = pd.Series([e["event"] for t in range(Q.shape[1] - 1) for e in D.monic(Q[:, t], Q[:, t + 1], tau, TAU_SPLIT)])
        sens[str(tau)] = c.value_counts().to_dict()
    ovl = [float((Q[:, t][Q[:, t] == k] == Q[:, t + 1][Q[:, t] == k]).mean()) for t in range(Q.shape[1] - 1) for k in range(K)]
    monthly_ovl = [float((W[:, t][W[:, t] == k] == W[:, t + 1][W[:, t] == k]).mean()) for t in range(T - 1) for k in range(K)]

    # --- 4. устойчивость: бутстрэп KEFRiN, соседние месяцы, другие методы ---
    bpath = OUT / "bootstrap.csv"
    if args.fresh or not bpath.exists():
        tb = time.time()
        bs = bootstrap(ids, W, months, args.boot, 0.8, 42)
        bs.to_csv(bpath, index=False)
        print(f"бутстрэп: {len(bs)} прогонов KEFRiN за {time.time() - tb:.0f} с", flush=True)
    bs = pd.read_csv(bpath)
    jc = [c for c in bs.columns if c.startswith("jaccard_type")]
    boot = {p: dict(runs=len(g), ari_mean=round(g.ari_vs_full.mean(), 3), ari_q05=round(g.ari_vs_full.quantile(.05), 3),
                    ari_q95=round(g.ari_vs_full.quantile(.95), 3),
                    jaccard_by_type={c[-1]: round(g[c].mean(), 3) for c in jc})
            for p, g in bs.groupby("period")}
    ari_adj = [ari(W[:, t], W[:, t + 1]) for t in range(T - 1)]
    ari_adj_q = [ari(Q[:, q], Q[:, q + 1]) for q in range(Q.shape[1] - 1)]
    c = pd.read_parquet(ROOT / "outputs/labels/compare_byK.parquet")
    c = c[(c.K == 7) & (c.period != "all")]
    between = {}
    for m, g in c.groupby("method"):
        if m == "kefrin":
            continue
        piv = g.pivot(index="mo_id", columns="period", values="cluster").reindex(index=ids, columns=months)
        between[m] = round(float(np.mean([ari(W[:, t], piv.iloc[:, t]) for t in range(T)])), 3)

    # --- 5. сезонность ---
    cnt = np.stack([np.bincount(W[:, t], minlength=K) for t in range(T)])  # T × K
    share = cnt / N
    dev = share - share.mean(0)
    seas = pd.DataFrame([dict(type=k, month=months[t], n_mo=int(cnt[t, k]), share=round(share[t, k], 4),
                              dev_pp=round(100 * dev[t, k], 2)) for k in range(K) for t in range(T)])
    seas.to_csv(OUT / "seasonal.csv", index=False)
    cal = np.array([m[5:] for m in months])
    by_type = {}
    for k in range(K):
        d23, d24 = dev[:12, k], dev[12:, k]
        by_type[k] = dict(mean_n_mo=round(float(cnt[:, k].mean()), 1),
                          summer_minus_rest_pp=round(100 * float(share[np.isin(cal, SUMMER), k].mean() - share[~np.isin(cal, SUMMER), k].mean()), 2),
                          dec_minus_rest_pp=round(100 * float(share[cal == "12", k].mean() - share[cal != "12", k].mean()), 2),
                          yoy_corr_of_monthly_profile=round(float(np.corrcoef(d23, d24)[0, 1]), 3),
                          min_month=months[int(cnt[:, k].argmin())], max_month=months[int(cnt[:, k].argmax())],
                          range_n_mo=int(cnt[:, k].max() - cnt[:, k].min()))
    rec = {}
    for tag, mm in (("summer", SUMMER), ("december", DEC)):
        per_year = []
        for yi, y in enumerate(("2023", "2024")):
            cols = [i for i, m in enumerate(months) if m[:4] == y and m[5:] in mm]
            per_year.append(D.period_modes(W, [cols])[:, 0])
        hit = (per_year[0] == per_year[1]) & (per_year[0] != Y[:, 0]) & (per_year[1] != Y[:, 1]) & (Y[:, 0] == Y[:, 1])
        fl = pd.Series([f"{a}->{b}" for a, b in zip(Y[hit, 0], per_year[0][hit])]).value_counts()
        rec[tag] = dict(n_mo=int(hit.sum()), flows={k: int(v) for k, v in fl.head(5).items()},
                        top_flow_regions={f: region_top(ids[hit & (Y[:, 0] == int(f.split("->")[0])) &
                                                            (per_year[0] == int(f.split("->")[1]))], mo)
                                          for f in fl.head(3).index},
                        top_flow_examples={f: examples(ids[hit & (Y[:, 0] == int(f.split("->")[0])) &
                                                           (per_year[0] == int(f.split("->")[1]))], mo)
                                           for f in fl.head(3).index})

    # --- номера final.parquet как есть: те же сводки для сравнения ---
    raw0 = D.raw_switches(W0)
    modal0 = np.array([D.mode_row(r) for r in W0])
    Q0 = D.period_modes(W0, [list(range(q * 3, q * 3 + 3)) for q in range(len(quarters))])
    Y0 = D.period_modes(W0, [[i for i, m in enumerate(months) if m.startswith(y)] for y in ("2023", "2024")])
    sv0 = [e["source"] == e["target"] for t in range(T - 1) for e in D.monic(W0[:, t], W0[:, t + 1]) if e["event"] == "survive"]
    as_given = dict(share_zero_raw=round(float((raw0 == 0).mean()), 3), raw_median=float(np.median(raw0)), raw_total=int(raw0.sum()),
                    h2_median=float(np.median([D.confirmed_switches(r, 2) for r in W0])),
                    h3_median=float(np.median([D.confirmed_switches(r, 3) for r in W0])),
                    modal_share_mean=round(float((W0 == modal0[:, None]).mean()), 3),
                    quarters_traj=pd.Series([D.trajectory_class(r) for r in Q0]).value_counts().to_dict(),
                    year_changed=int((Y0[:, 0] != Y0[:, 1]).sum()),
                    monthly_survive_same_number_share=round(float(np.mean(sv0)), 3),
                    months_relabeled_by_consensus={m: v for m, v in relabeled.items() if v})

    # профили типов на согласованных номерах (как таблица в final_choice.md §5)
    from src import features as F
    raw_f = F.monthly_raw(F.load_wide(ROOT))
    lab_s = pd.Series(W.ravel(), index=pd.MultiIndex.from_product([ids, pd.to_datetime(months)], names=["mo_id", "month"]))
    prof = raw_f.loc[lab_s.index].assign(type=lab_s.to_numpy()).groupby("type")
    type_profile = (prof[[c for c in raw_f.columns if c.startswith("sh_")]].mean().mul(100).round(1)
                    .assign(spend_geo_mean=np.exp(prof["log_level"].mean()).round(-2),
                            market_access=prof["market_access"].mean().round(0),
                            n_mo_months=prof.size(), n_mo_modal=pd.Series(modal).value_counts().sort_index()))

    # --- summary ---
    tc = pd.Series(traj).value_counts()
    summary = dict(
        n_mo=N, n_months=T, K=K, quarters=quarters, quarter_ties_1_1_1=q_ties,
        labels="номера согласованы под модальный тип МО (consensus_align); final.parquet как есть — as_given",
        as_given=as_given, type_profile=type_profile.to_dict("index"),
        switches=dict(raw_median=float(np.median(raw)), h2_median=float(np.median(h2)), h3_median=float(np.median(h3)),
                      raw_total=int(raw.sum()), h2_total=int(h2.sum()), h3_total=int(h3.sum()),
                      share_zero_raw=round(float((raw == 0).mean()), 3), share_zero_h2=round(float((h2 == 0).mean()), 3),
                      share_zero_h3=round(float((h3 == 0).mean()), 3),
                      dist_h3={int(k): int(v) for k, v in pd.Series(h3).value_counts().sort_index().items()}),
        modal_share_mean=round(float((W == modal[:, None]).mean()), 3),
        quarters_traj={k: int(v) for k, v in tc.items()},
        quarter_same_type_share=[round(x, 3) for x in q_stay],
        year_same_type=int((Y[:, 0] == Y[:, 1]).sum()), year_changed=int((Y[:, 0] != Y[:, 1]).sum()),
        year_changed_and_moved_for_good=int(((Y[:, 0] != Y[:, 1]) & (traj == "moved")).sum()),
        top_year_flows=flows.head(5)[["from_type", "to_type", "n_mo", "share_of_from", "n_moved_for_good", "top_regions", "examples"]].to_dict("records"),
        monic=dict(tau=TAU, tau_split=TAU_SPLIT,
                   quarter_counts=ev[ev.level == "quarter"].event.value_counts().to_dict(),
                   month_counts=ev[ev.level == "month"].event.value_counts().to_dict(),
                   survive_same_number_share=round(float(ev[ev.event == "survive"].same_number.mean()), 3),
                   quarter_overlap_min=round(min(ovl), 3), quarter_overlap_median=round(float(np.median(ovl)), 3),
                   month_overlap_min=round(min(monthly_ovl), 3), month_overlap_median=round(float(np.median(monthly_ovl)), 3),
                   tau_sensitivity_quarters=sens),
        stability=dict(ari_adjacent_months_mean=round(float(np.mean(ari_adj)), 3),
                       ari_adjacent_months_min=round(float(np.min(ari_adj)), 3),
                       ari_adjacent_quarters_mean=round(float(np.mean(ari_adj_q)), 3),
                       ari_kefrin_vs_method_same_month=between, bootstrap_kefrin_80pct=boot),
        seasonal=dict(by_type=by_type, recurrent_switchers=rec),
        seconds=round(time.time() - t0))
    json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: summary[k] for k in ("switches", "quarters_traj", "year_changed", "monic", "stability")},
                     ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
