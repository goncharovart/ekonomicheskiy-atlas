"""Базовые методы и выбор K по индексам ICVI.

  k-means   — sklearn KMeans на нормированных признаках (src/features.py), n_init из конфига.
  Leiden    — leidenalg, RBConfiguration (модулярность с разрешением γ) на kNN-графе.
              Нужное K получаем бисекцией по log γ (до 5 сидов); если ровно K не выходит — ближайшее.
              «Естественное» разбиение — γ = 1 (чистая модулярность).
  Ward, GMM, спектральная — ещё три базовых метода для сборщика (scripts/run_compare.py):
              Уорд и смесь гауссиан на тех же признаках, что k-means, спектральная — на общем графе.

Периоды: 24 месяца (помесячные признаки и граф месяца) и 'all' (сквозные признаки, граф по
долям всех 24 месяцев). Для каждого периода, метода и K считаются все индексы
(src/icvi.py) на одних и тех же признаках и одном графе, плюс z против перестановок.
K выбирается суммой рангов Борда по индексам из конфига: на 'all' и отдельно по медиане за
24 месяца (для справки). Помесячные метки сохраняются при K из 'all', чтобы типы были одни и
те же во всех месяцах, и перенумерованы под метки 'all' того же метода (венгерский алгоритм
по пересечениям): кластер 3 в марте и в июле — один тип.
"""
import leidenalg as la
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from src.icvi import DIRECTION, all_indices, pairwise_dist, perm_z
from src.network import to_igraph


def kmeans_labels(X, K, seed=42, n_init=20):
    return KMeans(n_clusters=K, n_init=n_init, random_state=seed).fit_predict(np.asarray(X))


def ward_labels(X, K, seed=None):
    """Агломеративная кластеризация Уорда: детерминирована, seed не нужен (он для общего вызова)."""
    return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(np.asarray(X))


def gmm_labels(X, K, seed=42, n_init=5):
    """Смесь гауссиан с диагональной ковариацией: доли шести категорий в сумме дают 1, поэтому полная
    ковариация признаков вырождена (держится только на reg_covar); диагональ устойчива и даёт
    12·30 = 360 параметров при K=12 на 15 признаках вместо 1 620 у полной."""
    X = np.asarray(X)
    return GaussianMixture(K, covariance_type="diag", n_init=n_init, random_state=seed).fit(X).predict(X)


def spectral_labels(A, K, seed=42):
    """Нормализованная спектральная кластеризация (Ши–Малик) на взвешенном графе A: K собственных
    векторов лапласиана случайного блуждания I − D⁻¹A, затем k-means по строкам (10 стартов, seed)."""
    return SpectralClustering(K, affinity="precomputed", random_state=seed).fit_predict(A)


def _leiden(g, gamma, seed, n_iterations):
    p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                          resolution_parameter=gamma, seed=seed, n_iterations=n_iterations)
    return np.array(p.membership)


def leiden_labels(A, K=None, seed=42, gamma_range=(1e-3, 20.0), steps=30, n_iterations=-1, g=None):
    """Разбиение Leiden с K кластерами (бисекция по γ) или при γ=1, если K=None.
    Возвращает (labels, gamma)."""
    g = g or to_igraph(A)
    if K is None:
        return _leiden(g, 1.0, seed, n_iterations), 1.0
    best = None
    for s in range(seed, seed + 5):  # ровно K не вышло — повторить бисекцию с другим сидом
        lo, hi = np.log(gamma_range[0]), np.log(gamma_range[1])
        for _ in range(steps):
            mid = (lo + hi) / 2
            m = _leiden(g, np.exp(mid), s, n_iterations)
            k = m.max() + 1
            if best is None or abs(k - K) < abs(best[0].max() + 1 - K):
                best = (m, float(np.exp(mid)))
            if k == K:
                return best
            lo, hi = (mid, hi) if k < K else (lo, mid)
    return best


def align(labels, ref):
    """Перенумеровать labels так, чтобы они максимально совпадали с ref (венгерский алгоритм)."""
    labels, ref = np.asarray(labels), np.asarray(ref)
    a, b = np.unique(labels), np.unique(ref)
    M = pd.crosstab(labels, ref).reindex(index=a, columns=b, fill_value=0).to_numpy()
    r, c = linear_sum_assignment(-M)
    mapping = {a[i]: b[j] for i, j in zip(r, c)}
    nxt = max(b.max(), a.max()) + 1
    for x in a:
        if x not in mapping:
            mapping[x], nxt = nxt, nxt + 1
    return np.array([mapping[x] for x in labels])


def evaluate(X, A, labels, D, n_perm, seed, s_dbw_variant):
    idx = all_indices(X, A, labels, D, s_dbw_variant)
    idx.update(perm_z(X, A, labels, n_perm, seed, D, s_dbw_variant, observed=idx))
    return idx


def sweep_period(period, X, A, cfg, n_perm):
    """Все K для одного периода: строки таблицы индексов и метки {(method, K): labels}."""
    c = cfg["clustering"]
    seed, variant = cfg["seed"], cfg["icvi"]["s_dbw_variant"]
    Xv = np.asarray(X)
    D, g = pairwise_dist(Xv), to_igraph(A)
    rows, labels = [], {}
    for K in range(c["k_range"][0], c["k_range"][1] + 1):
        km = kmeans_labels(Xv, K, seed, c["kmeans"]["n_init"])
        ld, gamma = leiden_labels(A, K, seed, c["leiden"]["gamma_range"], c["leiden"]["bisect_steps"],
                                  c["leiden"]["n_iterations"], g=g)
        for method, lab, extra in (("kmeans", km, {}), ("leiden", ld, {"gamma": gamma})):
            labels[(method, K)] = lab
            rows.append(dict(method=method, period=period, K_target=K, K=int(len(np.unique(lab))),
                             min_size=int(np.bincount(lab).min()), **extra,
                             **evaluate(Xv, A, lab, D, n_perm, seed, variant)))
    return rows, labels


def borda_best_k(tab, cols):
    """tab: строки одного метода, индекс — K. Сумма рангов (лучшее значение = наибольший ранг)."""
    score = sum((tab[c] * DIRECTION[c.removeprefix("z_")]).rank(method="average") for c in cols)
    return int(score.idxmax()), score


def select_k(df, cols):
    """Лучшее K для каждого метода: по 'all' и по медиане помесячных значений."""
    out = {}
    for method, d in df.groupby("method"):
        a = d[d.period == "all"].set_index("K_target")
        m = d[d.period != "all"].groupby("K_target")[cols].median()
        out[method] = {"all": borda_best_k(a, cols)[0], "month": borda_best_k(m, cols)[0]}
    return out
