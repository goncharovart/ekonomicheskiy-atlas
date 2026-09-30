"""Рёбра между МО и графы.

Сходства (плотные матрицы N×N, на диагонали 0):
  cosine_sim   — косинус векторов долей трат (z-оценки долей 6 категорий, то есть отклонения
                 от средней по стране структуры; без центрирования у всех МО косинус ≈ 1).
                 Помесячно — доли месяца; сквозной — доли всех 24 месяцев подряд (6×24).
  lagcorr_sim  — корреляция помесячных рядов ln(трат на жителя) по 5 категориям и итогу, из
                 которых вычтено среднее по МО в том же месяце (общая сезонность); берётся
                 максимум по сдвигу 0…max_lag месяцев в обе стороны, по категориям — среднее.
  dtw_dist     — DTW-расстояние тех же рядов (итог), полоса Сако–Чибы; векторно по парам,
                 для подвыборки МО.
  proximity    — транспортная близость exp(−d/σ), d — расстояние по дорогам
                 (connection.parquet); где дороги нет — по железной дороге, где нет и её —
                 по прямой между центрами × detour (центры — mo_coords). σ — медиана
                 расстояния до k-го соседа.
  combo        — «похожесть трат × близость»: max(cos, 0) · proximity.

Графы (scipy.sparse CSR, симметричные, веса > 0, без петель):
  knn_graph(S, k)          — k ближайших по сходству, симметризация max(w_ij, w_ji);
  threshold_graph(S, top)  — доля top самых сильных пар (у изолятов нет рёбер).
to_igraph(A) переводит матрицу в igraph.Graph с атрибутом weight.
"""
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
CONN = ROOT / "data/raw/hackathon/hackathonlicence/connection.parquet"


def cosine_sim(X):
    X = np.asarray(X, float)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, 0)
    return S


def _zt(a):
    """z-оценка по последней оси (времени)."""
    a = a - a.mean(-1, keepdims=True)
    return a / np.maximum(a.std(-1, keepdims=True), 1e-12)


def lagcorr_sim(series, max_lag=2):
    """series: (N, C, T). Возвращает (S, L): максимум корреляции по сдвигам |l| ≤ max_lag
    и сдвиг, на котором он достигнут (L[i, j] > 0 — ряд j отстаёт от i на L месяцев)."""
    N, C, T = series.shape
    best, lag = np.full((N, N), -np.inf), np.zeros((N, N), int)
    for l in range(max_lag + 1):
        a, b = _zt(series[:, :, :T - l]), _zt(series[:, :, l:])
        R = np.einsum("ict,jct->ij", a, b) / (C * (T - l))  # corr(x_i(t), x_j(t+l))
        for R_, s in ((R, l), (R.T, -l)):
            upd = R_ > best
            best[upd], lag[upd] = R_[upd], s
    np.fill_diagonal(best, 0)
    return best, lag


def dtw_dist(series, window=3, chunk=200_000):
    """DTW между всеми парами рядов series: (n, T), евклидова стоимость, полоса |i−j| ≤ window.
    Векторно: динамика по клеткам T×T, внутри клетки — сразу все пары."""
    s = _zt(np.asarray(series, float))
    n, T = s.shape
    iu, ju = np.triu_indices(n, 1)
    D = np.zeros((n, n))
    for c0 in range(0, len(iu), chunk):
        a, b = s[iu[c0:c0 + chunk]], s[ju[c0:c0 + chunk]]
        prev = np.full((len(a), T + 1), np.inf)
        prev[:, 0] = 0
        for i in range(1, T + 1):
            cur = np.full_like(prev, np.inf)
            for j in range(max(1, i - window), min(T, i + window) + 1):
                cost = (a[:, i - 1] - b[:, j - 1]) ** 2
                cur[:, j] = cost + np.minimum(np.minimum(prev[:, j], cur[:, j - 1]), prev[:, j - 1])
            prev = cur
        D[iu[c0:c0 + chunk], ju[c0:c0 + chunk]] = np.sqrt(prev[:, T])
    return D + D.T


def mo_coords(ids, root=ROOT):
    """Координаты центров МО из mo.parquet; у 242 внутригородских территорий Москвы, Петербурга
    и Севастополя центра нет — берём внутреннюю точку полигона из data/geo."""
    import geopandas as gpd
    mo = pd.read_parquet(root / "data/processed/mo.parquet").reindex(ids)
    g = gpd.read_file(root / "data/geo/mo_simplified.geojson").set_index("territory_id").reindex(ids)
    pt = g.geometry.representative_point()
    return mo.lat.fillna(pt.y).to_numpy(), mo.lon.fillna(pt.x).to_numpy()


def road_distance(ids, lat, lon, detour=1.3, path=CONN):
    """Матрица расстояний (км) между МО ids: дороги → ж/д → прямая × detour."""
    c = pd.read_parquet(path)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    c = c[c.territory_id_x.isin(pos.index) & c.territory_id_y.isin(pos.index)]
    D = np.full((len(ids), len(ids)), np.nan)
    for t in ("railway", "highway"):  # дороги перезаписывают ж/д
        e = c[c.type == t]
        i, j = pos[e.territory_id_x].to_numpy(), pos[e.territory_id_y].to_numpy()
        D[i, j] = D[j, i] = e.distance.to_numpy()
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    h = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2
    gc = 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    D = np.where(np.isnan(D), gc * detour, D)
    np.fill_diagonal(D, 0)
    assert not np.isnan(D).any(), "нет координат у части МО"
    return D


def proximity(D, k=15, sigma=None):
    """exp(−d/σ); σ по умолчанию — медиана расстояния до k-го соседа."""
    if sigma is None:
        sigma = float(np.median(np.sort(D, axis=1)[:, k]))
    P = np.exp(-D / max(sigma, 1e-9))
    np.fill_diagonal(P, 0)
    return P, sigma


def knn_graph(S, k):
    """k ближайших по сходству S (больше — ближе). Рёбра с весом ≤ 0 выбрасываются."""
    S = np.asarray(S, float).copy()
    np.fill_diagonal(S, -np.inf)
    N = len(S)
    nb = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(N), k)
    w = S[rows, nb.ravel()]
    keep = w > 0
    A = sp.csr_matrix((w[keep], (rows[keep], nb.ravel()[keep])), shape=(N, N))
    return A.maximum(A.T).tocsr()


def threshold_graph(S, top=0.01):
    """Оставить долю top самых сильных пар (по верхнему треугольнику)."""
    S = np.asarray(S, float)
    iu = np.triu_indices(len(S), 1)
    v = S[iu]
    thr = max(np.quantile(v, 1 - top), 0)
    keep = v > thr
    A = sp.csr_matrix((v[keep], (iu[0][keep], iu[1][keep])), shape=S.shape)
    return (A + A.T).tocsr()


def to_igraph(A):
    A = sp.triu(A, 1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())))
    g.es["weight"] = A.data.tolist()
    return g


def graph_stats(A):
    from scipy.sparse.csgraph import connected_components
    deg = np.asarray((A > 0).sum(1)).ravel()
    return dict(edges=int(A.nnz // 2), mean_degree=float(deg.mean()), isolates=int((deg == 0).sum()),
                components=int(connected_components(A, directed=False)[0]))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 2, 20))
    x[1] = np.roll(x[0], 2, axis=-1)  # МО 1 повторяет МО 0 с опозданием на 2 месяца
    S, L = lagcorr_sim(x, 2)
    assert S[0, 1] > 0.8 and L[0, 1] == 2, (S[0, 1], L[0, 1])
    d = dtw_dist(x[:, 0], window=3)
    assert d[0, 1] < np.delete(d[0], [0, 1]).min() and np.allclose(d, d.T)
    A = knn_graph(cosine_sim(rng.normal(size=(50, 4))), 5)
    assert (A != A.T).nnz == 0 and A.diagonal().sum() == 0 and np.asarray((A > 0).sum(1)).min() >= 1
    print("ok")
