"""Проверки индексов ICVI. Запуск: .venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_icvi.py

Игрушка с известным ответом: четыре явных кластера в признаках и граф, где связи только
внутри кластеров. Четыре, а не два: на неориентированном графе AVU в формуле лаборатории
вырожден при K ≤ 3 (K=2 → 1, K=3 → 2/3 у любого разбиения), см. test_avu_degenerate.
AVU смотрит только на форму разреза, а не на его вес (test_avu_cut_shape). Правильное разбиение должно быть лучше случайного по всем шести индексам
(SW, CH выше; S_Dbw ниже; AVI выше; AVU ниже; Q выше). SW и CH сверяются с sklearn,
S_Dbw — с пакетом s-dbw (если он стоит), графовые — с networkx (если он стоит).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.icvi import (all_indices, calinski_harabasz, graph_indices, perm_z,  # noqa: E402
                      s_dbw, silhouette)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(1)
    n = 60
    X = np.vstack([rng.normal(8 * k, 1, (n, 3)) for k in range(4)])
    truth = np.repeat(np.arange(4), n)
    same = truth[:, None] == truth[None, :]
    P = np.where(same, 0.3, 0.0)                    # вероятность ребра внутри / между
    A = np.triu(rng.random((4 * n, 4 * n)) < P, 1).astype(float)
    A = A + A.T
    rand = rng.permutation(truth)
    return X, sp.csr_matrix(A), truth, rand


def test_true_beats_random(toy):
    X, A, truth, rand = toy
    t, r = all_indices(X, A, truth), all_indices(X, A, rand)
    assert t["SW"] > r["SW"] and t["CH"] > r["CH"]
    assert t["S_Dbw"] < r["S_Dbw"]
    assert t["AVI"] > r["AVI"] and t["AVU"] < r["AVU"] and t["Q"] > r["Q"]
    assert t["AVI"] > 0.9 and t["AVU"] < 0.1 and t["Q"] > 0.4       # изолированные кластеры
    assert abs(r["AVI"] - 1 / 4) < 0.05 and abs(r["Q"]) < 0.05       # случайное: AVI ≈ 1/K, Q ≈ 0


def test_isolated_graph_limits():
    A = sp.block_diag([sp.csr_matrix(np.ones((5, 5)) - np.eye(5))] * 3).tocsr()
    g = graph_indices(A, np.repeat([0, 1, 2], 5))
    assert g["AVI"] == pytest.approx(1) and g["AVU"] == pytest.approx(0)
    assert g["Q"] == pytest.approx(1 - 1 / 3)


def test_avu_degenerate():
    rng = np.random.default_rng(5)
    W = np.triu(rng.random((30, 30)), 1)
    W = sp.csr_matrix(W + W.T)
    for K, v in ((2, 1.0), (3, 2 / 3)):
        for seed in range(3):
            y = np.random.default_rng(seed).integers(0, K, 30)
            assert graph_indices(W, y)["AVU"] == pytest.approx(v)


def test_avu_cut_shape():
    """AVU (Pattern) не зависит от доли внутренних рёбер: при равномерном шуме между
    кластерами он у правильного разбиения тот же, что у случайного, (K−1)/(2K−3)."""
    rng = np.random.default_rng(7)
    truth = np.repeat(np.arange(4), 50)
    P = np.where(truth[:, None] == truth[None, :], 0.5, 0.05)
    A = np.triu(rng.random((200, 200)) < P, 1).astype(float)
    A = sp.csr_matrix(A + A.T)
    t, r = graph_indices(A, truth), graph_indices(A, rng.permutation(truth))
    assert t["AVI"] > 0.7 > r["AVI"]
    assert t["AVU"] == pytest.approx(3 / 5, abs=0.02) and r["AVU"] == pytest.approx(3 / 5, abs=0.02)


def test_perm_z_signs(toy):
    X, A, truth, _ = toy
    z = perm_z(X, A, truth, n_perm=50, seed=0)
    assert z["z_SW"] > 3 and z["z_CH"] > 3 and z["z_AVI"] > 3 and z["z_Q"] > 3
    assert z["z_S_Dbw"] < -3 and z["z_AVU"] < -3


@pytest.mark.parametrize("K", [2, 3, 5, 8])
def test_sw_ch_match_sklearn(K):
    from sklearn.metrics import calinski_harabasz_score, silhouette_score
    rng = np.random.default_rng(K)
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, K, 200)
    y[:K] = np.arange(K)
    y[-1] = 99                                      # одноэлементный кластер и «дырявые» метки
    assert silhouette(X, y) == pytest.approx(silhouette_score(X, y), abs=1e-10)
    assert calinski_harabasz(X, y) == pytest.approx(calinski_harabasz_score(X, y), rel=1e-10)


@pytest.mark.parametrize("K", [2, 4, 6])
def test_s_dbw_matches_package(K):
    S_Dbw = pytest.importorskip("s_dbw").S_Dbw
    rng = np.random.default_rng(10 + K)
    X = np.vstack([rng.normal(3 * k, 1, (40, 3)) for k in range(K)])
    y = np.repeat(np.arange(K), 40)
    ref = S_Dbw(X, y, method="Halkidi", centr="mean", nearest_centr=False, metric="euclidean")
    assert s_dbw(X, y, variant="package") == pytest.approx(ref, rel=1e-10)


def test_graph_matches_networkx():
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(3)
    W = np.triu(rng.random((40, 40)) * (rng.random((40, 40)) < 0.2), 1)
    W = W + W.T
    y = rng.integers(0, 4, 40)
    G = nx.from_numpy_array(W)
    q = nx.community.modularity(G, [set(np.flatnonzero(y == k)) for k in range(4)], weight="weight")
    assert graph_indices(sp.csr_matrix(W), y)["Q"] == pytest.approx(q, abs=1e-12)
