"""Проверки методов жюри. Запуск: .venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_jury.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import jury_methods as jm  # noqa: E402

LABELS = ROOT / "outputs" / "labels"
HAS_EXT = (ROOT / "external" / "KEFRiN").exists() and (ROOT / "external" / "CANUS").exists()


def test_pattern_toy_groups():
    # три объекта с порядком a>b>c (разный масштаб — порядок тот же), два с c>a>b, один a=b>c
    X = np.array([[3, 2, 1], [30, 20, 10], [0.9, 0.5, 0.1],
                  [2, 1, 3], [5, 4, 9],
                  [2, 2, 1]], float)
    lab, uniq = jm.pattern_clusters(jm.pattern_codes(X, "pairs"))
    assert list(lab) == [0, 0, 0, 1, 1, 2]  # номера по убыванию частоты
    names = ["a", "b", "c"]
    assert jm.pattern_text(uniq[0], names) == "a > b > c"
    assert jm.pattern_text(uniq[1], names) == "c > a > b"
    assert jm.pattern_text(uniq[2], names) == "a = b > c"


def test_pattern_order_invariance():
    # паттерн не меняется при монотонном преобразовании, общем для всех показателей объекта
    rng = np.random.default_rng(0)
    X = rng.random((200, 5))
    a = jm.pattern_codes(X)
    assert (a == jm.pattern_codes(np.exp(3 * X) + 7)).all()
    assert len(np.unique(a, axis=0)) <= 120  # не больше 5! порядков


def test_pattern_eps_and_adjacent():
    X = np.array([[1.0, 1.04, 0.5]])
    assert jm.pattern_codes(X, "pairs", eps=0.05).tolist() == [[0, 1, 1]]
    assert jm.pattern_codes(X, "adjacent").tolist() == [[-1, 1]]
    assert jm.pattern_text(jm.pattern_codes(X, "adjacent")[0], ["a", "b", "c"], "adjacent") == "a < b > c"


def test_pattern_stability():
    lab = pd.DataFrame({"mo_id": [1, 1, 1, 2, 2, 2], "month": ["2023-01", "2023-02", "2023-03"] * 2,
                        "cluster": [0, 0, 0, 0, 1, 0]})
    s = jm.pattern_stability(lab).set_index("mo_id")
    assert s.loc[1, "n_changes"] == 0 and s.loc[2, "n_changes"] == 2
    assert s.loc[2, "modal_pattern"] == 0 and s.loc[2, "modal_share"] == pytest.approx(2 / 3)


def test_knn_graph():
    rng = np.random.default_rng(1)
    A = jm.knn_graph(rng.random((50, 5)), k=5)
    assert (A == A.T).all() and (np.diag(A) == 0).all()
    assert ((A > 0).sum(1) >= 5).all()


def _two_blobs():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal([1, 0, 0], 0.05, (30, 3)), rng.normal([0, 1, 0], 0.05, (30, 3))])
    X = np.abs(X) + 0.01
    return X, jm.knn_graph(X, 5)


@pytest.mark.skipif(not HAS_EXT, reason="нет external/: scripts/fetch_external.py")
def test_kefrin_canus_recover_blobs():
    X, A = _two_blobs()
    truth = np.r_[np.zeros(30), np.ones(30)]
    for lab in (jm.kefrin(X, A, n_clusters=2, n_init=3),
                jm.canus(jm.zscore(X), A, n_clusters=2, epochs=30)):
        assert len(lab) == 60
        agree = (lab == truth).mean()
        assert max(agree, 1 - agree) == 1.0


@pytest.mark.skipif(not HAS_EXT, reason="нет external/: scripts/fetch_external.py")
def test_canus_fast_grad_norms_match_autograd():
    torch = pytest.importorskip("torch")
    X, A = _two_blobs()
    jm._external("CANUS")
    from canus import CANUSClusterer
    m = CANUSClusterer(n_clusters=2, device="cpu", epochs=1)
    Xt, At = torch.as_tensor(jm.zscore(X), dtype=torch.float32), torch.as_tensor(A, dtype=torch.float32)
    m._init_centroids(Xt, At)
    for k in range(2):
        ref = m._per_sample_grad_norms(Xt, At, k)
        assert torch.allclose(jm._fast_grad_norms(m, Xt, At, k), ref, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", ["kefrin", "canus", "pattern"])
def test_labels_cover_all_mo(name):
    f = LABELS / f"{name}.parquet"
    if not f.exists():
        pytest.skip("нет меток: scripts/run_jury.py")
    lab = pd.read_parquet(f)
    assert list(lab.columns) == ["mo_id", "month", "cluster"]
    mo = pd.read_parquet(ROOT / "data" / "processed" / "mo.parquet", columns=["balanced"])
    ids = set(mo.index[mo.balanced])
    assert lab.cluster.notna().all()
    assert not lab.duplicated(["mo_id", "month"]).any()
    for month, g in lab.groupby("month"):
        assert set(g.mo_id) == ids, month
    assert "all" in set(lab.month)
