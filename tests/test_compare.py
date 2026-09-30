"""Проверки сборщика. Запуск: .venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_compare.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import compare as C  # noqa: E402
from src import jury_methods as jm  # noqa: E402

FINAL = ROOT / "outputs/labels/final.parquet"


def test_threshold_toy_known_answer():
    # оценки 1/2/3 по четырём индексам; по сумме (Борда) все трое равны — 8 баллов,
    # пороговое правило различает: у B нет троек, у C одна, у A две → B, C, A
    G = pd.DataFrame({"i1": [1, 2, 1], "i2": [1, 2, 2], "i3": [3, 2, 2], "i4": [3, 2, 3]}, index=list("ABC"))
    assert G.sum(axis=1).nunique() == 1
    assert C.threshold_places(G, 3).tolist() == [3, 1, 2]


def test_threshold_ties_and_second_grade():
    # тройки поровну (по одной) → решают двойки; одинаковые векторы делят место
    G = np.array([[1, 1, 3], [2, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert C.threshold_places(G, 3).tolist() == [1, 4, 2, 2]


def test_grades_from_values_direction():
    # 5 методов: места 1..5 → оценки 1,2,2,3,3; у S_Dbw и z_AVU меньше — лучше
    t = pd.DataFrame({"SW": [0.5, 0.4, 0.3, 0.2, 0.1], "S_Dbw": [0.5, 0.4, 0.3, 0.2, 0.1],
                      "z_AVU": [5.0, 4.0, 3.0, 2.0, 1.0]}, index=list("abcde"))
    G = C.grades(t, ["SW", "S_Dbw", "z_AVU"], 3)
    assert G.SW.tolist() == [1, 2, 2, 3, 3]
    assert G.S_Dbw.tolist() == [3, 3, 2, 2, 1] and G.z_AVU.tolist() == [3, 3, 2, 2, 1]
    # векторы (троек, двоек): a (2,0), b (2,1), c (0,3), d (1,2), e (1,0) → c, e, d, a, b
    assert C.threshold_places(G, 3).tolist() == [4, 5, 1, 3, 2]
    # Борда компенсаторна и ставит первым e (рекорд по двум индексам перекрывает худший SW)
    pts, place = C.borda(t, ["SW", "S_Dbw", "z_AVU"])
    assert pts.tolist() == [4, 5, 6, 7, 8] and place.tolist() == [5, 4, 3, 2, 1]


def test_merge_patterns_adjacent_and_ward():
    # порядки трёх показателей: p0 a>b>c, p1 a>c>b (соседний с p0), p2 c>b>a, p3 b>c>a (соседний с p2)
    X = np.array([[3, 2, 1], [3, 1, 2], [1, 2, 3], [1, 3, 2]], float)
    codes = jm.pattern_codes(X)
    # точки: p0 и p1 далеко друг от друга, p2 и p3 почти совпадают → Уорд сначала сливает p2+p3
    Z = np.array([[0.9, 0.5, 0.1]] * 5 + [[0.9, 0.0, 0.8]] * 5 + [[0.1, 0.45, 0.5]] * 5 + [[0.1, 0.5, 0.45]] * 5)
    lab = np.repeat(np.arange(4), 5)
    maps = C.merge_patterns(codes, Z, lab, k_min=2)
    m3, m2 = maps[3], maps[2]
    assert m3[2] == m3[3] and len({m3[0], m3[1], m3[2]}) == 3
    assert m2[0] == m2[1] and m2[2] == m2[3] and m2[0] != m2[2]  # сливаются только соседние порядки


def test_by_size_and_stability():
    wide = np.array([[5, 5, 5], [5, 5, 9], [9, 9, 9], [2, 2, 2]])
    lab = C.by_size(wide)
    assert lab[0, 0] == 0 and lab[2, 0] == 1 and lab[3, 0] == 2  # 5 — самый частый, потом 9, потом 2
    s = C.stability(wide)
    assert s["stay_share"] == pytest.approx(0.75) and s["median_switches"] == 0


@pytest.mark.skipif(not FINAL.exists(), reason="нет outputs/labels/final.parquet: запусти scripts/run_compare.py")
def test_final_covers_all_mo_months():
    f = pd.read_parquet(FINAL)
    mo = pd.read_parquet(ROOT / "data/processed/mo.parquet")
    ids = set(mo.index[mo.balanced])
    assert list(f.columns) == ["territory_id", "month", "label"]
    assert len(ids) == 2016 and set(f.territory_id) == ids
    months = sorted(f.month.unique())
    assert len(months) == 24 and months[0] == "2023-01" and months[-1] == "2024-12"
    assert len(f) == 2016 * 24 and not f.duplicated(["territory_id", "month"]).any()
    assert pd.api.types.is_integer_dtype(f.label) and f.label.min() == 0
    assert f.label.nunique() == f.label.max() + 1  # номера типов подряд с 0
    assert f.groupby("month").label.nunique().min() >= 2


def test_new_baselines_k_labels_and_seed():
    # три блоба по признакам и граф «клика внутри блоба»: Ward, GMM и спектральная дают ровно K
    # меток, находят блобы и повторяются при том же seed
    from scipy import sparse
    from sklearn.metrics import adjusted_rand_score
    from src.baseline import gmm_labels, spectral_labels, ward_labels
    rng = np.random.default_rng(0)
    truth = np.repeat(np.arange(3), 30)
    X = rng.normal(0, 0.3, (90, 4)) + np.eye(3, 4)[truth] * 5
    A = sparse.csr_matrix((truth[:, None] == truth[None, :]) & ~np.eye(90, dtype=bool), dtype=float)
    A = A + sparse.csr_matrix(rng.random((90, 90)) < 0.01) * 0.1
    A = A.maximum(A.T)
    for f, inp in ((ward_labels, X), (gmm_labels, X), (spectral_labels, A)):
        a, b = f(inp, 3, 7), f(inp, 3, 7)
        assert len(np.unique(a)) == 3 and adjusted_rand_score(truth, a) == 1, f.__name__
        assert (a == b).all(), f.__name__
        assert len(np.unique(f(inp, 5, 7))) == 5, f.__name__
