"""Проверки интерпретации. Запуск: .venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_interpret.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import features as F  # noqa: E402
from src import interpret as P  # noqa: E402

FINAL = ROOT / "outputs/labels/final.parquet"

# Игрушечный контекст: объекты 0–3, атрибуты a, b, c.
TOY = np.array([[1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 1]], bool)


def test_fca_toy_known_concepts():
    # решётка из шести понятий, выписана вручную
    known = {(): (0, 1, 2, 3), (0,): (0, 1, 2), (2,): (1, 2, 3), (0, 1): (0, 2), (0, 2): (1, 2), (0, 1, 2): (2,)}
    assert P.all_concepts(TOY) == known
    # порождающие наборы из ≥1 атрибута дают всё, кроме верхнего понятия с пустым содержанием
    got = {B: tuple(np.flatnonzero(A)) for B, A, _ in P.concepts_from_generators(TOY, max_len=3)}
    assert got == {B: A for B, A in known.items() if B}


def test_stability_toy_exact_and_bounds():
    # ({0,1,2},{a}): C' = {a}, только если C не внутри b' = {0,2} и не внутри c' = {1,2} → {0,1} и {0,1,2}: 2/8
    # ({1,2,3},{c}): C должно содержать 3 → 4/8; верхнее понятие: C содержит 0 и 3 → 4/16
    cases = [((0, 1, 2), (0,), 0.25), ((1, 2, 3), (2,), 0.5), ((0, 1, 2, 3), (), 0.25)]
    for A, B, sigma in cases:
        assert P.stability_exact(TOY, np.array(A), B) == sigma
        mask = np.zeros(4, bool)
        mask[list(A)] = True
        d, lo, hi = P.stability_bounds(TOY, mask, B)
        assert lo <= sigma <= hi and d >= 1


def test_readable_drops_implied():
    names = ["x — нижняя четверть", "x — ниже медианы", "y — выше медианы", "y — верхняя четверть"]
    assert P.readable((0, 1, 2, 3), names) == ["x — нижняя четверть", "y — верхняя четверть"]


def test_mirkin_contributions_sum_to_r2():
    rng = np.random.default_rng(0)
    Z = pd.DataFrame(rng.normal(size=(300, 4)))
    types = pd.Series(rng.integers(0, 7, 300))
    Z.iloc[types.to_numpy() == 3] += 2.0
    mk, T = P.mirkin(Z, types)
    within = sum(((g - g.mean()) ** 2).to_numpy().sum() for _, g in Z.groupby(types))
    assert mk.contrib.sum() == pytest.approx(1 - within / T)
    assert mk.groupby("type").size().tolist() == [4] * 7


def test_profiles_synthetic_seven_types():
    rng = np.random.default_rng(1)
    n = 70
    sh = rng.dirichlet(np.ones(6), n)
    t = pd.DataFrame({f"mean_{s}": sh[:, i] for i, s in enumerate(P.SH)})
    t["mean_log_level"] = rng.normal(10, 0.3, n)
    t["growth_24_23"], t["market_access"], t["population"] = rng.normal(0.1, 0.05, n), rng.uniform(100, 600, n), 1e4
    t["region_name"], t["mo_type"], t["status"] = rng.choice(["A", "B", "C"], n), "муниципальный район", None
    t["lat"], t["lon"] = 55.0, rng.uniform(30, 120, n)
    types = pd.Series(np.arange(n) % 7)
    prof = P.profiles(t, types)
    assert list(prof.index) == list(range(7)) + ["all"]
    assert prof.n_mo.iloc[:7].sum() == n
    assert np.allclose(prof.loc["all", [f"{s}_ratio" for s in P.SH]].astype(float), 1)
    assert prof.loc[list(range(7)), [f"{s}_pct" for s in P.SH]].sum(axis=1).round(6).eq(100).all()


@pytest.mark.skipif(not FINAL.exists(), reason="нет outputs/labels/final.parquet: запусти scripts/run_compare.py")
def test_real_profiles_cover_all_seven_types():
    mt = P.modal_type(pd.read_parquet(FINAL))
    t = P.mo_table(ROOT)
    types = mt["type"].reindex(t.index)
    prof = P.profiles(t, types)
    assert list(prof.index[:-1]) == list(range(7)) and (prof.n_mo.iloc[:7] > 0).all()
    assert prof.n_mo.iloc[:7].sum() == len(t) == 2016
    ex = P.examples(F.through_features(F.load_wide(ROOT)), types, t)
    assert ex.groupby(["type", "role"]).size().to_dict() == {(k, r): n for k in range(7)
                                                             for r, n in (("пограничное", 3), ("типичное", 5))}
