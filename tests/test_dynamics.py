"""Проверки динамики на игрушечных метках с известным ответом.
Запуск: .venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_dynamics.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import dynamics as D  # noqa: E402


def test_confirmed_switches_known():
    cases = {  # ряд: (сырых смен, смен при h=2, при h=3)
        (0, 0, 1, 0, 0): (2, 0, 0),              # одиночный выброс — шум
        (0, 0, 0, 1, 1, 1, 1): (1, 1, 1),        # настоящая смена
        (0, 0, 1, 1, 1, 1): (1, 1, 0),           # при h=3 старый тип у левого края не подтверждён
        (0, 0, 0, 1, 1, 0, 0, 0): (2, 2, 0),     # двухмесячный заход: при h=3 шум
        (0, 0, 0, 1, 2, 1, 2, 1, 1, 1): (5, 1, 1),  # мерцание 1/2, затем 1 держится
        (0, 0, 0, 0, 0, 1, 1): (1, 1, 0),        # смена у края: при h=3 не успела подтвердиться
        (3, 3, 3, 3): (0, 0, 0),
        (0, 1, 0, 1, 0, 1): (5, 0, 0),           # сплошное мерцание — ни одного эпизода
    }
    for row, (raw, h2, h3) in cases.items():
        assert D.raw_switches(np.array([row]))[0] == raw, row
        assert D.confirmed_switches(row, 2) == h2, row
        assert D.confirmed_switches(row, 3) == h3, row


def test_period_modes_and_trajectory():
    W = np.array([[0, 0, 1, 1, 1, 1],    # кварталы 0|1
                  [2, 1, 0, 5, 5, 5]])   # ничья 2/1/0 → больший номер (как на сайте)
    q = D.period_modes(W, [[0, 1, 2], [3, 4, 5]])
    assert q.tolist() == [[0, 1], [2, 5]]
    assert D.trajectory_class([0] * 8) == "stable"
    assert D.trajectory_class([0, 0, 0, 1, 1, 1, 1, 1]) == "moved"
    assert D.trajectory_class([0, 1, 1, 1, 1, 1, 1, 1]) == "other"   # смена у края
    assert D.trajectory_class([0, 0, 1, 1, 0, 0, 0, 0]) == "other"   # вернулся


def _events(ev):
    return sorted((e["event"], e["source"], e["target"]) for e in ev)


def test_monic_known_events():
    # t:   A=0 (10 объектов), B=1 (10), C=2 (10), D=3 (10)
    # t+1: A целиком → 0 (survive); B и C вместе → 1 (absorb); D на 4/3/3 → 4, 5, 6
    #      (доли 0,4/0,3/0,3: каждая < τ=0,5, но ≥ τ_split=0,25 и в сумме ≥ τ → split)
    prev = np.repeat([0, 1, 2, 3], 10)
    cur = np.r_[np.zeros(10), np.ones(20), np.full(4, 4), np.full(3, 5), np.full(3, 6)].astype(int)
    assert _events(D.monic(prev, cur)) == sorted([
        ("survive", (0,), (0,)), ("absorb", (1, 2), (1,)), ("absorb", (1, 2), (1,)), ("split", (3,), (4, 5, 6))])
    # ровно пополам — overlap 0,5 ≥ τ: по MONIC это ещё выживание в большую (первую) часть, не распад
    assert _events(D.monic(np.zeros(10, int), np.repeat([4, 5], 5)))[1] == ("survive", (0,), (4,))
    # исчезновение и появление: кластер 1 разбежался на 4 равные части (25 % < τ, но каждая ≥ τ_split=0,25 →
    # сумма 1 ≥ τ → это split); при τ_split=0,3 частей нет → disappear, а цели 2–5 никто не получил → emerge
    prev = np.repeat([0, 1], 8)
    cur = np.r_[np.zeros(8), [2, 2, 3, 3, 4, 4, 5, 5]].astype(int)
    assert _events(D.monic(prev, cur, tau_split=0.3)) == sorted(
        [("survive", (0,), (0,)), ("disappear", (1,), ())] + [("emerge", (), (y,)) for y in (2, 3, 4, 5)])
    assert ("split", (1,), (2, 3, 4, 5)) in _events(D.monic(prev, cur, tau_split=0.25))


def test_monic_identity_and_permutation():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 7, 500)
    ev = D.monic(lab, lab)
    assert {e["event"] for e in ev} == {"survive"} and len(ev) == 7
    ev = D.monic(lab, (lab + 3) % 7)  # номера другие — события те же, MONIC смотрит на состав
    assert {e["event"] for e in ev} == {"survive"}


def test_cluster_jaccard():
    ref = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    lab = np.array([5, 5, 5, 7, 7, 7, 7, 7])
    j = D.cluster_jaccard(ref, lab)
    assert j == {0: 0.75, 1: 0.8}


def test_consensus_align_undoes_month_relabel():
    # 30 МО в трёх устойчивых типах 12 месяцев; в месяце 5 номера 0 и 1 переставлены (как в final.parquet),
    # у трёх МО — одна настоящая смена 2 → 0 с месяца 8
    W = np.repeat(np.repeat([[0], [1], [2]], 10, 0), 12, 1)
    W[20:23, 8:] = 0
    Wbad = W.copy()
    Wbad[:, 5] = np.array([1, 0, 2])[W[:, 5]]
    assert (D.raw_switches(Wbad) > 0).sum() == 23          # перестановка номера — «смена» у 20 МО
    R = D.consensus_align(Wbad)
    assert (R == W).all()
    assert [D.confirmed_switches(r, 3) for r in R].count(1) == 3
