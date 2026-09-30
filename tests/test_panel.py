"""Проверки панели. Запуск: .venv/Scripts/python.exe -X utf8 -m pytest -q tests"""
from pathlib import Path

import pandas as pd
import pytest

P = Path(__file__).resolve().parents[1] / "data" / "processed"
TOTAL, OTHER = "Все категории", "Прочее (расчёт)"


@pytest.fixture(scope="module")
def panel():
    return pd.read_parquet(P / "panel.parquet")


@pytest.fixture(scope="module")
def mo():
    return pd.read_parquet(P / "mo.parquet")


def test_no_duplicates(panel):
    assert not panel.duplicated(["territory_id", "month", "category"]).any()


def test_months_consecutive(panel):
    months = pd.DatetimeIndex(sorted(panel.month.unique()))
    assert (months == pd.date_range(months[0], months[-1], freq="MS")).all()
    assert len(months) == 24


def test_balanced_mo_have_every_month_and_category(panel, mo):
    bal = mo.index[mo.balanced]
    n = panel[panel.territory_id.isin(bal)].groupby("territory_id").size()
    assert (n == 24 * panel.category.nunique()).all()


def test_categories_sum_to_total(panel):
    w = panel.pivot_table(index=["territory_id", "month"], columns="category", values="value")
    parts = w.drop(columns=[TOTAL, OTHER])
    assert (parts.sum(axis=1) <= w[TOTAL]).all()  # «Все категории» шире пяти категорий
    assert ((parts.sum(axis=1) + w[OTHER] - w[TOTAL]).abs() < 1e-6).all()
    assert (w[OTHER] >= 0).all()


def test_values_positive(panel):
    assert (panel.loc[panel.category != OTHER, "value"] > 0).all()
    assert panel.value.notna().all()


def test_every_panel_mo_in_directory(panel, mo):
    assert set(panel.territory_id.unique()) <= set(mo.index)
    assert mo.index.is_unique
