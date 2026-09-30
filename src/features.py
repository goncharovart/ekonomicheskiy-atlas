"""Признаки МО для кластеризации: помесячные и сквозные (за все 24 месяца).

Вход — data/processed/panel.parquet (МО × месяц × категория), только сбалансированные МО
(balanced=True в mo.parquet, 2 016 МО × 24 месяца). `value` — средние безналичные траты жителя
МО за месяц в рублях (модель СберИндекса), то есть уже «на жителя»: на население не делим.

Помесячные признаки (МО × месяц):
  sh_<кат>   6 долей в итоге «Все категории»: 5 базовых категорий + «Прочее (расчёт)»
             (= итог − пять категорий, в среднем 28% итога). Доли в сумме дают 1.
  log_level  ln(траты на жителя, все категории).
  growth     Δln(траты на жителя) к прошлому месяцу; у января 2023 прошлого месяца нет → 0.
             Общая для страны сезонность уходит при нормировке внутри месяца (см. ниже).
  market_access  индекс доступности рынков 2024 (СберИндекс), один на МО, повторяется по месяцам.
             Нет у 12 северных МО без круглогодичной дороги → ставим минимум по стране
             (нет дороги = худшая доступность).

Сквозные признаки (одна строка на МО):
  mean_sh_<кат>, mean_log_level — средние за 24 месяца;
  trend_sh_<кат>, trend_log_level — наклон МНК по времени, в единицах «за год»;
  market_access.

Нормировка: каждый признак — z-оценка по МО (внутри месяца для помесячных, по всей
сбалансированной выборке для сквозных), затем обрезка до ±clip_z (по умолчанию 4), чтобы
единичные выбросы (Чукотка, центральные районы Москвы) не тянули на себя центры k-means.
Все признаки после этого имеют один вес.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOTAL, OTHER = "Все категории", "Прочее (расчёт)"
BASE = ["Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт"]
CATS = BASE + [OTHER]
SHORT = {"Здоровье": "health", "Маркетплейсы": "market", "Общественное питание": "food_out",
         "Продовольствие": "grocery", "Транспорт": "transport", OTHER: "other"}
MA_FILE = ROOT / "data/raw/hackathon/hackathonlicence/market_access.parquet"


def load_wide(root=ROOT):
    """Широкая таблица сбалансированной панели: индекс (mo_id, month), столбцы — категории, value."""
    mo = pd.read_parquet(root / "data/processed/mo.parquet")
    p = pd.read_parquet(root / "data/processed/panel.parquet")
    p = p[p.territory_id.isin(mo.index[mo.balanced])]
    w = p.pivot_table(index=["territory_id", "month"], columns="category", values="value").sort_index()
    w.index = w.index.set_names(["mo_id", "month"])
    return w[[TOTAL] + CATS].astype(float)


def market_access(ids, path=MA_FILE):
    s = pd.read_parquet(path).set_index("territory_id")["market_access"].reindex(ids)
    return s.fillna(s.min())


def zscore(df, clip):
    z = (df - df.mean()) / df.std(ddof=0).replace(0, 1)
    return z.clip(-clip, clip)


def monthly_raw(w):
    """Помесячные признаки до нормировки: индекс (mo_id, month)."""
    f = pd.DataFrame({f"sh_{SHORT[c]}": w[c] / w[TOTAL] for c in CATS})
    f["log_level"] = np.log(w[TOTAL])
    f["growth"] = f.groupby(level="mo_id")["log_level"].diff().fillna(0.0)
    f["market_access"] = market_access(f.index.get_level_values("mo_id")).to_numpy()
    return f


def monthly_features(w, clip=4.0, cols=None):
    """Нормированные помесячные признаки: {month: DataFrame(mo_id × признак)}."""
    f = monthly_raw(w)
    if cols:
        f = f[cols]
    return {m: zscore(g.droplevel("month"), clip) for m, g in f.groupby(level="month")}


def through_raw(w):
    """Сквозные признаки до нормировки: средние и годовые тренды за 24 месяца."""
    f = monthly_raw(w).drop(columns=["growth", "market_access"])
    months = f.index.get_level_values("month")
    t = ((months.year - months.year.min()) * 12 + months.month - 1).to_numpy() / 12.0  # в годах
    t = pd.Series(t, index=f.index)
    tc = t - t.groupby(level="mo_id").transform("mean")
    fc = f - f.groupby(level="mo_id").transform("mean")
    slope = fc.mul(tc, axis=0).groupby(level="mo_id").sum().div((tc ** 2).groupby(level="mo_id").sum(), axis=0)
    out = pd.concat([f.groupby(level="mo_id").mean().add_prefix("mean_"), slope.add_prefix("trend_")], axis=1)
    out["market_access"] = market_access(out.index).to_numpy()
    return out


def through_features(w, clip=4.0):
    return zscore(through_raw(w), clip)


def share_matrix(w):
    """Доли 6 категорий по месяцам: массив (N, T, 6) и список mo_id."""
    sh = w[CATS].div(w[TOTAL], axis=0)
    ids = sh.index.get_level_values("mo_id").unique()
    return sh.to_numpy().reshape(len(ids), -1, len(CATS)), ids


def log_series(w):
    """ln трат на жителя по 5 базовым категориям и итогу, минус среднее по МО в том же месяце
    (убирает общую сезонность): массив (N, C, T)."""
    lg = np.log(w[BASE + [TOTAL]])
    lg = lg - lg.groupby(level="month").transform("mean")
    ids = lg.index.get_level_values("mo_id").unique()
    return lg.to_numpy().reshape(len(ids), -1, lg.shape[1]).transpose(0, 2, 1), ids


if __name__ == "__main__":
    w = load_wide()
    m = monthly_features(w)
    a = through_features(w)
    first = next(iter(m.values()))
    assert len(m) == 24 and first.shape == (2016, 9), first.shape
    assert a.shape == (2016, 15) and np.isfinite(a.to_numpy()).all(), a.shape
    assert np.allclose(monthly_raw(w).filter(like="sh_").sum(axis=1), 1)
    print("ok", first.shape, a.shape, list(a.columns))
