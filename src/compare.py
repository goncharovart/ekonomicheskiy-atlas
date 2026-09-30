"""Сборщик: пять методов на общих признаках и общем графе, рейтинг методов, итоговая типология.

Общие входы для всех методов:
  признаки — src/features.py: сквозные (15 столбцов) и помесячные (monthly_cols из baseline.yaml);
  граф     — src/network.py: kNN по корреляции помесячных рядов ln трат с лагом (lagcorr_sim).
             Граф один на весь период: это связь «чьи траты движутся вместе», атрибуты месяца
             (доли, уровень, доступность рынков) меняются, топология — нет. С признаками он почти не
             пересекается (Жаккар рёбер с косинусным kNN по долям 0,07), поэтому KEFRiN и CANUS
             получают от сети новую информацию, а не второй раз те же доли.

Паттерны сопоставимого размера (merge_patterns): агломеративное слияние паттернов Алескерова–Мячина.
  Старт — 127 паттернов полного порядка долей. На каждом шаге сливаются два кластера, чьи паттерны
  отличаются наименьшим числом попарных сравнений (у строгих порядков это одна перестановка соседей),
  а из таких пар — та, что меньше всего теряет R² в пространстве ранг-процентилей (приращение
  Уорда n_a·n_b/(n_a+n_b)·‖c_a − c_b‖²). Слияние общее для всех месяцев, поэтому номер кластера
  один и тот же во всех месяцах.

Рейтинг методов (threshold_places): пороговое агрегирование Алескерова (Aleskerov, Yakuba,
  Yuzbashev 2007, Math. Soc. Sci. 53(1) 106–110). Каждый индекс ставит методу оценку 1/2/3 по
  трети мест (1 — лучшая); выше метод с меньшим числом троек, при равенстве — с меньшим числом
  двоек. Правило некомпенсаторное: провал по одному индексу не покупается рекордом по другому.
  Рядом — Борда (сумма мест), она компенсаторная.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src import features as F
from src import network as N
from src.icvi import DIRECTION


def common_graph(w, knn_k, max_lag):
    """Общий граф: kNN (симметризация max) по корреляции рядов ln трат с лагом ≤ max_lag."""
    S, _ = N.lagcorr_sim(F.log_series(w)[0], max_lag)
    return N.knn_graph(S, knn_k)


# ---------- паттерны: агломеративное слияние ----------

def merge_patterns(codes, Z, lab, k_min=2):
    """codes: (P, пар) коды паттернов; Z: (n, d) точки в пространстве паттернов; lab: (n,) паттерн
    точки. Возвращает {K: массив длины P «паттерн → кластер 0..K−1»} для K от P−1 до k_min."""
    codes, Z, lab = np.asarray(codes), np.asarray(Z, float), np.asarray(lab)
    P = len(codes)
    n = np.bincount(lab, minlength=P).astype(float)
    S = np.zeros((P, Z.shape[1]))
    np.add.at(S, lab, Z)
    H = (codes[:, None, :] != codes[None, :, :]).sum(-1).astype(float)  # число несовпавших сравнений
    alive, group, maps = n > 0, np.arange(P), {}
    while alive.sum() > k_min:
        idx = np.flatnonzero(alive)
        na, c = n[idx], S[idx] / n[idx, None]
        cost = na[:, None] * na[None, :] / (na[:, None] + na[None, :]) * ((c[:, None] - c[None]) ** 2).sum(-1)
        h = H[np.ix_(idx, idx)]
        np.fill_diagonal(h, np.inf)
        cost = np.where(h == h.min(), cost, np.inf)
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        a, b = idx[i], idx[j]
        n[a], S[a], alive[b] = n[a] + n[b], S[a] + S[b], False
        H[a] = H[:, a] = np.minimum(H[a], H[b])
        group[group == b] = a
        maps[int(alive.sum())] = np.unique(group, return_inverse=True)[1]
    return maps


# ---------- рейтинг методов ----------

def grades(table, cols, n_grades=3):
    """Оценка 1..n_grades по каждому индексу: место метода (1 — лучший, по направлению индекса из
    DIRECTION; z_X — как X), разбитое на n_grades равных долей. NaN — худшая оценка."""
    m = len(table)
    out = {}
    for c in cols:
        v = table[c] * DIRECTION[c.removeprefix("z_")]
        r = v.rank(ascending=False, method="min").fillna(m)
        out[c] = np.ceil(n_grades * r / m).astype(int)
    return pd.DataFrame(out, index=table.index)


def threshold_places(G, n_grades=3):
    """Пороговое правило по таблице оценок G (строки — методы): сравниваем число худших оценок,
    при равенстве — следующих по худшести, и т.д. Место = 1 + число строго лучших методов."""
    G = np.asarray(G)
    key = [tuple(int((row == g).sum()) for g in range(n_grades, 1, -1)) for row in G]
    return np.array([1 + sum(k2 < k1 for k2 in key) for k1 in key])


def borda(table, cols):
    """Очки Борда: по каждому индексу метод получает число методов, которых он лучше (ничьи — пополам).
    Возвращает (очки, место)."""
    pts = sum((table[c] * DIRECTION[c.removeprefix("z_")]).rank(method="average") - 1 for c in cols)
    place = pts.rank(ascending=False, method="min").astype(int)
    return pts, place


# ---------- устойчивость по месяцам ----------

def stability(wide):
    """wide: (N, T) метки по месяцам с согласованными номерами. Сводка устойчивости."""
    wide = np.asarray(wide)
    sw = (wide[:, 1:] != wide[:, :-1]).sum(1)
    modal = np.array([np.bincount(r).argmax() for r in wide])
    ari = [adjusted_rand_score(wide[:, t], wide[:, t + 1]) for t in range(wide.shape[1] - 1)]
    return dict(stay_share=float((sw == 0).mean()), median_switches=float(np.median(sw)),
                modal_share=float((wide == modal[:, None]).mean()), ari_next_month=float(np.mean(ari)))


def by_size(wide):
    """Перенумеровать метки 0..K−1 по убыванию числа МО-месяцев (0 — самый массовый тип)."""
    wide = np.asarray(wide)
    u, inv = np.unique(wide, return_inverse=True)
    order = np.argsort(-np.bincount(inv.ravel()), kind="stable")
    rank = np.empty(len(u), int)
    rank[order] = np.arange(len(u))
    return rank[inv].reshape(wide.shape)
