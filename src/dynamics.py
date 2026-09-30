"""Динамика итоговой типологии: настоящие смены типа, переходы, события MONIC, устойчивость.

Вход — метки МО × месяц с согласованными номерами (outputs/labels/final.parquet): тип 3 в
марте и в июле — один тип. Всё здесь — чистые функции над массивами меток (N МО × T периодов),
без чтения файлов; сборка и выход — scripts/run_dynamics.py, проверки — tests/test_dynamics.py.

Настоящая смена (confirmed_switches). Ряд меток режем на серии одинаковых подряд. «Эпизод» —
серия длиной ≥ h месяцев. Смена засчитывается, когда следующий эпизод другого типа, чем
предыдущий; короткие серии между ними (мерцание соседних месяцев) пропускаются. Пример при h = 2:
A A B A A → 0 смен, A A B B → 1 смена. Эпизод у края ряда тоже должен быть ≥ h: смена в
последние h − 1 месяцев подтвердиться не успевает (консервативно).

Модальный тип периода (period_modes) — самый частый тип месяцев периода; при равенстве берётся
больший номер — то же правило, что mode_row в scripts/build_site.py, чтобы кварталы совпадали
с сайтом (номера упорядочены по убыванию размера, так что ничья уходит в меньший тип).

MONIC (Spiliopoulou et al., KDD 2006). Кластер X периода t, кластер Y периода t+1,
overlap(X, Y) = |X ∩ Y| / |X|. X «совпадает» с Y, если overlap ≥ τ. Тогда:
  survive — X совпал с Y, и больше никто с Y не совпал;
  absorb  — X совпал с Y вместе с другими X' (Y поглотил несколько кластеров);
  split   — ни одного совпадения, но есть ≥ 2 части Y с overlap ≥ τ_split и суммой ≥ τ;
  disappear — ничего из этого;
  emerge  — Y, в который никто не выжил, не влился и не распался.

Согласование номеров (consensus_align). В final.parquet каждый месяц перенумерован под сквозное
разбиение. Три сельских типа (0, 1, 2) и тип 6 в сквозном разбиении близки, и в 7 месяцах из 24
венгерский алгоритм раздаёт им номера по кругу, хотя по MONIC сами кластеры переживают месяц
целиком (overlap ≥ 0,65). Такая «смена» — смена номера, а не МО. Поэтому номера каждого месяца
подгоняются под модальный тип МО за 24 месяца (итерации до неподвижной точки). Состав кластеров
каждого месяца не меняется, ARI между месяцами тоже — меняются только номера.
"""
import numpy as np
import pandas as pd

from src.baseline import align


def to_wide(df, id_col="territory_id", t_col="month", y_col="label"):
    """Длинная таблица меток → DataFrame МО × период (столбцы по порядку времени)."""
    return df.pivot(index=id_col, columns=t_col, values=y_col).sort_index(axis=1)


def mode_row(a):
    c = np.bincount(np.asarray(a))
    return int(np.flatnonzero(c == c.max())[-1])


def period_modes(W, groups):
    """W: (N, T) метки; groups: список массивов индексов столбцов. → (N, len(groups)) модальные типы."""
    W = np.asarray(W)
    return np.array([[mode_row(r[g]) for g in groups] for r in W])


def runs(row):
    """Серии одинаковых меток: [(метка, начало, длина)]."""
    row = np.asarray(row)
    cut = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.r_[0, cut]
    lens = np.diff(np.r_[starts, len(row)])
    return [(int(row[s]), int(s), int(n)) for s, n in zip(starts, lens)]


def episodes(row, h):
    """Эпизоды — серии длиной ≥ h; подряд идущие эпизоды одного типа склеиваются."""
    out = []
    for lab, s, n in runs(row):
        if n < h:
            continue
        if out and out[-1][0] == lab:
            out[-1] = (lab, out[-1][1], s + n - out[-1][1])
        else:
            out.append((lab, s, n))
    return out


def confirmed_switches(row, h):
    return max(len(episodes(row, h)) - 1, 0)


def raw_switches(W):
    W = np.asarray(W)
    return (W[:, 1:] != W[:, :-1]).sum(1)


def trajectory_class(seq, min_side=2):
    """Последовательность модальных типов по периодам (кварталам) →
    'stable' (один тип везде), 'moved' (ровно одна смена A…A B…B, по обе стороны ≥ min_side
    периодов — сменил тип насовсем), 'other' (возвраты, несколько смен, смена у края)."""
    r = runs(seq)
    if len(r) == 1:
        return "stable"
    if len(r) == 2 and min(r[0][2], r[1][2]) >= min_side:
        return "moved"
    return "other"


def transition_matrix(a, b, K):
    """Матрица K × K: строка — тип в периоде t, столбец — в t+1, число МО."""
    return pd.crosstab(pd.Categorical(a, range(K)), pd.Categorical(b, range(K)), dropna=False).to_numpy()


def monic(prev, cur, tau=0.5, tau_split=0.25):
    """События MONIC между двумя разбиениями одних и тех же объектов (массивы меток).
    → список словарей: event, source (кортеж X), target (кортеж Y), overlap, size_ratio."""
    prev, cur = np.asarray(prev), np.asarray(cur)
    xs, ys = np.unique(prev), np.unique(cur)
    M = pd.crosstab(prev, cur).reindex(index=xs, columns=ys, fill_value=0).to_numpy()
    size_x, size_y = M.sum(1), M.sum(0)
    ov = M / size_x[:, None]
    match = {i: int(ov[i].argmax()) for i in range(len(xs)) if ov[i].max() >= tau}
    into = {}
    for i, j in match.items():
        into.setdefault(j, []).append(i)
    ev, hit = [], set(into)
    for i, x in enumerate(xs):
        x = int(x)
        if i in match:
            j = match[i]
            kind = "survive" if len(into[j]) == 1 else "absorb"
            ev.append(dict(event=kind, source=(x,) if kind == "survive" else tuple(int(xs[k]) for k in into[j]),
                           target=(int(ys[j]),), overlap=(round(float(ov[i, j]), 3),),
                           size_ratio=round(float(size_y[j] / size_x[i]), 3)))
            continue
        parts = np.flatnonzero(ov[i] >= tau_split)
        if len(parts) >= 2 and ov[i, parts].sum() >= tau:
            hit |= set(parts.tolist())
            ev.append(dict(event="split", source=(x,), target=tuple(int(ys[j]) for j in parts),
                           overlap=tuple(round(float(v), 3) for v in ov[i, parts]), size_ratio=np.nan))
        else:
            ev.append(dict(event="disappear", source=(x,), target=(), overlap=(round(float(ov[i].max()), 3),),
                           size_ratio=np.nan))
    for j, y in enumerate(ys):
        if j not in hit:
            ev.append(dict(event="emerge", source=(), target=(int(y),),
                           overlap=(round(float((M[:, j] / size_y[j]).max()), 3),), size_ratio=np.nan))
    return ev


def cluster_jaccard(ref, lab):
    """Для каждого кластера ref — Жаккар с лучшим по совпадению кластером lab (Hennig 2007).
    ref и lab — метки одних и тех же объектов. → {метка ref: Жаккар}."""
    ref, lab = np.asarray(ref), np.asarray(lab)
    out = {}
    for a in np.unique(ref):
        A = ref == a
        out[int(a)] = max(float((A & (lab == b)).sum() / (A | (lab == b)).sum()) for b in np.unique(lab))
    return out



def consensus_align(W, max_iter=20):
    """Перенумеровать каждый месяц (столбец W) под модальный тип МО; повторять, пока номера меняются."""
    W = np.asarray(W)
    for _ in range(max_iter):
        ref = np.array([mode_row(r) for r in W])
        new = np.stack([align(W[:, t], ref) for t in range(W.shape[1])], 1)
        if (new == W).all():
            break
        W = new
    return W
