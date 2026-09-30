"""Шесть внутренних индексов качества кластеризации (ICVI) из критериев конкурса, векторно.

В пространстве признаков X (N×d, евклидово расстояние):
  SW    силуэт, Rousseeuw 1987. Больше — лучше. У одноэлементного кластера s(i)=0.
  CH    Калински–Харабаш 1974: [tr(B)/(K−1)] / [tr(W)/(N−K)]. Больше — лучше.
  S_Dbw Halkidi, Vazirgiannis 2001: Scat + Dens_bw. Меньше — лучше.
        variant="paper" (по умолчанию, как в статье): σ — вектор дисперсий по признакам,
        density(v_k) и density(u_kl) считаются по точкам C_k ∪ C_l.
        variant="package": как в пакете s-dbw (method='Halkidi', centr='mean',
        nearest_centr=False): σ — вектор СКО, density(v_k) — по точкам одного C_k.

На графе A (симметричная неотрицательная N×N без петель; scipy.sparse или ndarray),
S = HᵀAH — веса связей между кластерами (H — индикаторы N×K):
  AVI   средняя изолированность: mean_a S_aa / Σ_b S_ab. Больше — лучше; у случайного ≈ 1/K.
  AVU   средняя «слитость»: Σ_a Σ_{b≠a} S_ab / (out_a + out_b − S_ab) / K. Меньше — лучше.
        Деление на K, а не на K(K−1) — как в коде лаборатории жюри (Pattern).
        Вырожден при K ≤ 3 на неориентированном графе: знаменатель U_ab — это все рёбра разреза,
        поэтому AVU = 1 при K=2 и 2/3 при K=3 у любого разбиения (проверено и на коде Pattern).
        Осмыслен с K ≥ 4 — отсюда нижняя граница диапазона K. И важно: AVU зависит только от
        формы разреза (как вес между кластерами распределён по парам), а не от его доли в
        общем весе; у случайного разбиения AVU ≈ (K−1)/(2K−3). Поэтому AVU читаем через z.
  ANUI  1 / (AVU + 1/AVI). Больше — лучше (в критериях нет, считаем для справки).
  Q     модулярность Ньюмана–Гирвана: Σ_a [S_aa/2m − (d_a/2m)²]. Это MQ из критериев. Больше — лучше.
  Формулы AVI, AVU, ANUI, Q — из reports/METODY-ZHYURI.md §2.4–2.6 (сверены с Pattern и networkx).
  MQ Манкоридиса на неориентированном графе равен K·AVI, отдельно не считаем.

perm_z — z-оценка против случайной перестановки меток (размеры кластеров те же):
z = (индекс − среднее по перестановкам) / СКО по перестановкам. Для индексов «меньше — лучше»
(S_Dbw, AVU) хорошее разбиение даёт отрицательный z.
"""
import numpy as np

DIRECTION = {"SW": 1, "CH": 1, "S_Dbw": -1, "AVI": 1, "AVU": -1, "ANUI": 1, "Q": 1}
FEATURE_IDX, GRAPH_IDX = ("SW", "CH", "S_Dbw"), ("AVI", "AVU", "ANUI", "Q")


def _onehot(labels):
    _, y = np.unique(np.asarray(labels), return_inverse=True)
    return y, np.eye(y.max() + 1)[y]


def pairwise_dist(X):
    X = np.asarray(X, float)
    sq = (X ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))


def silhouette(X, labels, D=None):
    y, H = _onehot(labels)
    if H.shape[1] < 2:
        return np.nan
    D = pairwise_dist(X) if D is None else D
    n = H.sum(0)
    sums = D @ H                                   # N×K: сумма расстояний до каждого кластера
    own = sums[np.arange(len(y)), y]
    a = np.divide(own, n[y] - 1, out=np.zeros_like(own), where=n[y] > 1)
    mean_other = sums / n
    mean_other[np.arange(len(y)), y] = np.inf
    b = mean_other.min(1)
    s = np.divide(b - a, np.maximum(a, b), out=np.zeros_like(a), where=np.maximum(a, b) > 0)
    s[n[y] == 1] = 0
    return float(s.mean())


def calinski_harabasz(X, labels):
    X = np.asarray(X, float)
    y, H = _onehot(labels)
    N, K = H.shape
    if K < 2 or K >= N:
        return np.nan
    n = H.sum(0)
    C = (H.T @ X) / n[:, None]
    B = (n * ((C - X.mean(0)) ** 2).sum(1)).sum()
    W = ((X - C[y]) ** 2).sum()
    return float(B / (K - 1) / (W / (N - K))) if W > 0 else np.inf


def s_dbw(X, labels, variant="paper"):
    X = np.asarray(X, float)
    y, H = _onehot(labels)
    N, K = H.shape
    if K < 2:
        return np.nan
    n = H.sum(0)
    C = (H.T @ X) / n[:, None]
    var = (H.T @ X ** 2) / n[:, None] - C ** 2     # K×d: дисперсии по признакам внутри кластеров
    var = np.maximum(var, 0)
    sig, sig_all = (var, X.var(0)) if variant == "paper" else (np.sqrt(var), X.std(0))
    nk = np.linalg.norm(sig, axis=1)
    scat = nk.mean() / np.linalg.norm(sig_all)
    stdev = np.sqrt(nk.sum()) / K
    k, l = np.triu_indices(K, 1)
    U = (C[k] + C[l]) / 2
    near_u = np.linalg.norm(X[:, None, :] - U[None], axis=2) <= stdev    # N×P
    near_c = np.linalg.norm(X[:, None, :] - C[None], axis=2) <= stdev    # N×K
    in_pair = H[:, k] + H[:, l]                                          # N×P: точка в C_k ∪ C_l
    dens_u = (near_u * in_pair).sum(0)
    if variant == "paper":
        dk, dl = (near_c[:, k] * in_pair).sum(0), (near_c[:, l] * in_pair).sum(0)
    else:
        own = (near_c * H).sum(0)
        dk, dl = own[k], own[l]
    mx = np.maximum(dk, dl)
    ratio = np.divide(dens_u, mx, out=np.zeros(len(k)), where=mx > 0)
    dens_bw = 2 * ratio.sum() / (K * (K - 1))    # пары (k,l) и (l,k) дают одно и то же
    return float(scat + dens_bw)


def graph_indices(A, labels):
    y, H = _onehot(labels)
    K = H.shape[1]
    S = np.asarray(H.T @ (A @ H))                  # K×K
    tot, d = S.sum(1), np.diag(S).copy()
    two_m = S.sum()
    avi = np.divide(d, tot, out=np.zeros(K), where=tot > 0).mean()
    out = tot - d
    den = out[:, None] + out[None, :] - S
    U = np.divide(S, den, out=np.zeros_like(S), where=den > 0)
    np.fill_diagonal(U, 0)
    avu = U.sum() / K
    anui = 1 / (avu + 1 / avi) if avi > 0 else 0.0
    q = float((d / two_m - (tot / two_m) ** 2).sum()) if two_m > 0 else np.nan
    return dict(AVI=float(avi), AVU=float(avu), ANUI=float(anui), Q=q)


def feature_indices(X, labels, D=None, s_dbw_variant="paper"):
    return dict(SW=silhouette(X, labels, D), CH=calinski_harabasz(X, labels),
                S_Dbw=s_dbw(X, labels, s_dbw_variant))


def all_indices(X, A, labels, D=None, s_dbw_variant="paper"):
    out = feature_indices(X, labels, D, s_dbw_variant) if X is not None else {}
    if A is not None:
        out.update(graph_indices(A, labels))
    return out


def perm_z(X, A, labels, n_perm=200, seed=0, D=None, s_dbw_variant="paper", observed=None):
    """z-оценки всех индексов против перестановок меток. Возвращает {'z_SW': …, …}."""
    if n_perm <= 0:
        return {}
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    if X is not None and D is None:
        D = pairwise_dist(X)
    obs = observed or all_indices(X, A, labels, D, s_dbw_variant)
    null = [all_indices(X, A, rng.permutation(labels), D, s_dbw_variant) for _ in range(n_perm)]
    z = {}
    for key, v in obs.items():
        r = np.array([p[key] for p in null])
        sd = r.std()
        z[f"z_{key}"] = float((v - r.mean()) / sd) if sd > 0 else np.nan
    return z

