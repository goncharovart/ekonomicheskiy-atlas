"""Методы членов жюри для трека «Кластеризация».

1. KEFRiN (Shalileh, Mirkin 2022) и CANUS (Shalileh 2025) — кластеризация сети с атрибутами.
   Код авторов без лицензии, поэтому в репозитории его нет: scripts/fetch_external.py
   клонирует его в external/, здесь только импорт.
2. Порядково-инвариантная паттерн-кластеризация (Алескеров, Мячин): паттерн объекта —
   знаки попарных сравнений его нормированных показателей; кластер = одинаковый паттерн.

load_shares и knn_graph ниже — первый прогон (scripts/run_jury.py: доли и косинусный kNN по тем же
долям). Основной прогон KEFRiN и CANUS — на общих признаках src/features.py и общем графе
src/network.py (kNN по корреляции рядов), его делает scripts/run_compare.py (configs/compare.yaml);
гиперпараметры обоих методов берутся из configs/jury.yaml.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external"
DATA = ROOT / "data" / "processed"


def _external(name):
    """Импорт модуля из клона репозитория автора (external/<name>)."""
    path = EXT / name
    if not path.exists():
        raise ImportError(f"нет {path}: запусти scripts/fetch_external.py")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ---------- данные и граф (временные, до унификации) ----------

def load_shares(categories):
    """Доли категорий: индекс (territory_id, month), колонки categories. Только сбалансированные МО."""
    mo = pd.read_parquet(DATA / "mo.parquet", columns=["balanced"])
    p = pd.read_parquet(DATA / "panel.parquet", columns=["territory_id", "month", "category", "share"])
    p = p[p.territory_id.isin(mo.index[mo.balanced]) & p.category.isin(categories)]
    return p.pivot_table(index=["territory_id", "month"], columns="category", values="share")[list(categories)]


def knn_graph(X, k=10):
    """Симметричный взвешенный kNN-граф по косинусному сходству строк X, без петель."""
    Z = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(X)), k)
    A = np.zeros_like(S)
    A[rows, idx.ravel()] = S[rows, idx.ravel()]
    return np.maximum(A, A.T)


def zscore(X):
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1)


# ---------- KEFRiN ----------

def kefrin(Y, A, n_clusters=8, distance="cosine", rho=1.0, xi=1.0, n_init=10, max_iter=300,
           preprocessing_y="z_score", preprocessing_p="none", seed=42):
    """KEFRiN из external/KEFRiN/kefrin.py (Sorooshi/KEFRiN). Y — признаки (N,V), A — связи (N,N)."""
    _external("KEFRiN")
    import kefrin as kf
    cfg = kf.KEFRiNConfig(
        n_clusters=n_clusters, rho=rho, xi=xi, distance_metric=kf.DistanceMetric(distance),
        max_iterations=max_iter, n_init=n_init, random_state=seed,
        preprocessing_y=kf.PreprocessingMethod(preprocessing_y),
        preprocessing_p=kf.PreprocessingMethod(preprocessing_p),
    )
    return kf.KEFRiN(cfg).fit_predict(np.asarray(Y, float), np.asarray(A, float)).astype(int)


# ---------- CANUS ----------

def _fast_grad_norms(self, x_vecs, a_vecs, k):
    """То же, что CANUSClusterer._per_sample_grad_norms для косинусных расстояний, но разом.

    В canus.py норма градиента считается циклом autograd по каждому узлу (N вызовов за эпоху).
    Для d = 1 - cos(v, c): ∂d/∂c = -(v̂ - cos·ĉ) / ||c||, норма = sqrt(1 - cos²) / ||c||.
    Совпадение с autograd проверяет tests/test_jury.py.
    """
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        out = torch.zeros(x_vecs.shape[0], device=x_vecs.device)
        for v, c in ((x_vecs, self.C_[k]), (a_vecs, self.L_[k])):
            vh, ch = F.normalize(v, dim=1, eps=1e-12), F.normalize(c, dim=0, eps=1e-12)
            cos = vh @ ch
            out += (vh - cos[:, None] * ch).norm(dim=1) / c.norm().clamp_min(1e-12)
    return out


def canus(X, A, n_clusters=8, fast_grad_norms=True, seed=42, **params):
    """CANUS из external/CANUS/canus.py (Sorooshi/CANUS). params — поля CANUSInit."""
    _external("CANUS")
    import torch
    from canus import CANUSClusterer
    torch.manual_seed(seed)  # бутстрэп полос фильтра в canus.py берёт глобальный генератор
    m = CANUSClusterer(n_clusters=n_clusters, seed=seed, device="cpu", **params)
    if fast_grad_norms:
        if m.cfg.attribute_distance != "cosine" or m.cfg.network_distance != "cosine":
            raise ValueError("fast_grad_norms только для косинусных расстояний")
        m._per_sample_grad_norms = _fast_grad_norms.__get__(m)
    return m.fit(np.asarray(X, np.float32), np.asarray(A, np.float32)).y_pred.astype(int)


# ---------- паттерн-кластеризация Алескерова–Мячина ----------

def pattern_pairs(n, mode="pairs"):
    """pairs — все пары j<k (порядок показателей, OIPC Мячина); adjacent — соседние оси (форма ломаной)."""
    if mode == "pairs":
        return [(j, k) for j in range(n) for k in range(j + 1, n)]
    if mode == "adjacent":
        return [(j, j + 1) for j in range(n - 1)]
    raise ValueError(mode)


def pattern_codes(X, mode="pairs", eps=0.0):
    """Строка кода объекта: знак x_j - x_k по парам (1 «>», -1 «<», 0 «=» при |разность| <= eps)."""
    X = np.asarray(X, float)
    D = np.stack([X[:, j] - X[:, k] for j, k in pattern_pairs(X.shape[1], mode)], axis=1)
    return np.where(np.abs(D) <= eps, 0, np.sign(D)).astype(np.int8)


def pattern_clusters(codes, count_mask=None):
    """Кластер = одинаковый код. Номера по убыванию частоты (0 — самый частый; частота по count_mask)."""
    uniq, inv = np.unique(codes, axis=0, return_inverse=True)
    inv = inv.ravel()
    w = np.ones(len(inv)) if count_mask is None else np.asarray(count_mask, float)
    order = np.lexsort((np.arange(len(uniq)), -np.bincount(inv, weights=w, minlength=len(uniq))))
    rank = np.empty(len(uniq), int)
    rank[order] = np.arange(len(uniq))
    return rank[inv], uniq[order]


def pattern_text(code, names, mode="pairs"):
    """Человеческая запись паттерна."""
    pairs = pattern_pairs(len(names), mode)
    sym = {1: ">", -1: "<", 0: "="}
    if mode == "adjacent":
        return names[0] + "".join(f" {sym[int(c)]} {names[k]}" for c, (_, k) in zip(code, pairs))
    score = np.zeros(len(names))
    for c, (j, k) in zip(code, pairs):
        score[j] += c
        score[k] -= c
    order = np.argsort(-score, kind="stable")
    return names[order[0]] + "".join(
        f" {'=' if score[a] == score[b] else '>'} {names[b]}" for a, b in zip(order, order[1:]))


def pattern_stability(labels):
    """labels: DataFrame (mo_id, month, cluster) по месяцам. Смены паттерна и модальный паттерн по МО."""
    wide = labels.pivot(index="mo_id", columns="month", values="cluster").sort_index(axis=1)
    w = wide.to_numpy()
    modal = pd.DataFrame(w).mode(axis=1)[0].to_numpy().astype(int)
    return pd.DataFrame({
        "mo_id": wide.index.to_numpy(),
        "n_changes": (w[:, 1:] != w[:, :-1]).sum(1),
        "n_patterns": [len(set(r)) for r in w],
        "modal_pattern": modal,
        "modal_share": (w == modal[:, None]).mean(1),
    })
