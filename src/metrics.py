# Метрики по ранжированной выдаче с бинарной релевантностью.
# rel - массив из 0 и 1 в порядке выдачи, вся галерея целиком.

import numpy as np


def precision_at_k(rel: np.ndarray, k: int) -> float:
    return float(rel[:k].mean())


# R-precision - точность в топ-R, где R - число релевантных. При ней P = полнота.
def r_precision(rel: np.ndarray) -> float:
    r = int(rel.sum())
    return float(rel[:r].mean()) if r else 0.0


def average_precision(rel: np.ndarray) -> float:
    r = rel.sum()
    if r == 0:
        return 0.0
    hits = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1)
    return float((hits / ranks)[rel.astype(bool)].sum() / r)


# Ожидание при случайной выдаче. Для P@k и R-precision это доля релевантных R/N.
# Для AP: релевантный кадр стоит на равновероятной позиции r, над ним в среднем (r-1)(R-1)/(N-1)
# других релевантных, E[1/r] = H_N / N.
def random_precision(n_relevant: int, n_gallery: int) -> float:
    return n_relevant / n_gallery


def random_average_precision(n_relevant: int, n_gallery: int) -> float:
    R, N = n_relevant, n_gallery
    if N == 1:
        return float(R)
    harmonic = np.sum(1.0 / np.arange(1, N + 1))
    return float((R - 1) / (N - 1) + harmonic / N * (N - R) / (N - 1))


def ranking_metrics(rel: np.ndarray, ks=(10, 50)) -> dict:
    out = {f"p@{k}": precision_at_k(rel, k) for k in ks}
    out["r_prec"] = r_precision(rel)
    out["ap"] = average_precision(rel)
    return out


def random_metrics(n_relevant: int, n_gallery: int, ks=(10, 50)) -> dict:
    p = random_precision(n_relevant, n_gallery)
    out = {f"p@{k}": p for k in ks}
    out["r_prec"] = p
    out["ap"] = random_average_precision(n_relevant, n_gallery)
    return out
