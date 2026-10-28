# Запросы - все сочетания из 1, 2 и 3 атрибутов с определёнными значениями.
# Релевантны кадры, у которых совпадают все атрибуты запроса.

import itertools

import numpy as np
import pandas as pd

from . import config

VALUES = {
    "weather": ["clear", "overcast", "partly cloudy", "rainy", "snowy", "foggy"],
    "scene": ["city street", "highway", "residential", "parking lot", "tunnel", "gas stations"],
    "timeofday": ["daytime", "night", "dawn/dusk"],
}


def query_id(q: dict) -> str:
    return "|".join(f"{a}={q[a]}" for a in config.ATTRS if q.get(a) is not None)


def all_queries() -> pd.DataFrame:
    rows = []
    for n in (1, 2, 3):
        for attrs in itertools.combinations(config.ATTRS, n):
            for values in itertools.product(*(VALUES[a] for a in attrs)):
                q = dict(zip(attrs, values))
                rows.append({"query_id": query_id(q), "n_attrs": n, **{a: q.get(a) for a in config.ATTRS}})
    return pd.DataFrame(rows)


def query_attrs(q) -> list[str]:
    return [a for a in config.ATTRS if isinstance(q[a], str)]


def relevant_mask(frames: pd.DataFrame, q) -> np.ndarray:
    mask = np.ones(len(frames), bool)
    for a in query_attrs(q):
        mask &= frames[a].values == q[a]
    return mask


# Галерея "strict" - кадры без undefined ни в одном атрибуте, одна на все запросы.
# Галерея "per_query" - для каждого запроса свои кадры: выкидываем только undefined в атрибутах запроса.
def gallery_mask(frames: pd.DataFrame, q, gallery: str) -> np.ndarray:
    attrs = config.ATTRS if gallery == "strict" else query_attrs(q)
    mask = np.ones(len(frames), bool)
    for a in attrs:
        mask &= frames[a].values != config.UNDEFINED
    return mask


def count_relevant(frames: pd.DataFrame, queries: pd.DataFrame, gallery: str) -> pd.DataFrame:
    out = queries.copy()
    n_rel, n_gal = [], []
    for q in queries.itertuples():
        g = gallery_mask(frames, q._asdict(), gallery)
        n_gal.append(int(g.sum()))
        n_rel.append(int((relevant_mask(frames, q._asdict()) & g).sum()))
    out["n_relevant"] = n_rel
    out["n_gallery"] = n_gal
    out["prevalence"] = out.n_relevant / out.n_gallery
    return out
