import pandas as pd

from src import config, queries

FRAMES = pd.DataFrame({
    "weather":   ["rainy", "rainy", "clear", "undefined", "rainy"],
    "scene":     ["highway", "city street", "highway", "highway", "highway"],
    "timeofday": ["night", "night", "night", "night", "daytime"],
})


def q(**kw):
    return {a: kw.get(a) for a in config.ATTRS}


def test_all_queries_count():
    qs = queries.all_queries()
    # 15 одиночных, 6*6 + 6*3 + 6*3 пар, 6*6*3 троек
    assert qs.n_attrs.value_counts().to_dict() == {1: 15, 2: 72, 3: 108}
    assert qs.query_id.is_unique
    assert not (qs[config.ATTRS] == config.UNDEFINED).any().any()


def test_relevant_needs_all_attributes():
    mask = queries.relevant_mask(FRAMES, q(weather="rainy", timeofday="night"))
    assert mask.tolist() == [True, True, False, False, False]


def test_strict_gallery_drops_any_undefined():
    g = queries.gallery_mask(FRAMES, q(timeofday="night"), "strict")
    assert g.tolist() == [True, True, True, False, True]


def test_per_query_gallery_keeps_undefined_in_other_attributes():
    g = queries.gallery_mask(FRAMES, q(timeofday="night"), "per_query")
    assert g.all()
    g = queries.gallery_mask(FRAMES, q(weather="rainy"), "per_query")
    assert g.tolist() == [True, True, True, False, True]


def test_count_relevant():
    qs = pd.DataFrame([{"query_id": "x", "n_attrs": 1, **q(timeofday="night")}])
    strict = queries.count_relevant(FRAMES, qs, "strict").iloc[0]
    per_query = queries.count_relevant(FRAMES, qs, "per_query").iloc[0]
    assert (strict.n_relevant, strict.n_gallery) == (3, 4)
    assert (per_query.n_relevant, per_query.n_gallery) == (4, 5)
