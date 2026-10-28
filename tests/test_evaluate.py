import numpy as np
import pandas as pd
import pytest

from src import config, evaluate, prompts, queries

FRAMES = pd.DataFrame({
    "weather":   ["rainy", "rainy", "clear", "undefined", "clear", "rainy"],
    "scene":     ["highway", "city street", "highway", "highway", "city street", "highway"],
    "timeofday": ["night", "night", "night", "night", "daytime", "daytime"],
})


def make_queries(*qs):
    rows = [{a: q.get(a) for a in config.ATTRS} for q in qs]
    for r in rows:
        r["query_id"] = queries.query_id(r)
        r["n_attrs"] = len(queries.query_attrs(r))
    return pd.DataFrame(rows)


def test_query_text():
    q = {"weather": "rainy", "scene": "highway", "timeofday": "night"}
    assert prompts.query_text(q, "tags") == "night, rainy, highway"
    assert prompts.query_text(q, "sentence") == "a photo of a highway in the rain at night"
    assert prompts.query_text({"timeofday": "night"}, "dashcam") == "a dashcam photo of a road at night"


def test_perfect_scores_give_perfect_metrics():
    qs = make_queries({"weather": "rainy", "timeofday": "night"})
    rel = queries.relevant_mask(FRAMES, qs.iloc[0].to_dict()).astype(np.float32)
    res = evaluate.evaluate_scores(FRAMES, qs, rel[:, None]).iloc[0]
    assert res.ap == 1.0 and res.r_prec == 1.0
    assert res.n_relevant == 2
    # weather undefined у кадра 3 - он выпадает из галереи этого запроса
    assert res.n_gallery == 5
    assert res.rand_ap == pytest.approx(evaluate.metrics.random_average_precision(2, 5))


def test_miss_shares():
    qs = make_queries({"weather": "rainy", "timeofday": "night"})
    # лучше всего оценены кадры 2 (clear/night) и 5 (rainy/daytime)
    scores = np.array([0, 0, 5, 0, 0, 4], np.float32)[:, None]
    res = evaluate.evaluate_scores(FRAMES, qs, scores).iloc[0]
    assert res["p@10"] == pytest.approx(2 / 5)
    assert np.isnan(res.miss_scene)
    # в топ-5 все пять кадров галереи: погода не совпала у 2 и 4, время - у 4 и 5
    assert res.miss_weather == pytest.approx(2 / 5)
    assert res.miss_timeofday == pytest.approx(2 / 5)


def test_per_attr_scoring_sums_log_probs():
    rng = np.random.default_rng(0)
    texts = evaluate.all_texts(make_queries({"weather": "rainy", "timeofday": "night"}), "sentence")
    text_emb = {t: rng.normal(size=8).astype(np.float32) for t in texts}
    img = rng.normal(size=(4, 8)).astype(np.float32)
    q2 = make_queries({"weather": "rainy", "timeofday": "night"}, {"weather": "rainy"}, {"timeofday": "night"})
    s = evaluate.score_matrix(img, q2, "sentence", "per_attr", text_emb)
    assert np.allclose(s[:, 0], s[:, 1] + s[:, 2], atol=1e-4)
    assert (s <= 0).all()
