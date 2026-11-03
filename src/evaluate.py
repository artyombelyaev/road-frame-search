# Zero-shot оценка CLIP. На train_sub сравниваю все варианты и выбираю лучший по mAP,
# на eval считаю только выбранный: без файла с выбором eval не запускается.
# Вариант = обработка кадра (crop / pad) x шаблон x способ скоринга:
#   joint - косинус кадра с эмбеддингом всего запроса;
#   per_attr - по каждому атрибуту запроса CLIP как классификатор среди всех значений этого атрибута,
#              сумма log-вероятностей. Так составной запрос не зависит от того, как CLIP склеивает слова.

import argparse
import json

import numpy as np
import pandas as pd
import torch

from . import config, data, embed, metrics, prompts, queries

SCORINGS = ["joint", "per_attr"]
CHOICE_PATH = config.RESULTS / "zs_choice.json"
TOP_MISS = 50


def text_embeddings(model, processor, texts: list[str]) -> dict:
    texts = list(dict.fromkeys(texts))
    return dict(zip(texts, embed.embed_texts(model, processor, texts)))


def all_texts(qs: pd.DataFrame, template: str) -> list[str]:
    texts = [prompts.query_text(q, template) for q in qs.to_dict("records")]
    for a in config.ATTRS:
        texts += [prompts.query_text({a: v}, template) for v in queries.VALUES[a]]
    return texts


def score_matrix(img: np.ndarray, qs: pd.DataFrame, template: str, scoring: str, text_emb: dict) -> np.ndarray:
    if scoring == "joint":
        t = np.stack([text_emb[prompts.query_text(q, template)] for q in qs.to_dict("records")])
        return img @ t.T
    log_probs = {}
    for a in config.ATTRS:
        t = np.stack([text_emb[prompts.query_text({a: v}, template)] for v in queries.VALUES[a]])
        lp = torch.log_softmax(torch.from_numpy(config.LOGIT_SCALE * img @ t.T), dim=1).numpy()
        log_probs[a] = dict(zip(queries.VALUES[a], lp.T))
    scores = np.zeros((len(img), len(qs)), np.float32)
    for j, q in enumerate(qs.to_dict("records")):
        for a in queries.query_attrs(q):
            scores[:, j] += log_probs[a][q[a]]
    return scores


# Метрики по каждому запросу, рядом ожидание для случайной выдачи. miss_<атрибут> - доля кадров
# в топ-50, у которых этот атрибут запроса не совпал, rand_miss_<атрибут> - такая же доля во всей галерее
# запроса, то есть у случайной выдачи.
def evaluate_scores(frames: pd.DataFrame, qs: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, q in enumerate(qs.to_dict("records")):
        g = queries.gallery_mask(frames, q, config.GALLERY)
        rel = queries.relevant_mask(frames, q)[g]
        order = np.argsort(-scores[g, j], kind="stable")
        rel_sorted = rel[order].astype(np.int8)
        row = {"query_id": q["query_id"], "n_attrs": q["n_attrs"], "n_relevant": int(rel.sum()), "n_gallery": int(g.sum())}
        row.update(metrics.ranking_metrics(rel_sorted, config.KS))
        row.update({f"rand_{k}": v for k, v in metrics.random_metrics(row["n_relevant"], row["n_gallery"], config.KS).items()})
        gallery = frames[g]
        top = gallery.iloc[order[:TOP_MISS]]
        for a in config.ATTRS:
            row[f"miss_{a}"] = float((top[a].values != q[a]).mean()) if isinstance(q[a], str) else np.nan
            row[f"rand_miss_{a}"] = float((gallery[a].values != q[a]).mean()) if isinstance(q[a], str) else np.nan
        rows.append({**{a: q[a] for a in config.ATTRS}, **row})
    return pd.DataFrame(rows)


METRIC_COLS = [f"p@{k}" for k in config.KS] + ["r_prec", "ap"]


def summary(per_query: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    cols = METRIC_COLS + [f"rand_{c}" for c in METRIC_COLS]
    if by is None:
        out = per_query[cols].mean().to_frame().T
        out.insert(0, "n_queries", len(per_query))
        return out
    out = per_query.groupby(by)[cols].mean()
    out.insert(0, "n_queries", per_query.groupby(by).size())
    return out


# Разбивка по значениям атрибутов: все запросы, где есть это значение.
def summary_by_value(per_query: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for a in config.ATTRS:
        for v in queries.VALUES[a]:
            sub = per_query[per_query[a] == v]
            if len(sub):
                parts.append(summary(sub).assign(attr=a, value=v, n_relevant_min=sub.n_relevant.min(),
                                                 n_relevant_median=sub.n_relevant.median()))
    return pd.concat(parts, ignore_index=True).set_index(["attr", "value"])


# Промахи по атрибутам для запросов из 2 и 3 атрибутов. ratio = miss / rand_miss по средним:
# 1 - не лучше случайной выдачи, чем меньше, тем реже CLIP ошибается в этом атрибуте.
def miss_summary(per_query: pd.DataFrame) -> pd.DataFrame:
    multi = per_query[per_query.n_attrs > 1]
    rows = []
    for n, sub in multi.groupby("n_attrs"):
        for a in config.ATTRS:
            m, r = sub[f"miss_{a}"].dropna(), sub[f"rand_miss_{a}"].dropna()
            rows.append({"n_attrs": n, "attr": a, "n_queries": len(m), "miss": m.mean(), "rand_miss": r.mean(),
                         "ratio": m.mean() / r.mean()})
    return pd.DataFrame(rows)


def available_modes() -> list[str]:
    return [m for m in embed.MODES if embed.emb_paths(m)[0].exists()]


def run_variants(frames, qs, variants, model, processor) -> pd.DataFrame:
    texts = sum((all_texts(qs, t) for t in prompts.TEMPLATES), [])
    text_emb = text_embeddings(model, processor, texts)
    out = []
    img_cache = {}
    for mode, template, scoring in variants:
        if mode not in img_cache:
            img_cache[mode] = embed.load_image_embeddings(mode, frames)
        per_query = evaluate_scores(frames, qs, score_matrix(img_cache[mode], qs, template, scoring, text_emb))
        out.append(per_query.assign(mode=mode, template=template, scoring=scoring))
    return pd.concat(out, ignore_index=True)


VARIANT = ["mode", "template", "scoring"]


# Парный бутстрэп по запросам: запросы берутся с возвращением, оба варианта считаются на одной выборке.
def paired_bootstrap(ap_a, ap_b, n_boot: int = config.N_BOOT, seed: int = config.SEED) -> tuple[float, float, float]:
    diff = np.asarray(ap_a) - np.asarray(ap_b)
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(n_boot, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    assert lo <= diff.mean() <= hi, (diff.mean(), lo, hi)
    return float(diff.mean()), float(lo), float(hi)


# Разница mAP "выбранный минус другой" со вторым вариантом и с вариантами, где изменён один фактор,
# а два других остались как у выбранного.
def bootstrap_comparisons(per_query: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    ap = per_query.pivot_table(index="query_id", columns=VARIANT, values="ap")
    best, second = table.index[0], table.index[1]
    pairs = [("1-й против 2-го", second)]
    for i, factor in enumerate(VARIANT):
        for level in sorted(set(ap.columns.get_level_values(i)) - {best[i]}):
            pairs.append((factor, tuple(level if j == i else best[j] for j in range(len(VARIANT)))))
    rows = []
    for name, other in pairs:
        diff, lo, hi = paired_bootstrap(ap[best].values, ap[other].values)
        rows.append({"сравнение": name, "выбранный": " / ".join(best), "другой": " / ".join(other),
                     "выбранный_минус_другой": diff, "ci_2.5": lo, "ci_97.5": hi, "ноль_в_интервале": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def select_on_train(model, processor):
    frames = data.load_frames("train_sub")
    qs = queries.load_queries()
    variants = [(m, t, s) for m in available_modes() for t in prompts.TEMPLATES for s in SCORINGS]
    per_query = run_variants(frames, qs, variants, model, processor)
    per_query.to_csv(config.RESULTS / "zs_train_sub_per_query.csv", index=False)
    table = summary(per_query, VARIANT).sort_values("ap", ascending=False)
    table.to_csv(config.RESULTS / "zs_train_sub_variants.csv")
    best = table.index[0]
    CHOICE_PATH.write_text(json.dumps(dict(zip(["mode", "template", "scoring"], best)), indent=1))
    boot = bootstrap_comparisons(per_query, table)
    boot.to_csv(config.RESULTS / "zs_train_sub_bootstrap.csv", index=False)
    print(table[["n_queries"] + METRIC_COLS].round(4).to_string())
    print("выбран:", best)
    print(boot.round(4).to_string())


def final_on_eval(model, processor):
    choice = json.loads(CHOICE_PATH.read_text())
    frames = data.load_frames("eval")
    qs = queries.load_queries()
    per_query = run_variants(frames, qs, [(choice["mode"], choice["template"], choice["scoring"])], model, processor)
    per_query.to_csv(config.RESULTS / "zs_eval_per_query.csv", index=False)
    summary(per_query, "n_attrs").to_csv(config.RESULTS / "zs_eval_by_type.csv")
    summary_by_value(per_query).to_csv(config.RESULTS / "zs_eval_by_value.csv")
    miss_summary(per_query).to_csv(config.RESULTS / "zs_eval_miss.csv", index=False)
    print(pd.concat([summary(per_query).assign(n_attrs="все").set_index("n_attrs"), summary(per_query, "n_attrs")]).round(4).to_string())


# Проверка после выбора, только train_sub, выбор не меняет: почему per_attr лучше joint и на запросах
# из одного атрибута. joint_centered - косинус кадра с запросом минус средний косинус кадра со всеми
# значениями того же атрибута. Если он догоняет per_attr, выигрыш даёт поправка на сам кадр:
# у некоторых кадров косинус высокий со всеми текстами сразу.
def centering_check(model, processor):
    choice = json.loads(CHOICE_PATH.read_text())
    frames = data.load_frames("train_sub")
    qs = queries.load_queries()
    qs = qs[qs.n_attrs == 1].reset_index(drop=True)
    img = embed.load_image_embeddings(choice["mode"], frames)
    text_emb = text_embeddings(model, processor, all_texts(qs, choice["template"]))
    joint = score_matrix(img, qs, choice["template"], "joint", text_emb)
    centered = joint.copy()
    for j, q in enumerate(qs.to_dict("records")):
        a = queries.query_attrs(q)[0]
        t = np.stack([text_emb[prompts.query_text({a: v}, choice["template"])] for v in queries.VALUES[a]])
        centered[:, j] -= (img @ t.T).mean(axis=1)
    per_attr = score_matrix(img, qs, choice["template"], "per_attr", text_emb)
    ap = pd.DataFrame({name: evaluate_scores(frames, qs, sc).ap.values
                       for name, sc in [("joint", joint), ("joint_centered", centered), ("per_attr", per_attr)]},
                      index=qs.query_id)
    ap.to_csv(config.RESULTS / "zs_train_sub_centering_per_query.csv")
    rows = []
    for a, b in [("per_attr", "joint"), ("joint_centered", "joint"), ("per_attr", "joint_centered")]:
        diff, lo, hi = paired_bootstrap(ap[a].values, ap[b].values)
        rows.append({"сравнение": f"{a} минус {b}", "разница_mAP": diff, "ci_2.5": lo, "ci_97.5": hi,
                     "ноль_в_интервале": bool(lo <= 0 <= hi)})
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS / "zs_train_sub_centering.csv", index=False)
    print(ap.mean().round(4).to_string())
    print(out.round(4).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["train_sub", "eval", "centering_check"], required=True)
    args = parser.parse_args()
    model, processor = embed.load_model()
    if args.part == "train_sub":
        select_on_train(model, processor)
    elif args.part == "eval":
        final_on_eval(model, processor)
    else:
        centering_check(model, processor)


if __name__ == "__main__":
    main()
