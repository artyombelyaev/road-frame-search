# Почти одинаковые кадры по эмбеддингам CLIP (crop) среди всех 100K. Ничего не выкидываем, только
# записываем: на этапе 4 пары train - eval понадобятся, чтобы обучение не видело кадров из eval.
# Порог 0,98: у ближайшего соседа медиана косинуса 0,935, распределение спадает к 0,975, выше 0,98
# идёт отдельный хвост. У 98% пар выше 0,98 общий префикс имени (первые 8 символов, похоже, одна
# поездка), у пар от 0,97 до 0,98 - только у 11%: там в основном разные тёмные ночные кадры.

import numpy as np
import pandas as pd
import torch

from . import config, data, embed

BLOCK = 2000


def near_pairs(emb: np.ndarray, threshold: float) -> pd.DataFrame:
    e = torch.from_numpy(emb)
    e = e / e.norm(dim=1, keepdim=True)
    rows = []
    for start in range(0, len(e), BLOCK):
        sim = e[start:start + BLOCK] @ e.T
        i, j = torch.nonzero(sim >= threshold, as_tuple=True)
        i = i + start
        keep = i < j
        rows.append(pd.DataFrame({"i": i[keep].numpy(), "j": j[keep].numpy(),
                                  "cos": sim[i[keep] - start, j[keep]].numpy()}))
    return pd.concat(rows, ignore_index=True)


def main():
    frames = data.load_frames()
    pairs = near_pairs(embed.load_image_embeddings("crop", frames), config.DUP_COS)
    a, b = frames.iloc[pairs.i].reset_index(drop=True), frames.iloc[pairs.j].reset_index(drop=True)
    out = pd.DataFrame({
        "split_a": a.split, "name_a": a["name"], "part_a": a.part,
        "split_b": b.split, "name_b": b["name"], "part_b": b.part,
        "cos": pairs.cos.round(5), "same_prefix": a["name"].str[:8].values == b["name"].str[:8].values,
    }).sort_values("cos", ascending=False)
    out.to_csv(config.RESULTS / "near_duplicates.csv", index=False)

    is_train = lambda s: s == "train"
    cross = (is_train(out.split_a) & (out.part_b == "eval")) | (is_train(out.split_b) & (out.part_a == "eval"))
    print(f"пар с косинусом >= {config.DUP_COS}: {len(out)}, общий префикс имени у {out.same_prefix.mean():.0%}")
    print(f"пар train (все 70 000) - eval: {int(cross.sum())}")
    print(out.assign(pair=out.part_a + " - " + out.part_b).groupby("pair").size().to_string())


if __name__ == "__main__":
    main()
