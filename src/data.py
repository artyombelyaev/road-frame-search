# Таблица кадров из архива разметки: имя, сплит BDD100K, три атрибута и наша часть.
# Части: train_sub - подбор промптов, eval - val + test для финальных метрик, train - остальное.
# В архиве на каждый кадр свой JSON, рамки объектов нам пока не нужны.

import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from . import config


def read_labels(path=config.LABELS_ZIP) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.endswith(".json"):
                continue
            rec = json.loads(z.read(member))
            rows.append({"name": rec["name"], "split": member.split("/")[1], **{a: rec["attributes"][a] for a in config.ATTRS}})
    return pd.DataFrame(rows).sort_values(["split", "name"]).reset_index(drop=True)


def assign_parts(frames: pd.DataFrame, n_train_sub: int = config.N_TRAIN_SUB, seed: int = config.SEED) -> pd.DataFrame:
    frames = frames.copy()
    frames["part"] = np.where(frames.split == "train", "train", "eval")
    train_idx = frames.index[frames.split == "train"]
    sub = np.random.default_rng(seed).choice(train_idx, n_train_sub, replace=False)
    frames.loc[sub, "part"] = "train_sub"
    return frames


def load_frames(part: str | None = None) -> pd.DataFrame:
    frames = pd.read_parquet(config.FRAMES_PATH)
    if part is None:
        return frames
    return frames[frames.part == part].reset_index(drop=True)


# Кадры читаем прямо из zip с картинками.
def read_image(z: zipfile.ZipFile, split: str, name: str) -> Image.Image:
    return Image.open(io.BytesIO(z.read(f"100k/{split}/{name}.jpg"))).convert("RGB")


def main():
    frames = assign_parts(read_labels())
    frames.to_parquet(config.FRAMES_PATH, index=False)
    print(frames.groupby(["split", "part"]).size().to_string())


if __name__ == "__main__":
    main()
