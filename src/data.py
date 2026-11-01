# Таблица кадров из архива разметки: имя, сплит BDD100K, три атрибута и наша часть.
# Части: train_sub - подбор промптов, eval - val + test для финальных метрик, train - остальное.
# Разметка - bdd100k_labels.zip, в нём свой JSON на каждый кадр. Рамки объектов не используем,
# только считаем, сколько их, чтобы видеть, полная ли разметка у test.

import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from . import config

SPLIT_SIZES = {"train": 70_000, "val": 10_000, "test": 20_000}


def read_labels(path=config.LABELS_ZIP) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.endswith(".json"):
                continue
            rec = json.loads(z.read(member))
            attrs = rec.get("attributes", {})
            objects = [o for f in rec.get("frames", []) for o in f.get("objects", [])]
            rows.append({
                "name": rec["name"], "split": member.split("/")[1],
                **{a: attrs.get(a) for a in config.ATTRS},
                "has_all_attrs": all(a in attrs for a in config.ATTRS),
                "n_objects": len(objects),
                "n_box2d": sum("box2d" in o for o in objects),
            })
    return pd.DataFrame(rows).sort_values(["split", "name"]).reset_index(drop=True)


def image_members(path=config.IMAGES_ZIP) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.namelist() if m.endswith(".jpg")]
    df = pd.DataFrame({"member": members})
    parts = df.member.str.split("/")
    df["split"] = parts.str[1]
    df["name"] = parts.str[2].str[:-4]
    return df.sort_values(["split", "name"]).reset_index(drop=True)


# Сверка разметки с архивом картинок по каждому сплиту.
def check_labels(frames: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, expected in SPLIT_SIZES.items():
        f = frames[frames.split == split]
        img_names = set(images.name[images.split == split])
        rows.append({
            "split": split,
            "ожидается": expected,
            "json": len(f),
            "картинок": len(img_names),
            "имена_совпадают": set(f.name) == img_names,
            "json_без_атрибута": int((~f.has_all_attrs).sum()),
            "кадров_с_объектами": int((f.n_objects > 0).sum()),
            "кадров_с_box2d": int((f.n_box2d > 0).sum()),
            "box2d_на_кадр": round(f.n_box2d.mean(), 1),
        })
    return pd.DataFrame(rows).set_index("split")


def labels_ok(check: pd.DataFrame) -> bool:
    return bool(((check.json == check["ожидается"]) & (check["картинок"] == check["ожидается"])
                 & check["имена_совпадают"] & (check["json_без_атрибута"] == 0)).all())


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
    frames = read_labels()
    check = check_labels(frames, image_members())
    check.to_csv(config.RESULTS / "labels_check.csv")
    print(check.to_string())
    if not labels_ok(check):
        raise SystemExit("разметка не сходится с архивом картинок")
    frames = assign_parts(frames)
    frames.to_parquet(config.FRAMES_PATH, index=False)
    print(frames.groupby(["split", "part"]).size().to_string())


if __name__ == "__main__":
    main()
