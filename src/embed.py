# Эмбеддинги CLIP. Картинки считаются в Colab (notebooks/02_colab_embeddings.ipynb) один раз
# для всех 100K кадров, тексты - где угодно, на CPU это секунды.
# Кадр 16:9, а CLIP ждёт квадрат 224. crop - стандартная обработка CLIP: короткая сторона 224 и центр,
# по бокам теряется 44% ширины. pad - весь кадр с чёрными полями сверху и снизу.

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from . import config

MODES = ["crop", "pad"]


def load_model(device: str = "cpu"):
    model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(device).eval()
    return model, CLIPProcessor.from_pretrained(config.CLIP_MODEL)


def letterbox(img: Image.Image) -> Image.Image:
    side = max(img.size)
    out = Image.new("RGB", (side, side))
    out.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return out


def zip_frames(zip_path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".jpg")]
    df = pd.DataFrame({"member": members})
    parts = df.member.str.split("/")
    df["split"] = parts.str[1]
    df["name"] = parts.str[2].str[:-4]
    return df.sort_values(["split", "name"]).reset_index(drop=True)


# ZipFile открывается лениво, чтобы у каждого воркера DataLoader был свой.
class ZipImages(torch.utils.data.Dataset):
    def __init__(self, zip_path, members, mode, processor):
        self.zip_path, self.members, self.mode, self.processor = zip_path, list(members), mode, processor
        self.z = None

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        if self.z is None:
            self.z = zipfile.ZipFile(self.zip_path)
        img = Image.open(io.BytesIO(self.z.read(self.members[i]))).convert("RGB")
        if self.mode == "pad":
            img = letterbox(img)
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed_images(model, processor, zip_path, members, mode, device, batch_size=256, num_workers=2) -> np.ndarray:
    ds = ZipImages(zip_path, members, mode, processor)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    out = []
    for i, pixels in enumerate(loader):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            emb = model.get_image_features(pixel_values=pixels.to(device)).pooler_output
        out.append(normalize(emb.float()).cpu().numpy().astype(np.float16))
        if i % 50 == 0:
            print(f"{mode}: {min((i + 1) * batch_size, len(ds))} / {len(ds)}", flush=True)
    return np.concatenate(out)


@torch.no_grad()
def embed_texts(model, processor, texts: list[str]) -> np.ndarray:
    tokens = processor(text=texts, return_tensors="pt", padding=True)
    emb = model.get_text_features(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]).pooler_output
    return normalize(emb).numpy().astype(np.float32)


def emb_paths(mode: str, emb_dir: Path | None = None) -> tuple[Path, Path, Path]:
    stem = Path(emb_dir or config.EMB_DIR) / f"{config.CLIP_TAG}_{mode}"
    return stem.with_suffix(".npy"), Path(f"{stem}_names.parquet"), Path(f"{stem}_meta.json")


# Эмбеддинги в порядке строк frames. Порядок в файле не важен: сводим по (split, name).
def load_image_embeddings(mode: str, frames: pd.DataFrame, emb_dir: Path | None = None) -> np.ndarray:
    npy, names_path, meta_path = emb_paths(mode, emb_dir)
    meta = json.loads(meta_path.read_text())
    assert meta["model"] == config.CLIP_MODEL and meta["mode"] == mode, meta
    emb = np.load(npy)
    names = pd.read_parquet(names_path)
    pos = pd.Series(np.arange(len(names)), index=pd.MultiIndex.from_frame(names[["split", "name"]]))
    idx = pos.reindex(pd.MultiIndex.from_frame(frames[["split", "name"]])).values
    if np.isnan(idx).any():
        raise ValueError(f"нет эмбеддингов для {int(np.isnan(idx).sum())} кадров")
    return emb[idx.astype(int)].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--images-zip", default=str(config.IMAGES_ZIP))
    parser.add_argument("--out", default=str(config.EMB_DIR))
    parser.add_argument("--limit", type=int, help="только первые N кадров, для проверки")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = zip_frames(args.images_zip)
    if args.limit:
        frames = frames.sample(args.limit, random_state=config.SEED).sort_values(["split", "name"])
    model, processor = load_model(device)
    emb = embed_images(model, processor, args.images_zip, frames.member, args.mode, device, args.batch_size, args.workers)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npy, names_path, meta_path = emb_paths(args.mode, out)
    np.save(npy, emb)
    frames[["split", "name"]].to_parquet(names_path, index=False)
    meta_path.write_text(json.dumps({"model": config.CLIP_MODEL, "mode": args.mode, "n": len(frames)}))


if __name__ == "__main__":
    main()
