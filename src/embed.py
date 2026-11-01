# Эмбеддинги CLIP. Картинки считаются в Colab (notebooks/02_colab_embeddings.ipynb) один раз
# для всех 100K кадров, тексты - где угодно, на CPU это секунды.
# Кадр 16:9, а CLIP ждёт квадрат 224. crop - стандартная обработка CLIP: короткая сторона 224 и центр,
# по бокам теряется 44% ширины. pad - весь кадр с чёрными полями сверху и снизу.
# Храню нормированные эмбеддинги во float16: 100 МБ на 100K кадров вместо 200.

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from . import config, data

MODES = ["crop", "pad"]
DTYPE = "float16"
PAD_COLOR = (0, 0, 0)
CHUNK_SIZE = 10_000


# Обработку кадра фиксирую на PIL: где есть torchvision (в Colab есть), transformers может выбрать
# другую реализацию ресайза, и эмбеддинги разойдутся с локальными.
def load_model(device: str = "cpu"):
    model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL, backend="pil")
    assert type(processor.image_processor).__name__ == "CLIPImageProcessorPil", type(processor.image_processor)
    return model, processor


def letterbox(img: Image.Image) -> Image.Image:
    side = max(img.size)
    out = Image.new("RGB", (side, side), PAD_COLOR)
    out.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return out


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
    for pixels in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            emb = model.get_image_features(pixel_values=pixels.to(device)).pooler_output
        out.append(normalize(emb.float()).cpu().numpy().astype(DTYPE))
    return np.concatenate(out)


@torch.no_grad()
def embed_texts(model, processor, texts: list[str]) -> np.ndarray:
    tokens = processor(text=texts, return_tensors="pt", padding=True)
    emb = model.get_text_features(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]).pooler_output
    return normalize(emb).numpy().astype(np.float32)


def emb_paths(mode: str, emb_dir: Path | None = None) -> tuple[Path, Path, Path]:
    stem = Path(emb_dir or config.EMB_DIR) / f"{config.CLIP_TAG}_{mode}"
    return stem.with_suffix(".npy"), Path(f"{stem}_names.parquet"), Path(f"{stem}_meta.json")


def save_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(tmp)
    os.replace(tmp, path)


# Проход кусками: каждый кусок сохраняется сразу, при перезапуске готовые куски пропускаются.
# Кусок считается готовым, только если его имена совпадают с нужными: иначе считаем заново.
def run_chunks(frames: pd.DataFrame, chunk_dir: Path, embed_fn, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, start in enumerate(range(0, len(frames), chunk_size)):
        chunk = frames.iloc[start:start + chunk_size]
        npy, names_path = chunk_dir / f"part_{i:03d}.npy", chunk_dir / f"part_{i:03d}_names.parquet"
        done = npy.exists() and names_path.exists() and pd.read_parquet(names_path).name.tolist() == chunk.name.tolist()
        if done:
            print(f"кусок {i}: уже есть", flush=True)
        else:
            emb = embed_fn(chunk.member)
            save_atomic(npy, lambda p: np.save(p, emb))
            save_atomic(names_path, lambda p: chunk[["split", "name"]].to_parquet(p, index=False))
            print(f"кусок {i}: {start + len(chunk)} / {len(frames)}", flush=True)
        parts.append(np.load(npy))
    return np.concatenate(parts)


def read_meta(mode: str, emb_dir: Path | None = None) -> dict:
    return json.loads(emb_paths(mode, emb_dir)[2].read_text())


# Эмбеддинги в порядке строк frames. Порядок в файле не важен: сводим по (split, name).
def load_image_embeddings(mode: str, frames: pd.DataFrame, emb_dir: Path | None = None) -> np.ndarray:
    npy, names_path, _ = emb_paths(mode, emb_dir)
    meta = read_meta(mode, emb_dir)
    assert meta["model"] == config.CLIP_MODEL and meta["mode"] == mode and meta["dtype"] == DTYPE, meta
    emb = np.load(npy)
    assert emb.dtype == DTYPE, emb.dtype
    names = pd.read_parquet(names_path)
    pos = pd.Series(np.arange(len(names)), index=pd.MultiIndex.from_frame(names[["split", "name"]]))
    idx = pos.reindex(pd.MultiIndex.from_frame(frames[["split", "name"]])).values
    if np.isnan(idx).any():
        raise ValueError(f"нет эмбеддингов для {int(np.isnan(idx).sum())} кадров")
    return emb[idx.astype(int)].astype(np.float32)


# Косинус между эмбеддингами одних и тех же кадров из двух прогонов.
def compare(mode: str, dir_a: Path, dir_b: Path) -> dict:
    names = pd.read_parquet(emb_paths(mode, dir_a)[1])
    a = load_image_embeddings(mode, names, dir_a)
    b = load_image_embeddings(mode, names, dir_b)
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return {"n": len(cos), "mean_cos": float(cos.mean()), "min_cos": float(cos.min())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--images-zip", default=str(config.IMAGES_ZIP))
    parser.add_argument("--out", default=str(config.EMB_DIR))
    parser.add_argument("--limit", type=int, help="только N случайных кадров (сид из config), для проверки")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compare-with", help="папка с эталонным прогоном тех же кадров")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    frames = data.image_members(args.images_zip)
    if args.limit:
        frames = frames.sample(args.limit, random_state=config.SEED).sort_values(["split", "name"]).reset_index(drop=True)
    model, processor = load_model(args.device)
    out = Path(args.out)
    npy, names_path, meta_path = emb_paths(args.mode, out)
    embed_fn = lambda members: embed_images(model, processor, args.images_zip, members, args.mode, args.device,
                                            args.batch_size, args.workers)
    emb = run_chunks(frames, out / f"{config.CLIP_TAG}_{args.mode}_chunks", embed_fn)

    save_atomic(npy, lambda p: np.save(p, emb))
    save_atomic(names_path, lambda p: frames[["split", "name"]].to_parquet(p, index=False))
    meta_path.write_text(json.dumps({
        "model": config.CLIP_MODEL, "mode": args.mode, "n": len(frames), "dtype": DTYPE,
        "image_processor": type(processor.image_processor).__name__,
        "pad_color": list(PAD_COLOR) if args.mode == "pad" else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": args.device, "autocast": "float16" if args.device == "cuda" else None,
    }, indent=1))
    print(f"готово: {npy}")

    if args.compare_with:
        res = compare(args.mode, out, Path(args.compare_with))
        print(f"сверка с {args.compare_with}: {res}")
        if res["min_cos"] <= config.CHECK_MIN_COS:
            sys.exit(f"минимальный косинус {res['min_cos']:.5f} не выше {config.CHECK_MIN_COS}, стоп")


if __name__ == "__main__":
    main()
