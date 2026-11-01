import numpy as np
import pandas as pd

from src import embed


def make_frames(n):
    names = [f"f{i:03d}" for i in range(n)]
    return pd.DataFrame({"member": [f"100k/val/{x}.jpg" for x in names], "split": "val", "name": names})


def fake_embed(calls):
    def fn(members):
        calls.append(len(members))
        return np.array([[int(m.split("/")[-1][1:4])] * 2 for m in members], np.float16)
    return fn


def test_chunks_cover_all_frames_in_order(tmp_path):
    calls = []
    emb = embed.run_chunks(make_frames(25), tmp_path, fake_embed(calls), chunk_size=10)
    assert calls == [10, 10, 5]
    assert emb[:, 0].tolist() == list(range(25))


def test_restart_skips_finished_chunks(tmp_path):
    embed.run_chunks(make_frames(25), tmp_path, fake_embed([]), chunk_size=10)
    (tmp_path / "part_002.npy").unlink()
    calls = []
    emb = embed.run_chunks(make_frames(25), tmp_path, fake_embed(calls), chunk_size=10)
    assert calls == [5]
    assert emb[:, 0].tolist() == list(range(25))


def test_chunk_with_other_frames_is_recomputed(tmp_path):
    embed.run_chunks(make_frames(25), tmp_path, fake_embed([]), chunk_size=10)
    calls = []
    other = make_frames(30).iloc[5:].reset_index(drop=True)
    emb = embed.run_chunks(other, tmp_path, fake_embed(calls), chunk_size=10)
    assert calls == [10, 10, 5]
    assert emb[:, 0].tolist() == list(range(5, 30))


def test_letterbox_keeps_whole_frame():
    img = embed.Image.new("RGB", (16, 9), (255, 255, 255))
    out = np.asarray(embed.letterbox(img))
    assert out.shape == (16, 16, 3)
    assert out[:3].max() == 0 and out[-4:].max() == 0
    assert out[3:12].min() == 255
