# Пути и общие константы.

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Архивы BDD100K лежат в data/ или в папке из BDD_DATA_DIR. Распаковывать их не нужно.
DATA_DIR = Path(os.environ.get("BDD_DATA_DIR", ROOT / "data"))
LABELS_ZIP = DATA_DIR / "bdd100k_labels.zip"
IMAGES_ZIP = DATA_DIR / "bdd100k_images_100k.zip"

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
RESULTS = ROOT / "results"

FRAMES_PATH = ARTIFACTS / "frames.parquet"

SEED = 42

# Подбор промптов идёт на подвыборке train того же размера, что и оценочная часть (val + test),
# чтобы порог по числу релевантных кадров значил одно и то же на обеих.
N_TRAIN_SUB = 30_000

ATTRS = ["weather", "scene", "timeofday"]
UNDEFINED = "undefined"
