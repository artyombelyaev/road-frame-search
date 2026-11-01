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

# Для каждого запроса выкидываем только кадры с undefined в его атрибутах: у 88% туннелей
# погода undefined, общая галерея без undefined оставила бы от них 8 из 76.
GALLERY = "per_query"
# Запрос оставляем, если релевантных не меньше порога и в eval, и в train_sub.
MIN_RELEVANT = 50
QUERIES_PATH = ARTIFACTS / "queries.parquet"

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_TAG = "clip_b32"
# Эмбеддинги картинок из Colab кладём сюда.
EMB_DIR = ARTIFACTS / "emb"
# Сверка Colab с эталоном на CPU (те же 256 кадров): если минимальный косинус не выше порога - стоп.
CHECK_MIN_COS = 0.999
# Масштаб логитов CLIP (logit_scale.exp() у ViT-B/32 равен 100), нужен для softmax по значениям атрибута.
LOGIT_SCALE = 100.0
KS = (10, 50)
# Повторов парного бутстрэпа по запросам при сравнении вариантов.
N_BOOT = 10_000
