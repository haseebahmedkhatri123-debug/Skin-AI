"""Central configuration: every path and hyperparameter lives here.

Edit values in this file (or use the CLI overrides in train.py) rather than
hard-coding numbers in the other modules.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
CLASSES_PATH = MODELS_DIR / "classes.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# Public Hugging Face repo with the trained model, used when models/ is absent
# (e.g. on Streamlit Community Cloud).
HF_REPO_ID = "haseebahmed2453/skinai"
HF_MODEL_FILENAME = "best_model.pth"
HF_CLASSES_FILENAME = "classes.json"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
METADATA_FILENAME = "HAM10000_metadata.csv"

# dx code -> human-readable name. The ORDER here defines the class indices.
DX_TO_NAME = {
    "nv": "Melanocytic Nevus",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma",
}
CLASS_NAMES = list(DX_TO_NAME.values())
NUM_CLASSES = len(CLASS_NAMES)

# Group split (on lesion_id) proportions.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SEED = 42

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# Friendly name -> timm model id (with pretrained ImageNet weights).
# "efficientnetv2_s" in timm has no pretrained weights, so we map it to the
# TensorFlow-ported variant that does. The two light backbones are there for
# CPU-only machines, where convnext_tiny training is slow.
BACKBONES = {
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",                 # CPU-friendly
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",         # CPU-friendliest
}
BACKBONE = "convnext_tiny"
IMG_SIZE = 224  # 224 (default), 260 or 288

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BATCH_SIZE = 32
EPOCHS = 13
LR = 3e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1
FREEZE_EPOCHS = 1          # backbone frozen (head-only) for the first N epochs
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 4    # on validation balanced accuracy
USE_AMP = True             # used on CUDA only (CPU autocast is slower on most CPUs)

# --quick mode: sanity-check the pipeline end to end.
QUICK_FRACTION = 0.20
QUICK_EPOCHS = 3

# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
CONFIDENCE_THRESHOLD = 0.45
UNSURE_MESSAGE = "Not sure — please retake a clearer, close-up photo"
