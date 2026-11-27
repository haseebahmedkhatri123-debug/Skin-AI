"""Reusable single-image inference.

    from predict import predict
    print(predict("photo.jpg"))          # path or PIL.Image -> class name

CLI:
    python src/predict.py path/to/image.jpg
"""
import sys
import time
from functools import lru_cache
from pathlib import Path

import json

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import eval_transforms, load_rgb  # noqa: E402
from model import load_checkpoint, predict_tta  # noqa: E402


def ensure_model_files() -> tuple[Path, Path]:
    """Return local paths to best_model.pth and classes.json.

    If either file is missing locally (e.g. on Streamlit Community Cloud, where
    models/ is not in the git repo), download it from the public Hugging Face
    repo into models/. Later calls find the local copy and skip the download.
    """
    files = {config.BEST_MODEL_PATH: config.HF_MODEL_FILENAME,
             config.CLASSES_PATH: config.HF_CLASSES_FILENAME}
    for local_path, filename in files.items():
        if not local_path.exists():
            from huggingface_hub import hf_hub_download  # only needed for the download
            print(f"{local_path.name} not found locally; downloading from "
                  f"huggingface.co/{config.HF_REPO_ID} ...", flush=True)
            hf_hub_download(repo_id=config.HF_REPO_ID, filename=filename,
                            local_dir=config.MODELS_DIR)
    return config.BEST_MODEL_PATH, config.CLASSES_PATH


class Predictor:
    """Loads the model once (on CPU) and classifies images."""

    def __init__(self, checkpoint=config.BEST_MODEL_PATH, classes_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        # Prefer classes.json when given; fall back to the names stored in the checkpoint.
        if classes_path is not None and Path(classes_path).exists():
            self.classes = json.loads(Path(classes_path).read_text(encoding="utf-8"))
        else:
            self.classes = ckpt["classes"]
        self.transform = eval_transforms(ckpt["img_size"])

    def predict_proba(self, image) -> torch.Tensor:
        """Class probabilities (TTA-averaged) for one image (path or PIL)."""
        x = self.transform(load_rgb(image)).unsqueeze(0).to(self.device)
        return predict_tta(self.model, x)[0].cpu()

    def predict(self, image) -> str:
        """Predicted class name, or the 'not sure' message if confidence is low."""
        probs = self.predict_proba(image)
        conf, idx = probs.max(dim=0)
        if conf.item() < config.CONFIDENCE_THRESHOLD:
            return config.UNSURE_MESSAGE
        return self.classes[idx.item()]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    model_path, classes_path = ensure_model_files()
    return Predictor(model_path, classes_path)


def predict(image) -> str:
    """predict(image) -> class name (uses a cached CPU model)."""
    return get_predictor().predict(image)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    if len(sys.argv) != 2:
        sys.exit("usage: python src/predict.py <image>")
    p = get_predictor()
    p.predict(sys.argv[1])  # warm-up
    t0 = time.perf_counter()
    result = p.predict(sys.argv[1])
    print(f"{result}  ({(time.perf_counter() - t0) * 1000:.0f} ms on CPU)")
