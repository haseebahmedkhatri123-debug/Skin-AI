"""Model construction, device selection, checkpoint I/O, and TTA."""
import sys
from pathlib import Path

import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def get_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(backbone: str = config.BACKBONE, pretrained: bool = True) -> torch.nn.Module:
    """Create a timm backbone with a fresh classification head."""
    timm_name = config.BACKBONES.get(backbone, backbone)  # allow raw timm ids too
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=config.NUM_CLASSES)


def load_checkpoint(path=config.BEST_MODEL_PATH, device="cpu"):
    """Rebuild the model from a checkpoint saved by train.py.

    Always deserialized onto the CPU first (works on machines without the GPU
    the checkpoint was trained on), then moved to `device`.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_tta(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Test-time augmentation: average softmax of original + horizontal flip."""
    probs = model(x).softmax(dim=1)
    probs += model(torch.flip(x, dims=[3])).softmax(dim=1)
    return probs / 2
