"""Explicit device selection; an explicitly requested backend never silently falls back."""

import importlib.metadata
import platform
import sys

import torch


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Run 'uv run clickseg doctor' to inspect the environment."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use --device cpu or --device mps on Apple Silicon."
        )
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def environment_report(requested: str = "auto") -> dict:
    from clickseg.checkpoint import UPSTREAM_REV

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "timm": importlib.metadata.version("timm"),
        "mobile_sam_revision": UPSTREAM_REV,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(choose_device(requested)),
    }
