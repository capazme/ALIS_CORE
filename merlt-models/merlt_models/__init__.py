"""
MERL-T Models
Proprietary trained models and configurations

This package provides utilities to load trained models
for the MERL-T framework.

Usage:
    from merlt_models import load_model, load_config

    model = load_model("expert_literal_v1")
    config = load_config("rlcf_training")
"""

__version__ = "0.1.0"

from pathlib import Path
from typing import Optional, Dict, Any
import yaml

PACKAGE_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = PACKAGE_DIR / "weights"
CONFIG_DIR = PACKAGE_DIR / "config"
CHECKPOINTS_DIR = PACKAGE_DIR / "checkpoints"


def load_config(name: str) -> Dict[str, Any]:
    """
    Load a configuration file by name

    Args:
        name: Config name (without .yaml extension)

    Returns:
        Configuration dictionary
    """
    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        # Try in subdirectories
        for subdir in CONFIG_DIR.iterdir():
            if subdir.is_dir():
                config_path = subdir / f"{name}.yaml"
                if config_path.exists():
                    break
        else:
            raise FileNotFoundError(f"Config {name} not found in {CONFIG_DIR}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(
    name: str,
    device: Optional[str] = None,
    **kwargs
):
    """
    Load a trained model by name

    Args:
        name: Model name
        device: Device to load model on (cpu/cuda/mps)
        **kwargs: Additional arguments for model loading

    Returns:
        Loaded model (type depends on model format)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to load models. Install with: pip install torch")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Try different file extensions
    extensions = [".safetensors", ".pt", ".pth", ".bin", ".ckpt"]
    model_path = None

    for ext in extensions:
        candidate = WEIGHTS_DIR / f"{name}{ext}"
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        # Check checkpoints
        for ext in extensions:
            candidate = CHECKPOINTS_DIR / f"{name}{ext}"
            if candidate.exists():
                model_path = candidate
                break

    if model_path is None:
        available = list(WEIGHTS_DIR.glob("*")) + list(CHECKPOINTS_DIR.glob("*"))
        raise FileNotFoundError(
            f"Model {name} not found. Available: {[p.stem for p in available]}"
        )

    # Load based on extension
    if model_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            return state_dict
        except ImportError:
            raise ImportError("safetensors required. Install with: pip install safetensors")
    else:
        return torch.load(model_path, map_location=device, **kwargs)


def list_models() -> list:
    """List all available models"""
    models = []
    for path in WEIGHTS_DIR.glob("*"):
        if path.suffix in [".safetensors", ".pt", ".pth", ".bin", ".ckpt"]:
            models.append(path.stem)
    for path in CHECKPOINTS_DIR.glob("*"):
        if path.suffix in [".safetensors", ".pt", ".pth", ".bin", ".ckpt"]:
            models.append(f"checkpoint/{path.stem}")
    return sorted(set(models))


def list_configs() -> list:
    """List all available configs"""
    configs = []
    for path in CONFIG_DIR.rglob("*.yaml"):
        relative = path.relative_to(CONFIG_DIR)
        configs.append(str(relative.with_suffix("")))
    return sorted(configs)


__all__ = [
    "load_model",
    "load_config",
    "list_models",
    "list_configs",
    "WEIGHTS_DIR",
    "CONFIG_DIR",
    "CHECKPOINTS_DIR",
]
