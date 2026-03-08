"""Reproducibility utilities for consistent ML results across runs."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("Using CPU (no CUDA device found)")
    return device


SEED = 42


def cleanup_mlflow_nulls(mlruns_dir: str = "mlruns") -> int:
    """Strip trailing null bytes from MLflow metric files.

    NTFS can leave null bytes when a process is killed mid-write (e.g. OOM
    during training). MLflow refuses to read these files. Call this after
    training or at startup to recover gracefully.

    Returns the number of files cleaned.
    """
    from pathlib import Path

    cleaned = 0
    for metric_file in Path(mlruns_dir).rglob("metrics/*"):
        if not metric_file.is_file():
            continue
        data = metric_file.read_bytes()
        if b"\x00" in data:
            metric_file.write_bytes(data.replace(b"\x00", b"").rstrip())
            cleaned += 1
    return cleaned
