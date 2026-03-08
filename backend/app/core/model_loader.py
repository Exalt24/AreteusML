"""Model loading with ONNX primary and PyTorch fallback. Singleton pattern."""

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from transformers import AutoTokenizer

from backend.app.core.config import settings
from ml.training.labels import LABEL_NAMES

_model_instance: dict | None = None
_loaded_at: str | None = None
_backend: str | None = None


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (where ml/ lives)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "ml").is_dir():
            return parent
    return Path.cwd()


def _load_onnx(onnx_dir: Path, tokenizer):
    """Load ONNX model via onnxruntime."""
    import onnxruntime as ort

    onnx_files = list(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files found in {onnx_dir}")

    model_path = onnx_files[0]
    logger.info(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    label_map = {str(i): name for i, name in enumerate(LABEL_NAMES)}

    return {
        "session": session,
        "tokenizer": tokenizer,
        "backend": "onnx",
        "label_map": label_map,
        "model_path": str(model_path),
    }


def _load_pytorch(model_dir: Path, tokenizer):
    """Load PyTorch model as fallback."""
    from transformers import AutoModelForSequenceClassification

    logger.info(f"Loading PyTorch model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    label_map = {str(i): name for i, name in enumerate(LABEL_NAMES)}

    return {
        "model": model,
        "tokenizer": tokenizer,
        "backend": "pytorch",
        "label_map": label_map,
        "model_path": str(model_dir),
    }


def get_model() -> dict:
    """Get or load the model singleton."""
    global _model_instance, _loaded_at, _backend

    if _model_instance is not None:
        return _model_instance

    root = _find_project_root()
    onnx_dir = root / settings.onnx_model_path
    pytorch_dir = root / settings.model_path

    # Load tokenizer from whichever path exists
    tokenizer_path = pytorch_dir if pytorch_dir.exists() else onnx_dir
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"No model directory found at {pytorch_dir} or {onnx_dir}")

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    # Try ONNX first, fall back to PyTorch
    try:
        _model_instance = _load_onnx(onnx_dir, tokenizer)
        _backend = "onnx"
    except Exception as e:
        logger.warning(f"ONNX load failed ({e}), falling back to PyTorch")
        _model_instance = _load_pytorch(pytorch_dir, tokenizer)
        _backend = "pytorch"

    _loaded_at = datetime.now(UTC).isoformat()
    logger.info(f"Model loaded via {_backend} backend")
    return _model_instance


def reload_model():
    """Force reload the model."""
    global _model_instance
    _model_instance = None
    return get_model()


def cleanup_model():
    """Cleanup model resources."""
    global _model_instance, _loaded_at, _backend
    _model_instance = None
    _loaded_at = None
    _backend = None
    logger.info("Model resources cleaned up")


def model_state() -> dict:
    """Return current model state metadata."""
    return {
        "model_name": settings.model_name,
        "model_path": _model_instance["model_path"] if _model_instance else "not loaded",
        "backend": _backend or "not loaded",
        "loaded_at": _loaded_at,
        "status": "loaded" if _model_instance else "not loaded",
    }
