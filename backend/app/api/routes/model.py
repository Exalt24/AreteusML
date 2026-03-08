"""Model management endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from backend.app.core.model_loader import get_model, model_state, reload_model
from backend.app.core.security import get_current_user

router = APIRouter()


class ModelInfo(BaseModel):
    model_name: str
    model_path: str
    backend: str
    loaded_at: str | None
    status: str


class ModelHealth(BaseModel):
    healthy: bool
    inference_ok: bool
    message: str


@router.get("/info", response_model=ModelInfo)
async def model_info():
    """Current model version and metadata."""
    state = model_state()
    return ModelInfo(
        model_name=state.get("model_name", "unknown"),
        model_path=state.get("model_path", "unknown"),
        backend=state.get("backend", "unknown"),
        loaded_at=state.get("loaded_at"),
        status=state.get("status", "unknown"),
    )


@router.get("/health", response_model=ModelHealth)
async def model_health():
    """Run a dummy prediction to verify model health."""
    try:
        model = get_model()
        tokenizer = model["tokenizer"]
        inputs = tokenizer("health check", return_tensors="np", truncation=True, max_length=128)

        if model["backend"] == "onnx":
            session = model["session"]
            ort_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
            session.run(None, ort_inputs)
        else:
            import torch

            pt_model = model["model"]
            pt_inputs = tokenizer("health check", return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                pt_model(**pt_inputs)

        return ModelHealth(healthy=True, inference_ok=True, message="Model is operational")
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        return ModelHealth(healthy=False, inference_ok=False, message=str(e))


@router.post("/reload")
async def model_reload(user: dict = Depends(get_current_user)):
    """Reload model from disk. Admin only."""
    logger.info(f"Model reload requested by {user.get('sub', 'unknown')}")
    try:
        reload_model()
        return {"status": "reloaded", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e
