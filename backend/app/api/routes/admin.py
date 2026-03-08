"""Admin endpoints for authentication and audit."""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from backend.app.core.security import create_access_token, get_current_user
from backend.app.services.audit import get_recent_audit_logs

router = APIRouter()

# In production, use a proper user store. This is a placeholder.
ADMIN_USERS = {
    "admin": "change-me-in-production",
}


class TokenRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuditEntry(BaseModel):
    id: int
    action: str
    details: str | None
    created_at: str


@router.post("/token", response_model=TokenResponse)
async def generate_token(body: TokenRequest):
    """Generate JWT token for admin access."""
    stored_password = ADMIN_USERS.get(body.username)
    if stored_password is None or stored_password != body.password:
        logger.warning(f"Failed login attempt for user: {body.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token(data={"sub": body.username, "role": "admin"})
    logger.info(f"Token generated for user: {body.username}")
    return TokenResponse(access_token=token)


@router.get("/audit", response_model=list[AuditEntry])
async def audit_log(limit: int = 50, user: dict = Depends(get_current_user)):
    """Get recent audit log entries. Admin only."""
    entries = await get_recent_audit_logs(limit=limit)
    return [
        AuditEntry(
            id=e["id"],
            action=e["action"],
            details=e.get("details"),
            created_at=e["created_at"],
        )
        for e in entries
    ]
