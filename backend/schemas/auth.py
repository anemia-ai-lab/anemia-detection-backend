"""Auth request/response contracts (API surface)."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    """``POST /auth/register`` body."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"email": "user@example.com", "password": "minimum8chars"}
        },
    )

    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    """``POST /auth/login`` body."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"email": "user@example.com", "password": "your-password"}
        },
    )

    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class UserOut(BaseModel):
    """Authenticated user subset exposed by the API."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "00000000-0000-0000-0000-000000000000",
                "email": "user@example.com",
                "created_at": "2026-04-01T12:00:00Z",
            }
        },
    )

    id: str = Field(description="Supabase Auth user id (`auth.users.id`).")
    email: Optional[str] = Field(default=None)
    created_at: Optional[datetime] = Field(default=None)


class WarningItem(BaseModel):
    """Aviso no bloqueante en respuestas de auth."""

    code: str
    message: str


class TokensOut(BaseModel):
    """OAuth-style session from Supabase (access + refresh)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "v1:...",
                "expires_in": 3600,
                "token_type": "bearer",
            }
        },
    )

    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str


class _AuthSessionBody(BaseModel):
    """Campos comunes a login y registro exitosos."""

    user: UserOut
    tokens: Optional[TokensOut] = Field(
        default=None,
        description="Omitted when Supabase returns no session (e.g. pending email).",
    )
    has_profile_row: Optional[bool] = Field(
        default=None,
        description=(
            "Misma semántica que en GET /auth/me/profile. "
            "null si no hay `access_token` (sin sesión: no se puede leer `profiles` con JWT)."
        ),
    )
    profile_completed: Optional[bool] = Field(
        default=None,
        description=(
            "Misma semántica que GET /auth/me/profile (columna persistida; se actualiza en PATCH). "
            "null si no hay `access_token`."
        ),
    )
    warnings: list[WarningItem] = Field(
        default_factory=list,
        description="Avisos estructurados (code + message).",
    )


class LoginAuthResponse(_AuthSessionBody):
    """``POST /auth/login`` — incluye estado de perfil cuando hay sesión (JWT)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user": {
                    "id": "00000000-0000-0000-0000-000000000000",
                    "email": "user@example.com",
                    "created_at": "2026-04-01T12:00:00Z",
                },
                "tokens": {
                    "access_token": "eyJ...",
                    "refresh_token": "v1:...",
                    "expires_in": 3600,
                    "token_type": "bearer",
                },
                "has_profile_row": True,
                "profile_completed": False,
                "warnings": [],
            }
        },
    )


class RegisterAuthResponse(_AuthSessionBody):
    """``POST /auth/register`` — incluye si en esta petición quedó fila en ``profiles``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user": {
                    "id": "00000000-0000-0000-0000-000000000000",
                    "email": "user@example.com",
                    "created_at": "2026-04-01T12:00:00Z",
                },
                "tokens": {
                    "access_token": "eyJ...",
                    "refresh_token": "v1:...",
                    "expires_in": 3600,
                    "token_type": "bearer",
                },
                "profile_created": True,
                "has_profile_row": True,
                "profile_completed": False,
                "warnings": [],
            }
        },
    )

    profile_created: bool = Field(
        description=(
            "True solo si en **este** alta se aseguró la fila en `profiles` (upsert). "
            "`has_profile_row` y `profile_completed` se leen con el JWT de la sesión."
        ),
    )
