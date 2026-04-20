"""Profile contracts: email comes from Supabase Auth, not stored in ``profiles``."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ProfileOut(BaseModel):
    """Profile for the current user (email from JWT / GoTrue)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "00000000-0000-0000-0000-000000000000",
                "email": "user@example.com",
                "has_profile_row": True,
                "profile_completed": True,
                "first_name": "Ada",
                "last_name": "Lovelace",
                "department": "La Libertad",
                "province": "Trujillo",
                "created_at": "2026-04-19T12:00:00Z",
            }
        },
    )

    id: str
    email: Optional[str] = Field(
        default=None,
        description="Desde Supabase Auth; no se persiste en profiles.",
    )
    has_profile_row: bool = Field(
        description=(
            "Si existe fila en `profiles` para este usuario. "
            "False p. ej. si falló el insert al registrarse (luego se puede crear con PATCH)."
        ),
    )
    profile_completed: bool = Field(
        description=(
            "Regla de negocio persistida (p. ej. nombre y apellido); solo cambia en PATCH. "
            "Si no hay fila, se expone como false."
        ),
    )
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    department: Optional[str] = None
    province: Optional[str] = None
    created_at: Optional[datetime] = Field(
        default=None,
        description="Solo si existe fila en `profiles`.",
    )


class ProfilePatchRequest(BaseModel):
    """``PATCH /auth/me/profile`` body (campos opcionales)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "first_name": "Ada",
                "last_name": "Lovelace",
                "department": "La Libertad",
                "province": "Trujillo",
            }
        },
    )

    first_name: Optional[str] = Field(default=None, max_length=200)
    last_name: Optional[str] = Field(default=None, max_length=200)
    department: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Texto libre; en Perú suele ser el departamento (ej. La Libertad).",
    )
    province: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Texto libre; en Perú suele ser la provincia (ej. Trujillo).",
    )
