"""Shared API error envelope (OpenAPI contracts)."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """JSON body for failed auth and other domain errors handled explicitly."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"detail": "Invalid login credentials", "code": "invalid_credentials"}
            ]
        },
    )

    detail: str = Field(description="Human-readable message safe for clients.")
    code: Optional[str] = Field(
        default=None,
        description="Stable machine code when available (e.g. Supabase error code).",
    )
