"""Cursor opaco para paginación de ``GET /predictions``."""

from __future__ import annotations

import base64
from datetime import datetime

from backend.core.exceptions import ClientHttpError

_CURSOR_SEP = "|"


def encode_prediction_cursor(effective_created_at: datetime, row_id: str) -> str:
    ts = effective_created_at.isoformat()
    raw = f"{ts}{_CURSOR_SEP}{row_id}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def decode_prediction_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        ts_part, row_id = raw.rsplit(_CURSOR_SEP, 1)
        if not row_id or not ts_part:
            raise ValueError("empty parts")
        return datetime.fromisoformat(ts_part.replace("Z", "+00:00")), row_id
    except (ValueError, UnicodeDecodeError) as exc:
        raise ClientHttpError(
            "Cursor de paginación inválido.",
            400,
            code="invalid_cursor",
        ) from exc
