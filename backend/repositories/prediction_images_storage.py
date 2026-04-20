"""Upload prediction images to Supabase Storage (caller JWT → RLS ``auth.uid()``)."""

from __future__ import annotations

import uuid
from typing import Final

from storage3.exceptions import StorageApiError
from storage3.utils import StorageException

from backend.core.config import settings
from backend.integrations.supabase_client import create_supabase_user_client
from backend.services.exceptions import PredictionServiceError

_ALLOWED_CT: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/png", "image/webp"},
)
_EXT: Final[dict[str, str]] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
_MAX_BYTES: Final[int] = 5 * 1024 * 1024
_SIGNED_URL_TTL_S: Final[int] = 3600


class PredictionImagesStorage:
    def upload_user_image(
        self,
        access_token: str,
        *,
        user_id: str,
        file_bytes: bytes,
        content_type: str,
    ) -> str:
        """Upload to ``{user_id}/{uuid}.ext``; returns storage object path only."""
        if len(file_bytes) > _MAX_BYTES:
            raise PredictionServiceError(
                "La imagen supera el tamaño máximo permitido (5 MB).",
                400,
                code="image_too_large",
            )
        ct = content_type.split(";")[0].strip().lower()
        if ct not in _ALLOWED_CT:
            raise PredictionServiceError(
                "Tipo de imagen no permitido (use JPEG, PNG o WebP).",
                400,
                code="image_invalid_type",
            )
        ext = _EXT[ct]
        object_path = f"{user_id}/{uuid.uuid4().hex}.{ext}"
        bucket = settings.predictions_storage_bucket.strip() or "prediction-images"
        client = create_supabase_user_client(access_token)
        try:
            client.storage.from_(bucket).upload(
                object_path,
                file_bytes,
                file_options={"content-type": ct, "upsert": "false"},
            )
        except (StorageApiError, StorageException) as e:
            msg = getattr(e, "message", str(e))
            code = getattr(e, "code", "storage_error")
            status = getattr(e, "status", 502)
            sc = int(status) if str(status).isdigit() else 502
            raise PredictionServiceError(
                msg or "No se pudo subir la imagen",
                sc if 400 <= sc < 600 else 502,
                code=str(code),
            ) from e
        return object_path

    def create_signed_url(self, access_token: str, object_path: str) -> str:
        """URL firmada temporal para un objeto ya existente (mismo JWT que posee el prefijo)."""
        bucket = settings.predictions_storage_bucket.strip() or "prediction-images"
        client = create_supabase_user_client(access_token)
        try:
            signed = client.storage.from_(bucket).create_signed_url(
                object_path,
                _SIGNED_URL_TTL_S,
            )
        except (StorageApiError, StorageException) as e:
            raise PredictionServiceError(
                getattr(e, "message", str(e)) or "No se pudo firmar la URL",
                502,
                code="signed_url_failed",
            ) from e
        url = signed.get("signedURL") or signed.get("signedUrl") or ""
        if not url:
            raise PredictionServiceError(
                "Respuesta de firma vacía",
                502,
                code="signed_url_empty",
            )
        return str(url)
