"""Upload prediction images to Supabase Storage (caller JWT → RLS ``auth.uid()``)."""

from __future__ import annotations

import logging
import uuid

from storage3.exceptions import StorageApiError
from storage3.utils import StorageException

from backend.core.config import settings
from backend.core.exceptions import PredictionServiceError
from backend.core.prediction_image_limits import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    IMAGE_EXT_BY_CONTENT_TYPE,
    prediction_image_max_bytes,
)
from backend.integrations.supabase_client import create_supabase_user_client

logger = logging.getLogger(__name__)

_SIGNED_URL_TTL_S: int = 3600


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
        max_b = prediction_image_max_bytes()
        if len(file_bytes) > max_b:
            mb = max_b / (1024 * 1024)
            raise PredictionServiceError(
                f"La imagen supera el tamaño máximo permitido ({mb:.0f} MB).",
                413,
                code="image_too_large",
            )
        ct = content_type.split(";")[0].strip().lower()
        if ct not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise PredictionServiceError(
                "El tipo de imagen no está permitido. Use JPEG, PNG o WebP.",
                415,
                code="unsupported_media_type",
            )
        ext = IMAGE_EXT_BY_CONTENT_TYPE[ct]
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
            logger.warning(
                "prediction_storage_error op=upload code=%s status=%s message=%s",
                code,
                status,
                msg or "upload_failed",
            )
            raise PredictionServiceError(
                "No se pudo almacenar la imagen de predicción.",
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
            logger.warning(
                "prediction_storage_error op=signed_url code=%s message=%s",
                getattr(e, "code", "signed_url_failed"),
                getattr(e, "message", str(e)) or "signed_url_failed",
            )
            raise PredictionServiceError(
                "No se pudo generar la URL firmada de la imagen.",
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
