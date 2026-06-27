import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi import UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from backend.core import patient_age
from backend.core.config import settings
from backend.core.exceptions import ClientHttpError, PredictionServiceError
from backend.core.prediction_cursor import encode_prediction_cursor
from backend.core.prediction_image_limits import prediction_image_max_bytes
from backend.core.prometheus_metrics import observe_predict_phase
from backend.core.risk_mapping import anemia_risk_label, risk_from_probability
from backend.core.upload_io import UploadExceedsMaxBytesError, read_upload_file_with_byte_limit
from backend.inference.image_predictor import ImagePredictor
from backend.inference.nail_presence import require_fingernail_presence
from backend.inference.prediction_image_input import prepare_prediction_image
from backend.inference.probability_calibration import (
    apply_temperature_calibration,
    binary_prediction_from_threshold,
)
from backend.inference.runtime import get_builtin_image_predictor
from backend.inference.tta import average_raw_probability_with_optional_tta
from backend.repositories.prediction_images_storage import PredictionImagesStorage
from backend.repositories.predictions_repository import PredictionsRepository, utc_now_iso
from backend.schemas.auth import UserOut
from backend.schemas.prediction import (
    SYNC_METADATA_BATCH_MAX,
    PredictionCreateBody,
    PredictionDetailOut,
    PredictionImageSignedUrlOut,
    PredictionImageUploadOut,
    PredictionListItem,
    PredictionListResponse,
    PredictionResponse,
    PredictionSyncMetadataItem,
    PredictionSyncMetadataRequest,
    PredictionSyncMetadataResponse,
    PredictionSyncMetadataResult,
)

_INFERENCE_MODE_BACKEND = "backend"
_CLIENT_CREATED_AT_FUTURE_TOLERANCE = timedelta(minutes=5)
_LIST_DEFAULT_LIMIT = 20
_LIST_MAX_LIMIT = 100

logger = logging.getLogger(__name__)


def _format_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Validation error"
    err0 = errors[0]
    loc_parts = [str(x) for x in err0.get("loc", ()) if x not in ("body",)]
    loc = ".".join(loc_parts) if loc_parts else ""
    msg = str(err0.get("msg", "Validation error"))
    return f"{msg}" + (f" ({loc})" if loc else "")


def _parse_effective_created_at(row: dict) -> datetime:
    raw = row.get("effective_created_at") or row.get("created_at")
    if raw is None:
        raise PredictionServiceError(
            "Unexpected prediction row shape",
            502,
            code="invalid_row_shape",
        )
    if isinstance(raw, datetime):
        return raw
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def _has_image(row: dict) -> bool:
    return bool(row.get("image_storage_path"))


def _row_to_list_item(row: dict) -> PredictionListItem:
    display = patient_age.age_display_from_months(row.get("age_months"))
    effective = _parse_effective_created_at(row)
    public = {
        k: v
        for k, v in row.items()
        if k
        not in (
            "image_storage_path",
            "user_id",
            "raw_probability",
            "calibrated_probability",
            "threshold_used",
            "prediction",
            "client_id",
            "client_created_at",
            "image_sha256",
            "synced_at",
            "preprocessing",
            "created_at",
        )
    }
    return PredictionListItem.model_validate(
        {
            **public,
            "age_display": display,
            "effective_created_at": effective,
            "inference_mode": row.get("inference_mode") or _INFERENCE_MODE_BACKEND,
            "has_image": _has_image(row),
        },
    )


class PredictionService:
    """Inferencia Keras con imagen obligatoria + validación previa + persistencia."""

    def __init__(
        self,
        repo: PredictionsRepository | None = None,
        images: PredictionImagesStorage | None = None,
        image_predictor: ImagePredictor | None = None,
        nail_checker: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self._repo = repo or PredictionsRepository()
        self._images = images or PredictionImagesStorage()
        self._image_predictor = image_predictor
        self._nail_checker = nail_checker or require_fingernail_presence

    def _effective_image_predictor(self) -> ImagePredictor:
        if self._image_predictor is not None:
            return self._image_predictor
        builtin = get_builtin_image_predictor()
        if builtin is None:
            raise PredictionServiceError(
                "No hay modelo de inferencia cargado. Configure INFERENCE_MODEL_PATH "
                + "con un .keras válido (p. ej. ml/artifacts/models/baseline_mobilenetv2_ghana_augmented_seed42.keras).",
                503,
                code="inference_model_unavailable",
            )
        return builtin

    @staticmethod
    def parse_prediction_body(
        *,
        birth_date: str | None,
        notes: str | None,
    ) -> PredictionCreateBody:
        fields: dict[str, object] = {"notes": notes}
        if birth_date not in (None, ""):
            fields["birth_date"] = birth_date
        try:
            return PredictionCreateBody.model_validate(fields)
        except ValidationError as exc:
            raise ClientHttpError(
                _format_validation_error(exc),
                422,
                code="validation_error",
            ) from exc

    @staticmethod
    def require_image_upload(image: UploadFile | None) -> UploadFile:
        if image is None or not (image.filename and str(image.filename).strip()):
            raise PredictionServiceError(
                "image is required for prediction",
                400,
                code="image_required",
            )
        return image

    async def read_prediction_image_bytes(self, upload: UploadFile) -> tuple[bytes, str | None]:
        max_b = prediction_image_max_bytes()
        try:
            raw = await read_upload_file_with_byte_limit(upload, max_b)
        except UploadExceedsMaxBytesError:
            mb = max_b / (1024 * 1024)
            raise PredictionServiceError(
                f"La imagen supera el tamaño máximo permitido ({mb:.0f} MB).",
                413,
                code="image_too_large",
            ) from None
        if not raw:
            raise PredictionServiceError(
                "image is required for prediction",
                400,
                code="image_required",
            )
        return raw, upload.content_type

    async def run_predict_from_upload(
        self,
        user: UserOut,
        access_token: str,
        *,
        image: UploadFile | None,
        birth_date: str | None,
        notes: str | None,
    ) -> PredictionResponse:
        upload = self.require_image_upload(image)
        raw, content_type = await self.read_prediction_image_bytes(upload)
        body = self.parse_prediction_body(birth_date=birth_date, notes=notes)
        return await run_in_threadpool(
            self.run_predict,
            user,
            access_token,
            body,
            raw,
            content_type,
        )

    def run_predict(
        self,
        user: UserOut,
        access_token: str,
        body: PredictionCreateBody,
        file_bytes: bytes,
        content_type: str | None,
    ) -> PredictionResponse:
        with observe_predict_phase("preprocess"):
            normalized_ct, processed_bytes, rgb = prepare_prediction_image(
                content_type,
                file_bytes,
            )
            self._nail_checker(rgb)
        with observe_predict_phase("inference"):
            predictor = self._effective_image_predictor()
            raw_probability = average_raw_probability_with_optional_tta(
                predictor,
                rgb,
                tta_enabled=bool(settings.inference_tta_enabled),
            )
        with observe_predict_phase("storage_upload"):
            path = self._images.upload_user_image(
                access_token,
                user_id=user.id,
                file_bytes=processed_bytes,
                content_type=normalized_ct,
            )
        preprocessing = {"tta_enabled": bool(settings.inference_tta_enabled)}
        try:
            return self._run_predict_core(
                user,
                access_token,
                body,
                image_storage_path=path,
                raw_probability=raw_probability,
                preprocessing=preprocessing,
            )
        except Exception:
            self._images.delete_user_image(access_token, path)
            raise

    def _run_predict_core(
        self,
        user: UserOut,
        access_token: str,
        body: PredictionCreateBody,
        *,
        image_storage_path: str,
        raw_probability: float,
        preprocessing: dict | None = None,
    ) -> PredictionResponse:
        temperature = float(settings.inference_calibration_temperature)
        low_upper = float(settings.inference_risk_tier_low_upper)
        high_lower = float(settings.inference_risk_tier_high_lower)
        threshold_used = high_lower
        calibrated_probability = apply_temperature_calibration(raw_probability, temperature)
        risk = risk_from_probability(
            calibrated_probability,
            low_upper=low_upper,
            high_lower=high_lower,
        )
        prediction = binary_prediction_from_threshold(calibrated_probability, high_lower)
        model_version = settings.model_version
        ref = patient_age.utc_today()
        birth = body.birth_date
        birth_iso = birth.isoformat() if birth is not None else None
        age_months: int | None = None
        if birth is not None:
            age_months = patient_age.completed_age_months(birth, ref)
        with observe_predict_phase("db_insert"):
            row = self._repo.insert_for_user(
                access_token,
                user_id=user.id,
                risk=risk,
                score=calibrated_probability,
                model_version=model_version,
                age_months=age_months,
                birth_date=birth_iso,
                notes=body.notes,
                image_storage_path=image_storage_path,
                inference_mode=_INFERENCE_MODE_BACKEND,
                raw_probability=raw_probability,
                calibrated_probability=calibrated_probability,
                threshold_used=threshold_used,
                prediction=prediction,
                preprocessing=preprocessing,
            )
        return self._row_to_prediction_response(
            row,
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            threshold_used=threshold_used,
            low_upper=low_upper,
            high_lower=high_lower,
            prediction=prediction,
        )

    def sync_metadata_batch(
        self,
        user: UserOut,
        access_token: str,
        body: PredictionSyncMetadataRequest,
    ) -> PredictionSyncMetadataResponse:
        if len(body.items) > SYNC_METADATA_BATCH_MAX:
            raise ClientHttpError(
                f"Máximo {SYNC_METADATA_BATCH_MAX} items por solicitud.",
                422,
                code="batch_too_large",
            )
        results: list[PredictionSyncMetadataResult] = []
        for item in body.items:
            results.append(self._sync_one_metadata(user, access_token, item))
        return PredictionSyncMetadataResponse(results=results)

    def _sync_one_metadata(
        self,
        user: UserOut,
        access_token: str,
        item: PredictionSyncMetadataItem,
    ) -> PredictionSyncMetadataResult:
        if item.inference_mode != "tflite_offline":
            raise ClientHttpError(
                "inference_mode debe ser tflite_offline en sync de metadatos.",
                422,
                code="invalid_inference_mode",
            )
        self._validate_client_created_at(item.client_created_at)

        existing = self._repo.find_by_client_id(
            access_token,
            user_id=user.id,
            client_id=item.client_id,
        )
        if existing is not None:
            return PredictionSyncMetadataResult(
                client_id=item.client_id,
                id=str(existing["id"]),
                image_pending=not _has_image(existing),
                created=False,
            )

        ref = patient_age.utc_today()
        birth = item.birth_date
        birth_iso = birth.isoformat() if birth is not None else None
        age_months: int | None = None
        if birth is not None:
            age_months = patient_age.completed_age_months(birth, ref)

        row = self._repo.insert_for_user(
            access_token,
            user_id=user.id,
            risk=item.risk,
            score=item.score,
            model_version=item.model_version,
            age_months=age_months,
            birth_date=birth_iso,
            notes=item.notes,
            image_storage_path=None,
            inference_mode=item.inference_mode,
            raw_probability=item.raw_probability,
            calibrated_probability=item.calibrated_probability,
            threshold_used=item.threshold_used,
            prediction=item.prediction,
            client_id=item.client_id,
            client_created_at=item.client_created_at.astimezone(UTC).isoformat(),
            image_sha256=item.image_sha256,
            synced_at=utc_now_iso(),
            preprocessing=item.preprocessing,
        )
        return PredictionSyncMetadataResult(
            client_id=item.client_id,
            id=str(row["id"]),
            image_pending=True,
            created=True,
        )

    @staticmethod
    def _validate_client_created_at(client_created_at: datetime) -> None:
        now = datetime.now(UTC)
        ts = client_created_at.astimezone(UTC)
        if ts > now + _CLIENT_CREATED_AT_FUTURE_TOLERANCE:
            raise ClientHttpError(
                "client_created_at no puede estar en el futuro.",
                422,
                code="client_created_at_future",
            )

    async def upload_prediction_image(
        self,
        user: UserOut,
        access_token: str,
        prediction_id: str,
        *,
        image: UploadFile | None,
        image_sha256: str | None = None,
    ) -> PredictionImageUploadOut:
        upload = self.require_image_upload(image)
        raw, content_type = await self.read_prediction_image_bytes(upload)
        return await run_in_threadpool(
            self._finish_upload_prediction_image,
            user,
            access_token,
            prediction_id,
            raw,
            content_type,
            image_sha256,
        )

    def _finish_upload_prediction_image(
        self,
        user: UserOut,
        access_token: str,
        prediction_id: str,
        raw: bytes,
        content_type: str | None,
        image_sha256: str | None,
    ) -> PredictionImageUploadOut:
        row = self._repo.fetch_by_id(access_token, prediction_id)
        if row is None:
            raise PredictionServiceError(
                "Predicción no encontrada.",
                404,
                code="prediction_not_found",
            )

        existing_path = row.get("image_storage_path")
        if existing_path:
            path = str(existing_path)
            self._assert_image_path_owned(user, path)
            url = self._images.create_signed_url(access_token, path)
            return PredictionImageUploadOut(
                id=prediction_id,
                image_storage_path=path,
                image_signed_url=url,
                image_sha256=row.get("image_sha256"),
            )

        if image_sha256:
            digest = hashlib.sha256(raw).hexdigest()
            if digest != image_sha256.lower():
                raise PredictionServiceError(
                    "image_sha256 no coincide con la imagen enviada.",
                    409,
                    code="image_sha256_mismatch",
                )
        elif row.get("image_sha256"):
            digest = hashlib.sha256(raw).hexdigest()
            if digest != str(row["image_sha256"]).lower():
                raise PredictionServiceError(
                    "image_sha256 no coincide con la imagen enviada.",
                    409,
                    code="image_sha256_mismatch",
                )
            image_sha256 = digest
        else:
            image_sha256 = hashlib.sha256(raw).hexdigest()

        normalized_ct, processed_bytes, rgb = prepare_prediction_image(content_type, raw)
        self._nail_checker(rgb)
        path = self._images.upload_user_image(
            access_token,
            user_id=user.id,
            file_bytes=processed_bytes,
            content_type=normalized_ct,
        )
        try:
            updated = self._repo.update_image_for_prediction(
                access_token,
                prediction_id,
                image_storage_path=path,
                image_sha256=image_sha256,
            )
        except Exception:
            self._images.delete_user_image(access_token, path)
            raise

        url = self._images.create_signed_url(access_token, path)
        return PredictionImageUploadOut(
            id=str(updated["id"]),
            image_storage_path=path,
            image_signed_url=url,
            image_sha256=image_sha256,
        )

    def list_predictions_paginated(
        self,
        access_token: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> PredictionListResponse:
        page_limit = _LIST_DEFAULT_LIMIT if limit is None else limit
        if page_limit < 1 or page_limit > _LIST_MAX_LIMIT:
            raise ClientHttpError(
                f"limit debe estar entre 1 y {_LIST_MAX_LIMIT}.",
                400,
                code="invalid_limit",
            )

        rows = self._repo.list_for_user_paginated(
            access_token,
            limit=page_limit,
            cursor=cursor,
        )
        has_more = len(rows) > page_limit
        page_rows = rows[:page_limit]
        items = [_row_to_list_item(r) for r in page_rows]

        next_cursor: str | None = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_prediction_cursor(
                _parse_effective_created_at(last),
                str(last["id"]),
            )

        return PredictionListResponse(items=items, next_cursor=next_cursor)

    def get_prediction_detail(
        self,
        user: UserOut,
        access_token: str,
        prediction_id: str,
    ) -> PredictionDetailOut:
        row = self._repo.fetch_by_id(access_token, prediction_id)
        if row is None:
            raise PredictionServiceError(
                "Predicción no encontrada.",
                404,
                code="prediction_not_found",
            )
        return self._row_to_detail(user, access_token, row)

    def delete_prediction(
        self,
        user: UserOut,
        access_token: str,
        prediction_id: str,
    ) -> None:
        row = self._repo.fetch_by_id(access_token, prediction_id)
        if row is None:
            raise PredictionServiceError(
                "Predicción no encontrada.",
                404,
                code="prediction_not_found",
            )
        image_path = row.get("image_storage_path")
        deleted = self._repo.delete_by_id(access_token, prediction_id)
        if not deleted:
            raise PredictionServiceError(
                "Predicción no encontrada.",
                404,
                code="prediction_not_found",
            )
        if image_path:
            self._images.delete_user_image(access_token, str(image_path))

    def signed_image_url_for_prediction(
        self,
        user: UserOut,
        access_token: str,
        prediction_id: str,
    ) -> PredictionImageSignedUrlOut:
        path = self._repo.fetch_image_storage_path(access_token, prediction_id)
        if not path:
            raise PredictionServiceError(
                "Predicción sin imagen o no encontrada.",
                404,
                code="prediction_image_not_found",
            )
        self._assert_image_path_owned(user, path)
        url = self._images.create_signed_url(access_token, path)
        return PredictionImageSignedUrlOut(signed_url=url)

    @staticmethod
    def _assert_image_path_owned(user: UserOut, path: str) -> None:
        prefix = f"{user.id}/"
        if not path.startswith(prefix):
            raise PredictionServiceError(
                "La ruta de imagen no corresponde al usuario autenticado.",
                403,
                code="image_path_forbidden",
            )

    def _row_to_detail(
        self,
        user: UserOut,
        access_token: str,
        row: dict,
    ) -> PredictionDetailOut:
        risk = row.get("risk")
        display = patient_age.age_display_from_months(row.get("age_months"))
        effective = _parse_effective_created_at(row)
        created_raw = row.get("created_at")
        created_at = (
            created_raw
            if isinstance(created_raw, datetime)
            else datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
        )
        has_image = _has_image(row)
        signed_url: str | None = None
        if has_image:
            path = str(row["image_storage_path"])
            self._assert_image_path_owned(user, path)
            signed_url = self._images.create_signed_url(access_token, path)

        client_created_raw = row.get("client_created_at")
        client_created_at: datetime | None = None
        if client_created_raw:
            client_created_at = (
                client_created_raw
                if isinstance(client_created_raw, datetime)
                else datetime.fromisoformat(str(client_created_raw).replace("Z", "+00:00"))
            )

        synced_raw = row.get("synced_at")
        synced_at: datetime | None = None
        if synced_raw:
            synced_at = (
                synced_raw
                if isinstance(synced_raw, datetime)
                else datetime.fromisoformat(str(synced_raw).replace("Z", "+00:00"))
            )

        return PredictionDetailOut(
            id=str(row["id"]),
            risk=risk,
            score=float(row["score"]),
            raw_probability=row.get("raw_probability"),
            calibrated_probability=row.get("calibrated_probability"),
            threshold_used=row.get("threshold_used"),
            prediction=row.get("prediction"),
            risk_label=anemia_risk_label(risk),
            model_version=str(row["model_version"]),
            birth_date=row.get("birth_date"),
            age_months=row.get("age_months"),
            age_display=display,
            notes=row.get("notes"),
            client_id=str(row["client_id"]) if row.get("client_id") else None,
            inference_mode=row.get("inference_mode") or _INFERENCE_MODE_BACKEND,
            client_created_at=client_created_at,
            effective_created_at=effective,
            created_at=created_at,
            synced_at=synced_at,
            image_sha256=row.get("image_sha256"),
            preprocessing=row.get("preprocessing"),
            has_image=has_image,
            image_signed_url=signed_url,
        )

    def _row_to_prediction_response(
        self,
        row: dict,
        *,
        raw_probability: float,
        calibrated_probability: float,
        threshold_used: float,
        low_upper: float,
        high_lower: float,
        prediction: int,
    ) -> PredictionResponse:
        display = patient_age.age_display_from_months(row.get("age_months"))
        human_summary = anemia_risk_label(row.get("risk"))
        inference_mode = row.get("inference_mode") or _INFERENCE_MODE_BACKEND
        try:
            response = PredictionResponse.model_validate(
                {
                    **row,
                    "age_display": display,
                    "inference_mode": inference_mode,
                    "raw_probability": raw_probability,
                    "calibrated_probability": calibrated_probability,
                    "threshold_used": threshold_used,
                    "risk_tier_low_upper": low_upper,
                    "risk_tier_high_lower": high_lower,
                    "prediction": prediction,
                    "risk_label": human_summary,
                    "message": human_summary,
                },
            )
        except ValueError as e:
            raise PredictionServiceError(
                "Unexpected prediction row shape",
                502,
                code="invalid_insert_shape",
            ) from e
        logger.info(
            "prediction_completed model_version=%s inference_mode=%s risk=%s prediction=%s",
            response.model_version,
            response.inference_mode,
            response.risk,
            response.prediction,
        )
        return response
