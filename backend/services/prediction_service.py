import logging
from collections.abc import Callable

import numpy as np
from fastapi import UploadFile
from pydantic import ValidationError

from backend.core import patient_age
from backend.core.config import settings
from backend.core.exceptions import ClientHttpError, PredictionServiceError
from backend.core.prediction_image_limits import prediction_image_max_bytes
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
from backend.repositories.predictions_repository import PredictionsRepository
from backend.schemas.auth import UserOut
from backend.schemas.prediction import (
    PredictionCreateBody,
    PredictionHistoryItem,
    PredictionImageSignedUrlOut,
    PredictionResponse,
)

_INFERENCE_MODE = "backend"

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
                + "con un .keras válido (p. ej. ml/artifacts/models/baseline_mobilenetv2.keras).",
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
        return self.run_predict(user, access_token, body, raw, content_type)

    def run_predict(
        self,
        user: UserOut,
        access_token: str,
        body: PredictionCreateBody,
        file_bytes: bytes,
        content_type: str | None,
    ) -> PredictionResponse:
        normalized_ct, processed_bytes, rgb = prepare_prediction_image(
            content_type,
            file_bytes,
        )
        self._nail_checker(rgb)
        predictor = self._effective_image_predictor()
        # prepare_prediction_image ya devuelve el RGB listo; el predictor usa ese array directamente
        # (sin volver a decodificar bytes), coherente con el pipeline de evaluación.
        raw_probability = average_raw_probability_with_optional_tta(
            predictor,
            rgb,
            tta_enabled=bool(settings.inference_tta_enabled),
        )
        path = self._images.upload_user_image(
            access_token,
            user_id=user.id,
            file_bytes=processed_bytes,
            content_type=normalized_ct,
        )
        try:
            return self._run_predict_core(
                user,
                access_token,
                body,
                image_storage_path=path,
                raw_probability=raw_probability,
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
        )
        display = patient_age.age_display_from_months(row.get("age_months"))
        human_summary = anemia_risk_label(risk)
        try:
            response = PredictionResponse.model_validate(
                {
                    **row,
                    "age_display": display,
                    "inference_mode": _INFERENCE_MODE,
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

    def list_predictions(self, access_token: str) -> list[PredictionHistoryItem]:
        rows = self._repo.list_for_user(access_token)
        out: list[PredictionHistoryItem] = []
        for r in rows:
            display = patient_age.age_display_from_months(r.get("age_months"))
            row_public = {k: v for k, v in r.items() if k != "image_storage_path"}
            try:
                out.append(
                    PredictionHistoryItem.model_validate(
                        {
                            **row_public,
                            "age_display": display,
                            "inference_mode": _INFERENCE_MODE,
                        },
                    ),
                )
            except ValueError as e:
                raise PredictionServiceError(
                    "Unexpected prediction row shape",
                    502,
                    code="invalid_list_shape",
                ) from e
        return out

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
        prefix = f"{user.id}/"
        if not path.startswith(prefix):
            raise PredictionServiceError(
                "La ruta de imagen no corresponde al usuario autenticado.",
                403,
                code="image_path_forbidden",
            )
        url = self._images.create_signed_url(access_token, path)
        return PredictionImageSignedUrlOut(signed_url=url)
