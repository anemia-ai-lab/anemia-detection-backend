from backend.core import patient_age
from backend.core.config import settings
from backend.core.risk_mapping import risk_from_probability
from backend.repositories.prediction_images_storage import PredictionImagesStorage
from backend.repositories.predictions_repository import PredictionsRepository
from backend.schemas.auth import UserOut
from backend.schemas.prediction import (
    PredictionCreateBody,
    PredictionHistoryItem,
    PredictionImageSignedUrlOut,
    PredictionResponse,
)
from backend.services.exceptions import PredictionServiceError


class PredictionService:
    """Mock inference + persistence (no ML here yet)."""

    def __init__(
        self,
        repo: PredictionsRepository | None = None,
        images: PredictionImagesStorage | None = None,
    ) -> None:
        self._repo = repo or PredictionsRepository()
        self._images = images or PredictionImagesStorage()

    def run_predict(
        self,
        user: UserOut,
        access_token: str,
        body: PredictionCreateBody,
    ) -> PredictionResponse:
        return self._run_predict_core(
            user,
            access_token,
            body,
            image_storage_path=None,
        )

    def run_predict_with_image(
        self,
        user: UserOut,
        access_token: str,
        body: PredictionCreateBody,
        file_bytes: bytes,
        content_type: str | None,
    ) -> PredictionResponse:
        mime = content_type or "application/octet-stream"
        path = self._images.upload_user_image(
            access_token,
            user_id=user.id,
            file_bytes=file_bytes,
            content_type=mime,
        )
        return self._run_predict_core(
            user,
            access_token,
            body,
            image_storage_path=path,
        )

    def _run_predict_core(
        self,
        user: UserOut,
        access_token: str,
        body: PredictionCreateBody,
        *,
        image_storage_path: str | None,
    ) -> PredictionResponse:
        score = 0.42
        risk = risk_from_probability(score, settings.risk_threshold)
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
            score=score,
            model_version=model_version,
            age_months=age_months,
            birth_date=birth_iso,
            notes=body.notes,
            image_storage_path=image_storage_path,
        )
        display = patient_age.age_display_from_months(row.get("age_months"))
        try:
            return PredictionResponse.model_validate({**row, "age_display": display})
        except ValueError as e:
            raise PredictionServiceError(
                "Unexpected prediction row shape",
                502,
                code="invalid_insert_shape",
            ) from e

    def list_predictions(self, access_token: str) -> list[PredictionHistoryItem]:
        rows = self._repo.list_for_user(access_token)
        out: list[PredictionHistoryItem] = []
        for r in rows:
            display = patient_age.age_display_from_months(r.get("age_months"))
            try:
                out.append(
                    PredictionHistoryItem.model_validate({**r, "age_display": display}),
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
