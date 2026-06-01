"""PredictionImagesStorage: errores y compensación."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from storage3.exceptions import StorageApiError

from backend.core.exceptions import PredictionServiceError
from backend.repositories.prediction_images_storage import PredictionImagesStorage
from backend.schemas.auth import UserOut
from backend.schemas.prediction import PredictionCreateBody
from backend.services.prediction_service import PredictionService


def test_upload_storage_api_error_maps_to_prediction_error() -> None:
    storage = PredictionImagesStorage()
    mock_client = MagicMock()
    mock_client.storage.from_.return_value.upload.side_effect = StorageApiError(
        "denied",
        "storage_forbidden",
        403,
    )
    with patch(
        "backend.repositories.prediction_images_storage.create_supabase_user_client",
        return_value=mock_client,
    ):
        with pytest.raises(PredictionServiceError) as exc:
            storage.upload_user_image(
                "token",
                user_id="11111111-1111-1111-1111-111111111111",
                file_bytes=b"x" * 64,
                content_type="image/png",
            )
    assert exc.value.status_code == 403


def test_predict_deletes_storage_object_when_insert_fails() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="u@example.com",
        created_at=None,
    )
    deleted: list[str] = []

    class _Img:
        def upload_user_image(self, *_a, **_k) -> str:
            return f"{user.id}/deadbeef.png"

        def delete_user_image(self, _token: str, path: str) -> None:
            deleted.append(path)

    class _Repo:
        def insert_for_user(self, *_a, **_k):
            raise PredictionServiceError("db fail", 502, code="postgrest_error")

    class _Pred:
        def predict_from_rgb(self, _rgb: np.ndarray) -> float:
            return 0.2

    def _fake_prepare(_ct: str | None, _fb: bytes) -> tuple[str, bytes, np.ndarray]:
        rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
        return ("image/png", _fb, rgb)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "backend.services.prediction_service.prepare_prediction_image",
        _fake_prepare,
    )
    svc = PredictionService(
        repo=_Repo(),
        images=_Img(),
        image_predictor=_Pred(),
        nail_checker=lambda _rgb: None,
    )
    try:
        with pytest.raises(PredictionServiceError):
            svc.run_predict(
                user,
                "tok",
                PredictionCreateBody(),
                b"png-bytes",
                "image/png",
            )
    finally:
        monkeypatch.undo()
    assert deleted == [f"{user.id}/deadbeef.png"]
