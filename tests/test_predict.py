from datetime import UTC, date, datetime
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from postgrest import APIError

import backend.core.patient_age as patient_age_module
import backend.inference.prediction_image_input as prediction_image_input_module
from backend.api import deps as api_deps
from backend.core import config as config_module
from backend.inference.image_predictor import StaticImagePredictor
from backend.main import app
from backend.repositories import predictions_repository as predictions_repo_module
from backend.schemas.auth import UserOut
from backend.services.prediction_service import PredictionService

client = TestClient(app)

_SKIN_PNG_CACHE: bytes | None = None


@pytest.fixture(autouse=True)
def _replace_tensorflow_image_ops_with_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    ``prepare_prediction_image`` importa TensorFlow; en el venv de tests no hace falta TF
    si sustituimos decode/resize/encode por Pillow (misma semántica para estas pruebas).
    """
    from PIL import Image

    from backend.services.exceptions import PredictionServiceError

    def decode_rgb_uint8(raw: bytes) -> np.ndarray:
        try:
            im = Image.open(BytesIO(raw)).convert("RGB")
            return np.asarray(im, dtype=np.uint8)
        except Exception as exc:
            raise PredictionServiceError(
                "El archivo no es una imagen válida o está dañado.",
                400,
                code="image_not_decodable",
            ) from exc

    def resize_rgb_max_edge(rgb: np.ndarray, max_edge: int) -> np.ndarray:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        m = max(h, w)
        if m <= max_edge:
            return rgb
        scale = max_edge / m
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        im = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)

    def rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(prediction_image_input_module, "decode_rgb_uint8", decode_rgb_uint8)
    monkeypatch.setattr(prediction_image_input_module, "resize_rgb_max_edge", resize_rgb_max_edge)
    monkeypatch.setattr(prediction_image_input_module, "rgb_to_png_bytes", rgb_to_png_bytes)


def _patch_identity_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    """En tests, ``T=1`` y umbral 0.5 equivalen a usar el score raw sin cambiar el umbral de riesgo."""
    monkeypatch.setattr(config_module.settings, "inference_calibration_temperature", 1.0)
    monkeypatch.setattr(
        config_module.settings,
        "inference_calibration_operational_threshold",
        0.5,
    )


def _skip_nail(_rgb: np.ndarray) -> None:
    return


def skin_patch_png() -> bytes:
    global _SKIN_PNG_CACHE
    if _SKIN_PNG_CACHE is None:
        from PIL import Image

        buf = BytesIO()
        Image.new("RGB", (32, 32), (220, 180, 140)).save(buf, format="PNG")
        _SKIN_PNG_CACHE = buf.getvalue()
    return _SKIN_PNG_CACHE


def black_png() -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (48, 48), (0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeImgStore:
    def upload_user_image(
        self,
        _access_token: str,
        *,
        user_id: str,
        file_bytes: bytes,
        content_type: str,
    ) -> str:
        _ = file_bytes, content_type
        return f"{user_id}/test.png"


class _ExplodingRepo:
    def insert_for_user(self, *_a, **_k) -> dict:
        raise AssertionError("persist should not be reached")


def test_predict_without_token() -> None:
    response = client.post("/predict")
    assert response.status_code == 401
    assert response.json()["code"] == "missing_token"


def test_predict_with_malformed_token() -> None:
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer not-a-jwt"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "malformed_token"


def test_predict_success_with_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_identity_calibration(monkeypatch)
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    class FakeRepo:
        def insert_for_user(
            self,
            _access_token: str,
            *,
            user_id: str,
            risk: str,
            score: float,
            model_version: str,
            age_months: int | None = None,
            birth_date: str | None = None,
            notes: str | None = None,
            image_storage_path: str | None = None,
        ) -> dict:
            assert user_id == user.id
            assert risk == "low"
            assert image_storage_path == f"{user.id}/test.png"
            return {
                "id": "22222222-2222-2222-2222-222222222222",
                "risk": risk,
                "score": score,
                "model_version": model_version,
                "age_months": age_months,
                "birth_date": birth_date,
                "notes": notes,
                "image_storage_path": image_storage_path,
                "created_at": "2026-05-01T10:00:00+00:00",
            }

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=FakeRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.42),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "22222222-2222-2222-2222-222222222222"
        assert data["risk"] == "low"
        assert data["score"] == pytest.approx(0.42)
        assert data["raw_probability"] == pytest.approx(0.42)
        assert data["calibrated_probability"] == pytest.approx(0.42)
        assert data["threshold_used"] == 0.5
        assert data["prediction"] == 0
        assert data["risk_label"] == "Low anemia risk prediction"
        assert data["message"] == data["risk_label"]
        assert data["model_version"] == "v1.0"
        assert data["created_at"].startswith("2026-05-01T10:00:00")
        assert data["birth_date"] is None
        assert data["age_months"] is None
        assert data["age_display"] is None
        assert data["inference_mode"] == "backend"
    finally:
        app.dependency_overrides.clear()


def test_predict_missing_image_returns_400() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            data={"notes": "x"},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"] == "image is required for prediction"
        assert body["code"] == "image_required"

        response_empty = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert response_empty.status_code == 400
        assert response_empty.json()["code"] == "image_required"
    finally:
        app.dependency_overrides.clear()


def test_predict_rejects_invalid_image_decode() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=_ExplodingRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.5),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("x.png", b"not-a-valid-image", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "image_not_decodable"
    finally:
        app.dependency_overrides.clear()


def test_predict_rejects_invalid_content_type() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=_ExplodingRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.5),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("x.bin", skin_patch_png(), "application/octet-stream")},
        )
        assert response.status_code == 415
        assert response.json()["code"] == "unsupported_media_type"
    finally:
        app.dependency_overrides.clear()


def test_predict_rejects_image_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module.settings, "prediction_image_max_bytes", 50)
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=_ExplodingRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.5),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
        )
        assert response.status_code == 413
        assert response.json()["code"] == "image_too_large"
    finally:
        app.dependency_overrides.clear()


def test_predict_rejects_when_no_fingernail_detected() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=_ExplodingRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.99),
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("dark.png", black_png(), "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "no_fingernail_detected"
    finally:
        app.dependency_overrides.clear()


def test_predict_rejects_image_too_small() -> None:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (4, 4), (220, 180, 140)).save(buf, format="PNG")
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=_ExplodingRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.99),
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("tiny.png", buf.getvalue(), "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "image_too_small"
    finally:
        app.dependency_overrides.clear()


def test_predict_rejects_decoded_image_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module.settings, "prediction_image_max_pixels", 100)
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=_ExplodingRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.99),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
        )
        assert response.status_code == 413
        assert response.json()["code"] == "image_resolution_too_large"
    finally:
        app.dependency_overrides.clear()


def test_predict_rejects_pre_decode_pixel_dimensions_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PNG ligero con WxH grande: rechazo tras cabecera (Pillow) sin llegar al predictor."""
    from PIL import Image

    monkeypatch.setattr(config_module.settings, "prediction_image_max_pixels", 10_000)
    buf = BytesIO()
    Image.new("RGB", (1, 20_000), (0, 0, 0)).save(buf, format="PNG")
    raw_png = buf.getvalue()
    assert len(raw_png) < 512 * 1024
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=_ExplodingRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.99),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("tall.png", raw_png, "image/png")},
        )
        assert response.status_code == 413
        assert response.json()["code"] == "image_resolution_too_large"
    finally:
        app.dependency_overrides.clear()


def test_predict_risk_high_when_score_meets_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock score 0.42 → high si el umbral calibrado es <= 0.42 (con T=1, score raw = calibrado)."""
    monkeypatch.setattr(config_module.settings, "inference_calibration_temperature", 1.0)
    monkeypatch.setattr(
        config_module.settings,
        "inference_calibration_operational_threshold",
        0.41,
    )
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    class FakeRepo:
        def insert_for_user(
            self,
            _access_token: str,
            *,
            user_id: str,
            risk: str,
            score: float,
            model_version: str,
            age_months: int | None = None,
            birth_date: str | None = None,
            notes: str | None = None,
            image_storage_path: str | None = None,
        ) -> dict:
            assert risk == "high"
            return {
                "id": "22222222-2222-2222-2222-222222222222",
                "risk": risk,
                "score": score,
                "model_version": model_version,
                "age_months": age_months,
                "birth_date": birth_date,
                "notes": notes,
                "image_storage_path": image_storage_path,
                "created_at": "2026-05-01T10:00:00+00:00",
            }

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=FakeRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.42),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["risk"] == "high"
        assert body["risk_label"] == "High anemia risk prediction"
        assert body["message"] == body["risk_label"]
    finally:
        app.dependency_overrides.clear()


def test_predict_with_birth_date_sets_age_display(monkeypatch: pytest.MonkeyPatch) -> None:
    """Service computes age_months y age_display desde birth_date (ref = UTC hoy)."""
    _patch_identity_calibration(monkeypatch)
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    monkeypatch.setattr(
        patient_age_module,
        "utc_today",
        lambda: date(2025, 4, 15),
    )

    class FakeRepo:
        def insert_for_user(
            self,
            _access_token: str,
            *,
            user_id: str,
            risk: str,
            score: float,
            model_version: str,
            age_months: int | None = None,
            birth_date: str | None = None,
            notes: str | None = None,
            image_storage_path: str | None = None,
        ) -> dict:
            assert birth_date == "2016-01-15"
            assert age_months == 111
            return {
                "id": "22222222-2222-2222-2222-222222222222",
                "risk": risk,
                "score": score,
                "model_version": model_version,
                "age_months": age_months,
                "birth_date": birth_date,
                "notes": notes,
                "image_storage_path": image_storage_path,
                "created_at": "2026-05-01T10:00:00+00:00",
            }

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=FakeRepo(),
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.42),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
            data={"birth_date": "2016-01-15"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["birth_date"] == "2016-01-15"
        assert data["age_months"] == 111
        assert data["age_display"] == "9 años 3 meses"
        assert data["inference_mode"] == "backend"
        assert data["message"] == data["risk_label"]
    finally:
        app.dependency_overrides.clear()


def test_predict_implausible_birth_date_422(monkeypatch: pytest.MonkeyPatch) -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    monkeypatch.setattr(
        patient_age_module,
        "utc_today",
        lambda: date(2026, 6, 1),
    )
    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
            data={"birth_date": "1800-01-01"},
        )
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"
    finally:
        app.dependency_overrides.clear()


def test_predict_future_birth_date_422() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
            data={"birth_date": "2099-01-01"},
        )
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"
    finally:
        app.dependency_overrides.clear()


def test_predict_postgrest_api_error_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_identity_calibration(monkeypatch)
    user = UserOut(id="11111111-1111-1111-1111-111111111111", email="e@e.co", created_at=None)

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    mock_client = MagicMock()
    mock_client.from_.return_value.insert.return_value.execute.side_effect = APIError(
        {"message": "RLS blocked", "code": "42501"},
    )

    monkeypatch.setattr(
        predictions_repo_module,
        "create_supabase_user_client",
        lambda _token: mock_client,
    )

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            images=_FakeImgStore(),
            image_predictor=StaticImagePredictor(0.42),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
        )
        assert response.status_code == 502
        assert response.json()["code"] == "42501"
        assert response.json()["detail"] == "Could not save prediction"
    finally:
        app.dependency_overrides.clear()


def test_predict_without_loaded_model_returns_503() -> None:
    user = UserOut(id="11111111-1111-1111-1111-111111111111", email="e@e.co", created_at=None)

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    def fake_prediction_service() -> PredictionService:
        return PredictionService(
            repo=_ExplodingRepo(),
            images=_FakeImgStore(),
            image_predictor=None,
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
        )
        assert response.status_code == 503
        assert response.json()["code"] == "inference_model_unavailable"
    finally:
        app.dependency_overrides.clear()


def test_predictions_get_without_token() -> None:
    response = client.get("/predictions")
    assert response.status_code == 401
    assert response.json()["code"] == "missing_token"


def test_predictions_get_malformed_token() -> None:
    response = client.get(
        "/predictions",
        headers={"Authorization": "Bearer bad"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "malformed_token"


def test_predictions_get_success_with_overrides() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )
    created = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    class FakeRepo:
        def list_for_user(self, _access_token: str) -> list:
            return [
                {
                    "id": "33333333-3333-3333-3333-333333333333",
                    "risk": "low",
                    "score": 0.1,
                    "model_version": "v1.0",
                    "age_months": 24,
                    "birth_date": "2024-05-01",
                    "notes": None,
                    "image_storage_path": None,
                    "created_at": created.isoformat(),
                },
            ]

    def fake_prediction_service() -> PredictionService:
        return PredictionService(repo=FakeRepo())

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_prediction_service
    try:
        response = client.get(
            "/predictions",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "33333333-3333-3333-3333-333333333333"
        assert data[0]["risk"] == "low"
        assert data[0]["score"] == 0.1
        assert data[0]["age_months"] == 24
        assert data[0]["age_display"] == "2 años"
        assert data[0]["inference_mode"] == "backend"
    finally:
        app.dependency_overrides.clear()


def test_predictions_get_postgrest_error(monkeypatch: pytest.MonkeyPatch) -> None:
    user = UserOut(id="11111111-1111-1111-1111-111111111111", email="e@e.co", created_at=None)

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    mock_client = MagicMock()
    chain = (
        mock_client.from_.return_value.select.return_value.order.return_value.order.return_value
    )
    chain.execute.side_effect = APIError({"message": "table missing", "code": "42P01"})

    monkeypatch.setattr(
        predictions_repo_module,
        "create_supabase_user_client",
        lambda _token: mock_client,
    )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    try:
        response = client.get(
            "/predictions",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
        )
        assert response.status_code == 502
        assert response.json()["code"] == "42P01"
    finally:
        app.dependency_overrides.clear()


def test_predict_with_image_multipart(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_identity_calibration(monkeypatch)
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    class FakeImgStore:
        def upload_user_image(
            self,
            access_token: str,
            *,
            user_id: str,
            file_bytes: bytes,
            content_type: str,
        ) -> str:
            assert user_id == user.id
            assert content_type.startswith("image/")
            return f"{user_id}/test.png"

    class FakeRepoImg:
        def insert_for_user(
            self,
            _access_token: str,
            *,
            user_id: str,
            risk: str,
            score: float,
            model_version: str,
            age_months: int | None = None,
            birth_date: str | None = None,
            notes: str | None = None,
            image_storage_path: str | None = None,
        ) -> dict:
            assert image_storage_path == f"{user.id}/test.png"
            return {
                "id": "22222222-2222-2222-2222-222222222222",
                "risk": risk,
                "score": score,
                "model_version": model_version,
                "age_months": age_months,
                "birth_date": birth_date,
                "notes": notes,
                "image_storage_path": image_storage_path,
                "created_at": "2026-05-01T10:00:00+00:00",
            }

    def fake_svc() -> PredictionService:
        return PredictionService(
            repo=FakeRepoImg(),
            images=FakeImgStore(),
            image_predictor=StaticImagePredictor(0.42),
            nail_checker=_skip_nail,
        )

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_svc
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["image_storage_path"] == f"{user.id}/test.png"
        assert data["raw_probability"] == pytest.approx(0.42)
        assert data["calibrated_probability"] == pytest.approx(0.42)
        assert data["threshold_used"] == 0.5
        assert data["prediction"] == 0
        assert data["message"] == data["risk_label"]
        assert "image_signed_url" not in data
    finally:
        app.dependency_overrides.clear()


def test_prediction_image_signed_url_endpoint() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )
    pid = "33333333-3333-3333-3333-333333333333"
    path = f"{user.id}/a.png"

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    class FakeRepoSigned:
        def fetch_image_storage_path(self, _token: str, prediction_id: str) -> str | None:
            assert prediction_id == pid
            return path

    class FakeImgSigned:
        def create_signed_url(self, _token: str, object_path: str) -> str:
            assert object_path == path
            return "https://signed/once"

    def fake_svc() -> PredictionService:
        return PredictionService(repo=FakeRepoSigned(), images=FakeImgSigned())

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_svc
    try:
        response = client.get(
            f"/predictions/{pid}/image-signed-url",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
        )
        assert response.status_code == 200
        assert response.json() == {"signed_url": "https://signed/once"}
    finally:
        app.dependency_overrides.clear()


def test_prediction_image_signed_url_403_when_path_not_owned() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    class FakeRepoOther:
        def fetch_image_storage_path(self, _token: str, prediction_id: str) -> str | None:
            assert prediction_id == "33333333-3333-3333-3333-333333333333"
            return "other-user-id/a.png"

    def fake_svc() -> PredictionService:
        return PredictionService(repo=FakeRepoOther())

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_svc
    try:
        response = client.get(
            "/predictions/33333333-3333-3333-3333-333333333333/image-signed-url",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
        )
        assert response.status_code == 403
        assert response.json()["code"] == "image_path_forbidden"
    finally:
        app.dependency_overrides.clear()


def test_prediction_image_signed_url_404_when_no_image() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="p@example.com",
        created_at=None,
    )

    def fake_context() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    class FakeRepoEmpty:
        def fetch_image_storage_path(self, _token: str, prediction_id: str) -> str | None:
            return None

    def fake_svc() -> PredictionService:
        return PredictionService(repo=FakeRepoEmpty())

    app.dependency_overrides[api_deps.get_predict_context] = fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = fake_svc
    try:
        response = client.get(
            "/predictions/33333333-3333-3333-3333-333333333333/image-signed-url",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
        )
        assert response.status_code == 404
        assert response.json()["code"] == "prediction_image_not_found"
    finally:
        app.dependency_overrides.clear()
