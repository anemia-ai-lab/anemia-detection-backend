"""Sync offline, historial paginado, detalle y delete."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from backend.api import deps as api_deps
from backend.core.prediction_cursor import encode_prediction_cursor
from backend.main import app
from backend.schemas.auth import UserOut
from backend.services.prediction_service import PredictionService
from tests.test_predict import _patch_identity_calibration, _skip_nail, skin_patch_png

client = TestClient(app)

USER_ID = "11111111-1111-1111-1111-111111111111"
PRED_ID = "33333333-3333-3333-3333-333333333333"
CLIENT_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TOKEN = "aaa.bbb.ccc"


def _user() -> UserOut:
    return UserOut(id=USER_ID, email="p@example.com", created_at=None)


def _fake_context() -> tuple[UserOut, str]:
    return (_user(), TOKEN)


def _base_row(**overrides: object) -> dict:
    created = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    row = {
        "id": PRED_ID,
        "user_id": USER_ID,
        "client_id": CLIENT_ID,
        "risk": "low",
        "score": 0.12,
        "model_version": "v2.0",
        "birth_date": None,
        "age_months": None,
        "notes": "campo",
        "image_storage_path": None,
        "inference_mode": "tflite_offline",
        "raw_probability": 0.15,
        "calibrated_probability": 0.12,
        "threshold_used": 0.168,
        "prediction": 0,
        "client_created_at": "2026-04-30T08:00:00+00:00",
        "image_sha256": "abc123",
        "synced_at": "2026-05-01T10:00:00+00:00",
        "preprocessing": {"fingers": ["thumb", "index"], "aggregation": "max"},
        "created_at": created.isoformat(),
        "effective_created_at": "2026-04-30T08:00:00+00:00",
    }
    row.update(overrides)
    return row


def _sync_item_payload() -> dict:
    return {
        "client_id": CLIENT_ID,
        "risk": "low",
        "score": 0.12,
        "raw_probability": 0.15,
        "calibrated_probability": 0.12,
        "threshold_used": 0.168,
        "prediction": 0,
        "model_version": "v2.0",
        "inference_mode": "tflite_offline",
        "client_created_at": "2026-04-30T08:00:00+00:00",
        "notes": "campo",
        "image_sha256": "abc123",
        "preprocessing": {"fingers": ["thumb", "index"], "aggregation": "max"},
    }


def test_sync_metadata_creates_row() -> None:
    inserted: dict | None = None

    class FakeRepo:
        def find_by_client_id(self, _token: str, *, user_id: str, client_id: str) -> dict | None:
            assert user_id == USER_ID
            assert client_id == CLIENT_ID
            return None

        def insert_for_user(self, _token: str, **kwargs) -> dict:
            nonlocal inserted
            inserted = kwargs
            return _base_row()

    app.dependency_overrides[api_deps.get_predict_context] = _fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: PredictionService(
        repo=FakeRepo()
    )
    try:
        response = client.post(
            "/predictions/sync/metadata",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={"items": [_sync_item_payload()]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["created"] is True
        assert data["results"][0]["image_pending"] is True
        assert data["results"][0]["id"] == PRED_ID
        assert inserted is not None
        assert inserted["inference_mode"] == "tflite_offline"
        assert inserted["preprocessing"] == {"fingers": ["thumb", "index"], "aggregation": "max"}
        assert inserted["image_storage_path"] is None
    finally:
        app.dependency_overrides.clear()


def test_sync_metadata_idempotent() -> None:
    class FakeRepo:
        def find_by_client_id(self, _token: str, *, user_id: str, client_id: str) -> dict | None:
            return _base_row(image_storage_path=f"{USER_ID}/img.png")

        def insert_for_user(self, *_a, **_k) -> dict:
            raise AssertionError("should not insert")

    app.dependency_overrides[api_deps.get_predict_context] = _fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: PredictionService(
        repo=FakeRepo()
    )
    try:
        response = client.post(
            "/predictions/sync/metadata",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={"items": [_sync_item_payload()]},
        )
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["created"] is False
        assert result["image_pending"] is False
    finally:
        app.dependency_overrides.clear()


def test_get_prediction_detail_with_preprocessing() -> None:
    class FakeRepo:
        def fetch_by_id(self, _token: str, prediction_id: str) -> dict | None:
            assert prediction_id == PRED_ID
            return _base_row(image_storage_path=f"{USER_ID}/a.png")

    class FakeImg:
        def create_signed_url(self, _token: str, path: str) -> str:
            assert path == f"{USER_ID}/a.png"
            return "https://signed/detail"

    app.dependency_overrides[api_deps.get_predict_context] = _fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: PredictionService(
        repo=FakeRepo(),
        images=FakeImg(),
    )
    try:
        response = client.get(
            f"/predictions/{PRED_ID}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["preprocessing"]["aggregation"] == "max"
        assert data["has_image"] is True
        assert data["image_signed_url"] == "https://signed/detail"
        assert data["inference_mode"] == "tflite_offline"
    finally:
        app.dependency_overrides.clear()


def test_list_predictions_pagination_cursor() -> None:
    t1 = datetime(2026, 5, 2, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    rows = [
        _base_row(
            id="44444444-4444-4444-4444-444444444444",
            effective_created_at=t1.isoformat(),
            created_at=t1.isoformat(),
        ),
        _base_row(effective_created_at=t2.isoformat(), created_at=t2.isoformat()),
    ]

    class FakeRepo:
        def list_for_user_paginated(
            self, _token: str, *, limit: int, cursor: str | None = None
        ) -> list:
            if cursor is None:
                return rows[: limit + 1]
            return rows[1:]

    app.dependency_overrides[api_deps.get_predict_context] = _fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: PredictionService(
        repo=FakeRepo()
    )
    try:
        response = client.get(
            "/predictions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            params={"limit": 1},
        )
        assert response.status_code == 200
        page1 = response.json()
        assert len(page1["items"]) == 1
        assert page1["items"][0]["id"] == "44444444-4444-4444-4444-444444444444"
        assert page1["next_cursor"] is not None

        response2 = client.get(
            "/predictions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            params={"limit": 1, "cursor": page1["next_cursor"]},
        )
        assert response2.status_code == 200
        page2 = response2.json()
        assert page2["items"][0]["id"] == PRED_ID
        assert page2["next_cursor"] is None
    finally:
        app.dependency_overrides.clear()


def test_delete_prediction_removes_row_and_storage() -> None:
    deleted_path: list[str] = []

    class FakeRepo:
        def delete_by_id(self, _token: str, prediction_id: str) -> dict | None:
            assert prediction_id == PRED_ID
            return {"id": PRED_ID, "image_storage_path": f"{USER_ID}/gone.png"}

    class FakeImg:
        def delete_user_image(self, _token: str, path: str) -> None:
            deleted_path.append(path)

    app.dependency_overrides[api_deps.get_predict_context] = _fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: PredictionService(
        repo=FakeRepo(),
        images=FakeImg(),
    )
    try:
        response = client.delete(
            f"/predictions/{PRED_ID}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 204
        assert deleted_path == [f"{USER_ID}/gone.png"]
    finally:
        app.dependency_overrides.clear()


def test_delete_prediction_404() -> None:
    class FakeRepo:
        def delete_by_id(self, _token: str, prediction_id: str) -> dict | None:
            return None

    app.dependency_overrides[api_deps.get_predict_context] = _fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: PredictionService(
        repo=FakeRepo()
    )
    try:
        response = client.delete(
            f"/predictions/{PRED_ID}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_upload_prediction_image_sha256_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_identity_calibration(monkeypatch)

    class FakeRepo:
        def fetch_by_id(self, _token: str, prediction_id: str) -> dict | None:
            return _base_row(image_sha256="deadbeef")

        def update_image_for_prediction(self, *_a, **_k) -> dict:
            raise AssertionError("should not update")

    app.dependency_overrides[api_deps.get_predict_context] = _fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: PredictionService(
        repo=FakeRepo(),
        nail_checker=_skip_nail,
    )
    try:
        response = client.post(
            f"/predictions/{PRED_ID}/image",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
            data={"image_sha256": "not-a-match"},
        )
        assert response.status_code == 409
        assert response.json()["code"] == "image_sha256_mismatch"
    finally:
        app.dependency_overrides.clear()


def test_upload_prediction_image_idempotent_when_exists() -> None:
    path = f"{USER_ID}/existing.png"

    class FakeRepo:
        def fetch_by_id(self, _token: str, prediction_id: str) -> dict | None:
            return _base_row(image_storage_path=path, image_sha256="hash")

    class FakeImg:
        def create_signed_url(self, _token: str, object_path: str) -> str:
            assert object_path == path
            return "https://signed/existing"

    app.dependency_overrides[api_deps.get_predict_context] = _fake_context
    app.dependency_overrides[api_deps.get_prediction_service] = lambda: PredictionService(
        repo=FakeRepo(),
        images=FakeImg(),
    )
    try:
        response = client.post(
            f"/predictions/{PRED_ID}/image",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"image": ("m.png", skin_patch_png(), "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["image_storage_path"] == path
        assert data["image_signed_url"] == "https://signed/existing"
    finally:
        app.dependency_overrides.clear()


def test_prediction_cursor_roundtrip() -> None:
    ts = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    encoded = encode_prediction_cursor(ts, PRED_ID)
    from backend.core.prediction_cursor import decode_prediction_cursor

    decoded_ts, decoded_id = decode_prediction_cursor(encoded)
    assert decoded_id == PRED_ID
    assert decoded_ts == ts


def test_list_predictions_invalid_cursor() -> None:
    app.dependency_overrides[api_deps.get_predict_context] = _fake_context
    try:
        response = client.get(
            "/predictions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            params={"cursor": "not-valid"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_cursor"
    finally:
        app.dependency_overrides.clear()
