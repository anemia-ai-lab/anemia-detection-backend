from datetime import UTC, datetime
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.api import deps as api_deps
from backend.main import app
from backend.schemas.auth import UserOut
from backend.schemas.profile import ProfileOut
from backend.services.profile_service import ProfileService

client = TestClient(app)


def test_get_profile_without_token() -> None:
    r = client.get("/auth/me/profile")
    assert r.status_code == 401


def test_get_profile_no_row_returns_flags() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="u@example.com",
        created_at=None,
    )

    def fake_ctx() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    svc = MagicMock(spec=ProfileService)
    svc.get_profile.return_value = ProfileOut(
        id=user.id,
        email=user.email,
        has_profile_row=False,
        profile_completed=False,
        first_name=None,
        last_name=None,
        department=None,
        province=None,
        created_at=None,
    )

    app.dependency_overrides[api_deps.get_predict_context] = fake_ctx
    app.dependency_overrides[api_deps.get_profile_service] = lambda: svc
    try:
        r = client.get(
            "/auth/me/profile",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
        )
        assert r.status_code == 200
        b = r.json()
        assert b["has_profile_row"] is False
        assert b["profile_completed"] is False
        assert b["email"] == "u@example.com"
        assert b["created_at"] is None
    finally:
        app.dependency_overrides.clear()


def test_patch_profile_delegates_to_service() -> None:
    user = UserOut(
        id="11111111-1111-1111-1111-111111111111",
        email="u@example.com",
        created_at=None,
    )
    out = ProfileOut(
        id=user.id,
        email=user.email,
        has_profile_row=True,
        profile_completed=True,
        first_name="A",
        last_name="B",
        department=None,
        province=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    def fake_ctx() -> tuple[UserOut, str]:
        return (user, "aaa.bbb.ccc")

    svc = MagicMock(spec=ProfileService)
    svc.update_profile.return_value = out

    app.dependency_overrides[api_deps.get_predict_context] = fake_ctx
    app.dependency_overrides[api_deps.get_profile_service] = lambda: svc
    try:
        r = client.patch(
            "/auth/me/profile",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
            json={"first_name": "A", "last_name": "B"},
        )
        assert r.status_code == 200
        assert r.json()["profile_completed"] is True
        svc.update_profile.assert_called_once()
    finally:
        app.dependency_overrides.clear()
