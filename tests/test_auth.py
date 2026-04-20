from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from supabase_auth.errors import AuthApiError

from backend.api import deps as api_deps
from backend.core import config as config_module
from backend.main import app
from backend.repositories import auth_repository as auth_repository_module
from backend.repositories import profiles_repository as profiles_repository_module
from backend.schemas.auth import LoginAuthResponse, TokensOut, UserOut
from backend.schemas.profile import ProfileOut
from backend.services import auth_service as auth_service_module
from backend.services.auth_service import AuthService
from backend.services.exceptions import AuthServiceError
from tests.conftest import sample_auth_response, sample_user_response

client = TestClient(app)


class StubProfileService:
    """Evita PostgREST real en tests de login/register con sesión."""

    has_profile_row = True
    profile_completed = False

    def get_profile(self, user: UserOut, access_token: str) -> ProfileOut:
        return ProfileOut(
            id=user.id,
            email=user.email,
            has_profile_row=self.has_profile_row,
            profile_completed=self.profile_completed,
            first_name=None,
            last_name=None,
            department=None,
            province=None,
            created_at=None,
        )


def test_register_profile_insert_failure_returns_200_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MagicMock()
    mock_client.auth.sign_up.return_value = sample_auth_response(
        with_session=True,
        email="warn@example.com",
    )
    monkeypatch.setattr(config_module.settings, "debug", False)
    monkeypatch.setattr(
        auth_repository_module,
        "create_supabase_anon_client",
        lambda: mock_client,
    )
    monkeypatch.setattr(
        profiles_repository_module.ProfilesRepository,
        "try_insert_on_register",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(auth_service_module, "ProfileService", StubProfileService)
    response = client.post(
        "/auth/register",
        json={"email": "warn@example.com", "password": "longenough"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["profile_created"] is False
    assert len(data["warnings"]) == 1
    assert data["warnings"][0]["code"] == "profile_not_persisted"
    assert "message" in data["warnings"][0]
    assert data["has_profile_row"] is True
    assert data["profile_completed"] is False


def test_register_validation_too_short_password() -> None:
    response = client.post(
        "/auth/register",
        json={"email": "a@b.co", "password": "short"},
    )
    assert response.status_code == 422


def test_auth_service_error_returns_json_detail_and_code() -> None:
    def boom() -> AuthService:
        svc = MagicMock(spec=AuthService)
        svc.register.side_effect = AuthServiceError(
            "Email already registered",
            409,
            code="user_already_exists",
        )
        return svc

    app.dependency_overrides[api_deps.get_auth_service] = boom
    try:
        response = client.post(
            "/auth/register",
            json={"email": "new@example.com", "password": "longenough"},
        )
        assert response.status_code == 409
        assert response.json() == {
            "detail": "Email already registered",
            "code": "user_already_exists",
        }
    finally:
        app.dependency_overrides.clear()


def test_me_with_service_override() -> None:
    fixed_user = UserOut(
        id="u1",
        email="me@example.com",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    def fake_svc() -> AuthService:
        svc = MagicMock(spec=AuthService)
        svc.me.return_value = fixed_user
        return svc

    app.dependency_overrides[api_deps.get_auth_service] = fake_svc
    try:
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer aaa.bbb.ccc"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "u1"
        assert body["email"] == "me@example.com"
    finally:
        app.dependency_overrides.clear()


def test_login_success_shape_with_override() -> None:
    payload = LoginAuthResponse(
        user=UserOut(id="u2", email="x@y.co", created_at=None),
        tokens=TokensOut(
            access_token="a",
            refresh_token="r",
            expires_in=3600,
            token_type="bearer",
        ),
        has_profile_row=True,
        profile_completed=False,
    )

    def fake_svc() -> AuthService:
        svc = MagicMock(spec=AuthService)
        svc.login.return_value = payload
        return svc

    app.dependency_overrides[api_deps.get_auth_service] = fake_svc
    try:
        response = client.post(
            "/auth/login",
            json={"email": "x@y.co", "password": "secretpass"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["id"] == "u2"
        assert data["tokens"]["access_token"] == "a"
        assert "profile_created" not in data
        assert data["has_profile_row"] is True
        assert data["profile_completed"] is False
        assert data["warnings"] == []
    finally:
        app.dependency_overrides.clear()


def test_me_without_authorization_header() -> None:
    response = client.get("/auth/me")
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated",
        "code": "missing_token",
    }


def test_me_with_empty_bearer_token() -> None:
    response = client.get("/auth/me", headers={"Authorization": "Bearer "})
    assert response.status_code == 401
    # Starlette may omit empty credentials → treated as missing bearer.
    assert response.json()["code"] in ("empty_token", "missing_token")


def test_me_with_malformed_jwt_not_three_segments() -> None:
    response = client.get("/auth/me", headers={"Authorization": "Bearer onlyone"})
    assert response.status_code == 401
    assert response.json()["code"] == "malformed_token"


def test_me_with_two_segments_only() -> None:
    response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer part1.part2"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "malformed_token"


def test_me_with_valid_jwt_shape_calls_supabase_get_user(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock()
    mock_client.auth.get_user.return_value = sample_user_response(email="ok@example.com")

    monkeypatch.setattr(
        auth_repository_module,
        "create_supabase_anon_client",
        lambda: mock_client,
    )

    response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer eyJ.hbG.ccc"},
    )
    assert response.status_code == 200
    assert response.json()["email"] == "ok@example.com"
    mock_client.auth.get_user.assert_called_once_with("eyJ.hbG.ccc")


def test_me_supabase_rejects_token(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock()
    mock_client.auth.get_user.side_effect = AuthApiError(
        "invalid JWT",
        403,
        "bad_jwt",
    )

    monkeypatch.setattr(
        auth_repository_module,
        "create_supabase_anon_client",
        lambda: mock_client,
    )

    response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer aaa.bbb.ccc"},
    )
    assert response.status_code == 401
    body = response.json()
    assert "detail" in body
    assert body.get("code") == "bad_jwt"


def test_register_success_via_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock()
    mock_client.auth.sign_up.return_value = sample_auth_response(
        with_session=True,
        email="reg@example.com",
    )

    monkeypatch.setattr(config_module.settings, "debug", False)
    monkeypatch.setattr(
        auth_repository_module,
        "create_supabase_anon_client",
        lambda: mock_client,
    )
    monkeypatch.setattr(
        profiles_repository_module.ProfilesRepository,
        "try_insert_on_register",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(auth_service_module, "ProfileService", StubProfileService)

    response = client.post(
        "/auth/register",
        json={"email": "reg@example.com", "password": "longenough"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["email"] == "reg@example.com"
    assert data["tokens"]["token_type"] == "bearer"
    assert data["tokens"]["access_token"] == "aaa.bbb.ccc"
    assert data["profile_created"] is True
    assert data["has_profile_row"] is True
    assert data["profile_completed"] is False
    assert data["warnings"] == []
    mock_client.auth.sign_up.assert_called_once()


def test_register_without_session_profile_flags_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MagicMock()
    mock_client.auth.sign_up.return_value = sample_auth_response(
        with_session=False,
        email="pending@example.com",
    )
    monkeypatch.setattr(config_module.settings, "debug", False)
    monkeypatch.setattr(
        auth_repository_module,
        "create_supabase_anon_client",
        lambda: mock_client,
    )
    monkeypatch.setattr(
        profiles_repository_module.ProfilesRepository,
        "try_insert_on_register",
        lambda *args, **kwargs: True,
    )

    response = client.post(
        "/auth/register",
        json={"email": "pending@example.com", "password": "longenough"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tokens"] is None
    assert data["has_profile_row"] is None
    assert data["profile_completed"] is None


def test_register_success_does_not_include_supabase_raw_when_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MagicMock()
    mock_client.auth.sign_up.return_value = sample_auth_response(
        with_session=True,
        email="raw@example.com",
    )

    monkeypatch.setattr(
        auth_repository_module,
        "create_supabase_anon_client",
        lambda: mock_client,
    )
    monkeypatch.setattr(config_module.settings, "debug", True)
    monkeypatch.setattr(
        profiles_repository_module.ProfilesRepository,
        "try_insert_on_register",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(auth_service_module, "ProfileService", StubProfileService)

    response = client.post(
        "/auth/register",
        json={"email": "raw@example.com", "password": "longenough"},
    )
    assert response.status_code == 200
    assert "supabase_raw" not in response.json()


def test_login_success_via_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock()
    mock_client.auth.sign_in_with_password.return_value = sample_auth_response(
        email="login@example.com",
    )

    monkeypatch.setattr(config_module.settings, "debug", False)
    monkeypatch.setattr(
        auth_repository_module,
        "create_supabase_anon_client",
        lambda: mock_client,
    )
    monkeypatch.setattr(auth_service_module, "ProfileService", StubProfileService)

    response = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "secretpass"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["email"] == "login@example.com"
    assert data["has_profile_row"] is True
    assert data["profile_completed"] is False
    mock_client.auth.sign_in_with_password.assert_called_once()


def test_login_invalid_credentials_from_supabase(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock()
    mock_client.auth.sign_in_with_password.side_effect = AuthApiError(
        "Invalid login credentials",
        400,
        "invalid_credentials",
    )

    monkeypatch.setattr(config_module.settings, "debug", False)
    monkeypatch.setattr(
        auth_repository_module,
        "create_supabase_anon_client",
        lambda: mock_client,
    )

    response = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "wrong"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "invalid_credentials"
