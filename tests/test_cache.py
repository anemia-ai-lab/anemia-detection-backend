"""Tests de caché in-memory (URLs firmadas y auth)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from storage3.exceptions import StorageApiError

from backend.core.auth_user_cache import (
    clear_auth_user_cache_for_tests,
    get_cached_user,
    set_cached_user,
)
from backend.repositories.prediction_images_storage import (
    PredictionImagesStorage,
    clear_signed_url_cache_for_tests,
)
from backend.schemas.auth import UserOut


def test_create_signed_url_uses_cache_on_second_call() -> None:
    clear_signed_url_cache_for_tests()
    storage = PredictionImagesStorage()
    mock_client = MagicMock()
    mock_client.storage.from_.return_value.create_signed_url.return_value = {
        "signedURL": "https://example.com/signed",
    }
    path = "11111111-1111-1111-1111-111111111111/abc.png"
    with patch(
        "backend.repositories.prediction_images_storage.create_supabase_user_client",
        return_value=mock_client,
    ):
        url1 = storage.create_signed_url("token-a", path)
        url2 = storage.create_signed_url("token-b", path)
    assert url1 == "https://example.com/signed"
    assert url2 == url1
    mock_client.storage.from_.return_value.create_signed_url.assert_called_once()


def test_delete_user_image_evicts_signed_url_cache() -> None:
    clear_signed_url_cache_for_tests()
    storage = PredictionImagesStorage()
    mock_client = MagicMock()
    mock_client.storage.from_.return_value.create_signed_url.return_value = {
        "signedURL": "https://example.com/signed",
    }
    mock_client.storage.from_.return_value.remove.return_value = None
    path = "11111111-1111-1111-1111-111111111111/abc.png"
    with patch(
        "backend.repositories.prediction_images_storage.create_supabase_user_client",
        return_value=mock_client,
    ):
        storage.create_signed_url("token", path)
        storage.delete_user_image("token", path)
        storage.create_signed_url("token", path)
    assert mock_client.storage.from_.return_value.create_signed_url.call_count == 2


def test_create_signed_url_still_raises_on_storage_error() -> None:
    clear_signed_url_cache_for_tests()
    storage = PredictionImagesStorage()
    mock_client = MagicMock()
    mock_client.storage.from_.return_value.create_signed_url.side_effect = StorageApiError(
        "fail",
        "signed_url_failed",
        502,
    )
    with patch(
        "backend.repositories.prediction_images_storage.create_supabase_user_client",
        return_value=mock_client,
    ):
        from backend.core.exceptions import PredictionServiceError

        with pytest.raises(PredictionServiceError):
            storage.create_signed_url("token", "user/file.png")


def test_auth_user_cache_hit_and_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_auth_user_cache_for_tests()
    monkeypatch.setattr(
        "backend.core.auth_user_cache.settings.auth_user_cache_ttl_seconds",
        60,
    )
    user = UserOut(id="u1", email="a@b.com", created_at=None)
    token = "aaa.bbb.ccc"
    assert get_cached_user(token) is None
    set_cached_user(token, user)
    assert get_cached_user(token) == user
