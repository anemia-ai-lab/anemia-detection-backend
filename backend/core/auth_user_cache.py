"""Caché in-memory de corta duración para ``auth.get_user`` (reduce latencia por JWT)."""

from __future__ import annotations

import hashlib

from cachetools import TTLCache

from backend.core.config import settings
from backend.schemas.auth import UserOut

_cache: TTLCache[str, UserOut] | None = None


def _get_cache() -> TTLCache[str, UserOut] | None:
    global _cache
    ttl = int(settings.auth_user_cache_ttl_seconds)
    if ttl <= 0:
        return None
    if _cache is None or _cache.ttl != ttl:
        _cache = TTLCache(maxsize=4096, ttl=ttl)
    return _cache


def _cache_key(access_token: str) -> str:
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()


def get_cached_user(access_token: str) -> UserOut | None:
    cache = _get_cache()
    if cache is None:
        return None
    return cache.get(_cache_key(access_token))


def set_cached_user(access_token: str, user: UserOut) -> None:
    cache = _get_cache()
    if cache is None:
        return
    cache[_cache_key(access_token)] = user


def clear_auth_user_cache_for_tests() -> None:
    """Vacía la caché de usuario (tests)."""
    global _cache
    if _cache is not None:
        _cache.clear()
