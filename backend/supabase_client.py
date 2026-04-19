"""Supabase client (supabase-py) for use from services, not from routers.

See https://supabase.com/docs/reference/python/initializing
"""

from functools import lru_cache

from supabase import Client, create_client

from .config import settings


def _require_setting(name: str, value: str) -> str:
    if not value or not value.strip():
        raise RuntimeError(f"Missing or empty configuration: {name}")
    return value.strip()


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Client using the publishable/anon key (`SUPABASE_KEY`).

    Subject to RLS. Prefer this for normal API usage.
    """
    url = _require_setting("SUPABASE_URL", settings.supabase_url)
    key = _require_setting("SUPABASE_KEY", settings.supabase_key)
    return create_client(url, key)


@lru_cache(maxsize=1)
def get_supabase_service_client() -> Client:
    """Client using `SUPABASE_SERVICE_ROLE_KEY` (bypasses RLS).

    Trusted server only. Never expose this key to browsers or clients.
    """
    url = _require_setting("SUPABASE_URL", settings.supabase_url)
    key = _require_setting(
        "SUPABASE_SERVICE_ROLE_KEY",
        settings.supabase_service_role_key,
    )
    return create_client(url, key)
