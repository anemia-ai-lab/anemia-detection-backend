"""Supabase client (supabase-py) for use from services, not from routers.

See https://supabase.com/docs/reference/python/initializing
"""

from functools import lru_cache

from backend.core.config import settings
from supabase import Client, create_client


def _require_setting(name: str, value: str) -> str:
    if not value or not value.strip():
        raise RuntimeError(f"Missing or empty configuration: {name}")
    return value.strip()


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Singleton client using the publishable/anon key (`SUPABASE_KEY`).

    Do not use this for GoTrue ``auth.*`` flows in concurrent HTTP handlers:
    the SDK stores the session in process memory. Use
    :func:`create_supabase_anon_client` for sign-in / sign-up / ``get_user``.
    """
    url = _require_setting("SUPABASE_URL", settings.supabase_url)
    key = _require_setting("SUPABASE_KEY", settings.supabase_key)
    return create_client(url, key)


def create_supabase_anon_client() -> Client:
    """New anon client instance (no shared in-memory auth session)."""
    url = _require_setting("SUPABASE_URL", settings.supabase_url)
    key = _require_setting("SUPABASE_KEY", settings.supabase_key)
    return create_client(url, key)


def create_supabase_user_client(access_token: str) -> Client:
    """Anon project key + user JWT for PostgREST and Storage (RLS as ``auth.uid()``)."""
    url = _require_setting("SUPABASE_URL", settings.supabase_url)
    key = _require_setting("SUPABASE_KEY", settings.supabase_key)
    client = create_client(url, key)
    client.options.headers["Authorization"] = f"Bearer {access_token}"
    client._postgrest = None
    client._storage = None
    client._functions = None
    return client


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
