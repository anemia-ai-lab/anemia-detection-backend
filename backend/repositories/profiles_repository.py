"""``public.profiles`` via PostgREST + user JWT (RLS) or service role for bootstrap.

RLS (migración ``20260419200000_profiles``): SELECT, INSERT y UPDATE para rol
``authenticated`` solo cuando ``auth.uid() = id``.
"""

import logging
from typing import Any

from postgrest import APIError

from backend.integrations.supabase_client import (
    create_supabase_anon_client,
    get_supabase_service_client,
)

logger = logging.getLogger(__name__)


class ProfilesRepository:
    def try_insert_on_register(
        self,
        *,
        user_id: str,
        access_token: str | None,
    ) -> bool:
        """Idempotent upsert (``ON CONFLICT DO NOTHING``): no pisa datos si ya existe fila."""
        payload: dict[str, Any] = {
            "id": user_id,
            "first_name": None,
            "last_name": None,
            "department": None,
            "province": None,
            "profile_completed": False,
        }
        try:
            if access_token:
                client = create_supabase_anon_client()
                client.postgrest.auth(access_token)
                client.from_("profiles").upsert(
                    payload,
                    on_conflict="id",
                    ignore_duplicates=True,
                ).execute()
            else:
                client = get_supabase_service_client()
                client.from_("profiles").upsert(
                    payload,
                    on_conflict="id",
                    ignore_duplicates=True,
                ).execute()
            return True
        except APIError as e:
            logger.warning(
                "profiles try_insert_on_register failed user_id=%s: %s",
                user_id,
                e.message,
                exc_info=True,
            )
            return False

    def fetch_by_user_id(self, access_token: str, user_id: str) -> dict[str, Any] | None:
        client = create_supabase_anon_client()
        client.postgrest.auth(access_token)
        result = (
            client.from_("profiles")
            .select(
                "id,first_name,last_name,department,province,profile_completed,created_at",
            )
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        rows = result.data
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0]
        return row if isinstance(row, dict) else None

    def upsert_profile(
        self,
        access_token: str,
        *,
        user_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Insert or update the row for ``user_id`` (same RLS rules)."""
        client = create_supabase_anon_client()
        client.postgrest.auth(access_token)
        sel = (
            client.from_("profiles")
            .select("id")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        rows = sel.data
        exists = (
            isinstance(rows, list)
            and len(rows) == 1
            and isinstance(rows[0], dict)
            and str(rows[0].get("id")) == str(user_id)
        )
        try:
            if not exists:
                insert_body = {"id": user_id, **payload}
                res = client.from_("profiles").insert(insert_body).execute()
            else:
                res = (
                    client.from_("profiles")
                    .update(payload)
                    .eq("id", user_id)
                    .execute()
                )
        except APIError:
            logger.exception("profiles upsert_profile failed user_id=%s", user_id)
            raise
        rows = res.data
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise RuntimeError("Unexpected upsert response")
        return rows[0]
