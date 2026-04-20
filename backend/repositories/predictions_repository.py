"""Inserts into ``public.predictions`` using the caller's JWT (RLS)."""

from typing import Any

from postgrest import APIError

from backend.integrations.supabase_client import create_supabase_user_client
from backend.services.exceptions import PredictionServiceError


class PredictionsRepository:
    _SELECT_RETURN = (
        "id,risk,score,model_version,birth_date,age_months,notes,image_storage_path,created_at"
    )

    def insert_for_user(
        self,
        access_token: str,
        *,
        user_id: str,
        risk: str,
        score: float,
        model_version: str,
        age_months: int | None = None,
        birth_date: str | None = None,
        notes: str | None = None,
        image_storage_path: str | None = None,
    ) -> dict[str, Any]:
        client = create_supabase_user_client(access_token)
        payload: dict[str, Any] = {
            "user_id": user_id,
            "risk": risk,
            "score": score,
            "model_version": model_version,
            "age_months": age_months,
            "birth_date": birth_date,
            "notes": notes,
            "image_storage_path": image_storage_path,
        }
        try:
            result = (
                client.from_("predictions")
                .insert(payload)
                .select(self._SELECT_RETURN)
                .execute()
            )
        except APIError as e:
            msg = e.message or "Could not save prediction"
            raise PredictionServiceError(
                msg,
                502,
                code=e.code or "postgrest_error",
            ) from e
        rows = result.data
        if not isinstance(rows, list) or not rows:
            raise PredictionServiceError(
                "Empty insert response",
                502,
                code="empty_insert",
            )
        row = rows[0]
        if not isinstance(row, dict):
            raise PredictionServiceError(
                "Unexpected insert response",
                502,
                code="invalid_insert_shape",
            )
        return row

    def list_for_user(self, access_token: str) -> list[dict[str, Any]]:
        client = create_supabase_user_client(access_token)
        try:
            result = (
                client.from_("predictions")
                .select(self._SELECT_RETURN)
                .order("created_at", desc=True)
                .order("id", desc=True)
                .execute()
            )
        except APIError as e:
            msg = e.message or "Could not load predictions"
            raise PredictionServiceError(
                msg,
                502,
                code=e.code or "postgrest_error",
            ) from e
        rows = result.data
        if rows is None:
            return []
        if not isinstance(rows, list):
            raise PredictionServiceError(
                "Unexpected list response",
                502,
                code="invalid_list_shape",
            )
        return [r for r in rows if isinstance(r, dict)]

    def fetch_image_storage_path(
        self,
        access_token: str,
        prediction_id: str,
    ) -> str | None:
        """Path de imagen para una fila propia (RLS); ``None`` si no existe o no hay imagen."""
        client = create_supabase_user_client(access_token)
        try:
            result = (
                client.from_("predictions")
                .select("image_storage_path")
                .eq("id", prediction_id)
                .limit(1)
                .execute()
            )
        except APIError as e:
            msg = e.message or "Could not load prediction"
            raise PredictionServiceError(
                msg,
                502,
                code=e.code or "postgrest_error",
            ) from e
        rows = result.data
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0]
        if not isinstance(row, dict):
            return None
        p = row.get("image_storage_path")
        return str(p) if p else None
