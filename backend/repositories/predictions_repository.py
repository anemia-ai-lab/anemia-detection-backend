"""Inserts into ``public.predictions`` using the caller's JWT (RLS)."""

import logging
from datetime import UTC, datetime
from typing import Any

from postgrest import APIError

from backend.core.exceptions import PredictionServiceError
from backend.core.prediction_cursor import decode_prediction_cursor
from backend.integrations.supabase_client import create_supabase_user_client

logger = logging.getLogger(__name__)

_SELECT_RETURN = (
    "id,user_id,client_id,risk,score,model_version,birth_date,age_months,notes,"
    "image_storage_path,inference_mode,raw_probability,calibrated_probability,"
    "threshold_used,prediction,client_created_at,image_sha256,synced_at,preprocessing,"
    "created_at,effective_created_at"
)


class PredictionsRepository:
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
        inference_mode: str = "backend",
        raw_probability: float | None = None,
        calibrated_probability: float | None = None,
        threshold_used: float | None = None,
        prediction: int | None = None,
        client_id: str | None = None,
        client_created_at: str | None = None,
        image_sha256: str | None = None,
        synced_at: str | None = None,
        preprocessing: dict[str, Any] | None = None,
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
            "inference_mode": inference_mode,
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "threshold_used": threshold_used,
            "prediction": prediction,
            "client_id": client_id,
            "client_created_at": client_created_at,
            "image_sha256": image_sha256,
            "synced_at": synced_at,
            "preprocessing": preprocessing,
        }
        return self._insert_and_fetch(client, payload)

    def find_by_client_id(
        self,
        access_token: str,
        *,
        user_id: str,
        client_id: str,
    ) -> dict[str, Any] | None:
        client = create_supabase_user_client(access_token)
        try:
            result = (
                client.from_("predictions")
                .select(_SELECT_RETURN)
                .eq("user_id", user_id)
                .eq("client_id", client_id)
                .limit(1)
                .execute()
            )
        except APIError as e:
            self._raise_db(e, "Could not load prediction by client_id", op="find_client_id")
        rows = result.data
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0]
        return row if isinstance(row, dict) else None

    def fetch_by_id(self, access_token: str, prediction_id: str) -> dict[str, Any] | None:
        client = create_supabase_user_client(access_token)
        try:
            result = (
                client.from_("predictions")
                .select(_SELECT_RETURN)
                .eq("id", prediction_id)
                .limit(1)
                .execute()
            )
        except APIError as e:
            self._raise_db(e, "Could not load prediction", op="fetch_by_id")
        rows = result.data
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0]
        return row if isinstance(row, dict) else None

    def update_image_for_prediction(
        self,
        access_token: str,
        prediction_id: str,
        *,
        image_storage_path: str,
        image_sha256: str | None = None,
    ) -> dict[str, Any]:
        client = create_supabase_user_client(access_token)
        payload: dict[str, Any] = {"image_storage_path": image_storage_path}
        if image_sha256 is not None:
            payload["image_sha256"] = image_sha256
        try:
            client.from_("predictions").update(payload).eq("id", prediction_id).execute()
        except APIError as e:
            self._raise_db(e, "Could not update prediction image", op="update_image")
        row = self.fetch_by_id(access_token, prediction_id)
        if row is None:
            raise PredictionServiceError(
                "Prediction not found after image update",
                502,
                code="update_fetch_miss",
            )
        return row

    def delete_by_id(self, access_token: str, prediction_id: str) -> bool:
        """Delete row; returns True if a row was removed."""
        client = create_supabase_user_client(access_token)
        try:
            result = (
                client.from_("predictions").delete().eq("id", prediction_id).select("id").execute()
            )
        except APIError as e:
            self._raise_db(e, "Could not delete prediction", op="delete")
        rows = result.data
        return isinstance(rows, list) and len(rows) > 0

    def list_for_user_paginated(
        self,
        access_token: str,
        *,
        limit: int,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        cursor_ts = None
        cursor_id: str | None = None
        if cursor:
            cursor_ts, cursor_id = decode_prediction_cursor(cursor)
        client = create_supabase_user_client(access_token)
        query = (
            client.from_("predictions")
            .select(_SELECT_RETURN)
            .order("effective_created_at", desc=True)
            .order("id", desc=True)
            .limit(limit + 1)
        )
        if cursor_ts is not None and cursor_id is not None:
            ts_iso = cursor_ts.astimezone(UTC).isoformat()
            query = query.or_(
                f"effective_created_at.lt.{ts_iso},"
                f"and(effective_created_at.eq.{ts_iso},id.lt.{cursor_id})",
            )
        try:
            result = query.execute()
        except APIError as e:
            self._raise_db(e, "Could not load predictions", op="list_paginated")
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

    def list_for_user(self, access_token: str) -> list[dict[str, Any]]:
        """Legacy: all rows (tests / compat). Prefer ``list_for_user_paginated``."""
        client = create_supabase_user_client(access_token)
        try:
            result = (
                client.from_("predictions")
                .select(_SELECT_RETURN)
                .order("effective_created_at", desc=True)
                .order("id", desc=True)
                .execute()
            )
        except APIError as e:
            self._raise_db(e, "Could not load predictions", op="list")
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
        row = self.fetch_by_id(access_token, prediction_id)
        if row is None:
            return None
        p = row.get("image_storage_path")
        return str(p) if p else None

    def _insert_and_fetch(self, client: Any, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            insert_result = client.from_("predictions").insert(payload).execute()
        except APIError as e:
            self._raise_db(e, "Could not save prediction", op="insert")
        ins_rows = insert_result.data
        if not isinstance(ins_rows, list) or not ins_rows:
            raise PredictionServiceError(
                "Empty insert response",
                502,
                code="empty_insert",
            )
        ins_first = ins_rows[0]
        if not isinstance(ins_first, dict):
            raise PredictionServiceError(
                "Unexpected insert response",
                502,
                code="invalid_insert_shape",
            )
        new_id = ins_first.get("id")
        if new_id is None:
            raise PredictionServiceError(
                "Insert did not return a row id",
                502,
                code="insert_no_id",
            )
        try:
            fetch_result = (
                client.from_("predictions")
                .select(_SELECT_RETURN)
                .eq("id", new_id)
                .limit(1)
                .execute()
            )
        except APIError as e:
            self._raise_db(e, "Could not load prediction after insert", op="select_after_insert")
        rows = fetch_result.data
        if not isinstance(rows, list) or not rows:
            raise PredictionServiceError(
                "Inserted prediction not found after save",
                502,
                code="insert_fetch_miss",
            )
        row = rows[0]
        if not isinstance(row, dict):
            raise PredictionServiceError(
                "Unexpected fetch response",
                502,
                code="invalid_fetch_shape",
            )
        return row

    @staticmethod
    def _raise_db(exc: APIError, message: str, *, op: str) -> None:
        msg = exc.message or message
        logger.warning(
            "predictions_db_error op=%s code=%s message=%s",
            op,
            exc.code or "postgrest_error",
            msg,
        )
        raise PredictionServiceError(
            message,
            502,
            code=exc.code or "postgrest_error",
        ) from exc


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
