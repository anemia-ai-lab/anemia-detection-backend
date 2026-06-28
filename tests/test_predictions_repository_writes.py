"""PostgREST write paths must not chain .select() (unsupported in postgrest 2.28.x)."""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.repositories.predictions_repository import PredictionsRepository


class _FakeExecute:
    data = [{"id": "pred-1", "risk": "low", "score": 0.1}]


def test_insert_and_fetch_uses_insert_execute_without_select() -> None:
    table = MagicMock()
    insert_builder = MagicMock()
    insert_builder.execute.return_value = _FakeExecute()
    table.insert.return_value = insert_builder

    client = MagicMock()
    client.from_.return_value = table

    row = PredictionsRepository()._insert_and_fetch(client, {"user_id": "u1", "risk": "low"})
    assert row["id"] == "pred-1"
    table.insert.assert_called_once()
    insert_builder.select.assert_not_called()
    insert_builder.execute.assert_called_once()
