"""NDJSON debug logs for agent debug sessions (no secrets)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_DEBUG_LOG_PATH = Path(__file__).resolve().parents[2] / ".cursor" / "debug-45fd61.log"
_SESSION_ID = "45fd61"


def agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": _SESSION_ID,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
            "runId": run_id,
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # #endregion
