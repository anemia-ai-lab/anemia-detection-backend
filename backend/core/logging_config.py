"""Configuración mínima de logging para la aplicación (sin dependencias extra)."""

from __future__ import annotations

import logging
import sys

from backend.core.config import settings


def configure_logging() -> None:
    """Registra un manejador en el logger raíz ``backend`` si aún no hay uno."""
    level = logging.DEBUG if settings.debug else logging.INFO
    log = logging.getLogger("backend")
    log.setLevel(level)
    if log.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ),
    )
    log.addHandler(handler)
    log.propagate = False
