"""Lectura acotada de subidas multipart (sin cargar el fichero completo por encima del límite)."""

from __future__ import annotations

from fastapi import UploadFile


class UploadExceedsMaxBytesError(Exception):
    """El cuerpo leído supera el máximo configurado (tamaño acumulado > límite)."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(max_bytes)


# Tamaño de lectura por iteración; alinea con buffers habituales y limita picos de memoria por trozo.
_CHUNK = 64 * 1024


async def read_upload_file_with_byte_limit(upload: UploadFile, max_bytes: int) -> bytes:
    """
    Lee ``upload`` por trozos; deja de acumular en cuanto el total supera ``max_bytes``.

    No garantiza que el cliente deje de enviar datos en red; eso debe limitarlo el proxy.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        data = await upload.read(_CHUNK)
        if not data:
            break
        total += len(data)
        if total > max_bytes:
            raise UploadExceedsMaxBytesError(max_bytes)
        chunks.append(data)
    return b"".join(chunks)
