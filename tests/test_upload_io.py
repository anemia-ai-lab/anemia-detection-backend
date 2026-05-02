"""Pruebas de lectura acotada de subidas (sin cargar más allá del límite)."""

from __future__ import annotations

import asyncio

import pytest

from backend.core.upload_io import UploadExceedsMaxBytesError, read_upload_file_with_byte_limit


class _FakeUpload:
    """Emula UploadFile con lecturas por trozos (sin materializar el fichero completo)."""

    def __init__(self, chunk_len: int, num_chunks: int) -> None:
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.read_count = 0

    async def read(self, _chunk_size: int) -> bytes:
        # Starlette pasa el tamaño solicitado; este fake usa chunk_len fijo.
        if self.read_count >= self.num_chunks:
            return b""
        self.read_count += 1
        return b"x" * self.chunk_len


def test_read_upload_stops_before_accumulating_beyond_limit() -> None:
    """Superar max_bytes en la acumulación dispara error sin seguir leyendo trozos extra."""

    async def _run() -> None:
        fake = _FakeUpload(chunk_len=30, num_chunks=10)
        with pytest.raises(UploadExceedsMaxBytesError):
            await read_upload_file_with_byte_limit(fake, max_bytes=50)
        assert fake.read_count == 2

    asyncio.run(_run())


def test_read_upload_joins_chunks_under_limit() -> None:

    async def _run() -> None:
        fake = _FakeUpload(chunk_len=10, num_chunks=3)
        out = await read_upload_file_with_byte_limit(fake, max_bytes=100)
        assert out == b"x" * 30
        assert fake.read_count == 3

    asyncio.run(_run())
