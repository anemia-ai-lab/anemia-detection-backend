"""Clave de cliente para rate limiting (cabeceras de proxy)."""

from __future__ import annotations

import pytest
from starlette.requests import Request

from backend.core import config as config_module
from backend.core.rate_limit import rate_limit_client_key


def _request(*, client_host: str, x_forwarded_for: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if x_forwarded_for is not None:
        headers.append((b"x-forwarded-for", x_forwarded_for.encode("ascii")))
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "method": "POST",
            "path": "/predict",
            "raw_path": b"/predict",
            "root_path": "",
            "scheme": "http",
            "query_string": b"",
            "headers": headers,
            "client": (client_host, 12345),
            "server": ("testserver", 80),
        }
    )


def test_client_key_uses_socket_host_when_proxy_headers_untrusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module.settings, "trust_proxy_headers", False)
    req = _request(client_host="10.0.0.2", x_forwarded_for="203.0.113.99")
    assert rate_limit_client_key(req) == "10.0.0.2"


def test_client_key_uses_xff_first_hop_when_proxy_headers_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module.settings, "trust_proxy_headers", True)
    req = _request(client_host="10.0.0.2", x_forwarded_for="203.0.113.5, 198.51.100.1")
    assert rate_limit_client_key(req) == "203.0.113.5"


def test_spoofed_xff_does_not_change_key_when_trust_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Misma IP de socket: rate limit no se elude variando X-Forwarded-For."""
    monkeypatch.setattr(config_module.settings, "trust_proxy_headers", False)
    a = _request(client_host="198.51.100.10", x_forwarded_for="1.1.1.1")
    b = _request(client_host="198.51.100.10", x_forwarded_for="2.2.2.2")
    assert rate_limit_client_key(a) == rate_limit_client_key(b) == "198.51.100.10"
