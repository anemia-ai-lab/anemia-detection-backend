"""Supabase GoTrue calls (no business rules)."""

from supabase_auth import AuthResponse, UserResponse

from backend.integrations.supabase_client import create_supabase_anon_client


class AuthRepository:
    def sign_up_with_email(self, email: str, password: str) -> AuthResponse:
        client = create_supabase_anon_client()
        return client.auth.sign_up({"email": email, "password": password})

    def sign_in_with_email(self, email: str, password: str) -> AuthResponse:
        client = create_supabase_anon_client()
        return client.auth.sign_in_with_password({"email": email, "password": password})

    def get_user(self, access_token: str) -> UserResponse:
        client = create_supabase_anon_client()
        result = client.auth.get_user(access_token)
        assert result is not None  # jwt provided: GoTrue returns user or raises
        return result
