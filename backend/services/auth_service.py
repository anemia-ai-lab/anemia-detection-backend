from supabase_auth import AuthResponse, User
from supabase_auth.errors import AuthError

from backend.core.exceptions import AuthServiceError
from backend.repositories.auth_repository import AuthRepository
from backend.repositories.profiles_repository import ProfilesRepository
from backend.schemas.auth import (
    LoginAuthResponse,
    RegisterAuthResponse,
    TokensOut,
    UserOut,
    WarningItem,
)
from backend.services.auth_error_mapping import map_supabase_auth_error
from backend.services.profile_service import ProfileService


class AuthService:
    """Auth flows backed by Supabase GoTrue (delegates persistence to repository)."""

    def __init__(
        self,
        repo: AuthRepository | None = None,
        profiles: ProfilesRepository | None = None,
        profile_service: ProfileService | None = None,
    ) -> None:
        self._repo = repo or AuthRepository()
        self._profiles = profiles or ProfilesRepository()
        self._profile_service = profile_service or ProfileService()

    def register(self, email: str, password: str) -> RegisterAuthResponse:
        try:
            raw = self._repo.sign_up_with_email(email, password)
        except AuthError as e:
            raise map_supabase_auth_error(e) from e
        if raw.user is None:
            raise AuthServiceError(
                "Unexpected auth response",
                502,
                code="empty_user",
            )
        access_token = raw.session.access_token if raw.session else None
        profile_ok = self._profiles.try_insert_on_register(
            user_id=raw.user.id,
            access_token=access_token,
        )
        warnings: list[WarningItem] = []
        if not profile_ok:
            warnings.append(
                WarningItem(
                    code="profile_not_persisted",
                    message=(
                        "No se pudo asegurar la fila de perfil; "
                        "completa o reintenta con PATCH /auth/me/profile."
                    ),
                ),
            )
        return self._register_session_response(
            raw,
            profile_created=profile_ok,
            warnings=warnings,
        )

    def login(self, email: str, password: str) -> LoginAuthResponse:
        try:
            raw = self._repo.sign_in_with_email(email, password)
        except AuthError as e:
            raise map_supabase_auth_error(e, prefer_unauthorized=True) from e
        return self._login_session_response(raw, warnings=[])

    def me(self, access_token: str) -> UserOut:
        """Resolve current user; ``access_token`` must already pass bearer + shape checks."""
        try:
            ur = self._repo.get_user(access_token)
        except AuthError as e:
            raise map_supabase_auth_error(e, prefer_unauthorized=True) from e
        return self._user_to_out(ur.user)

    def _login_session_response(
        self,
        raw: AuthResponse,
        *,
        warnings: list[WarningItem] | None = None,
    ) -> LoginAuthResponse:
        user_out, tokens = self._session_user_and_tokens(raw)
        access = tokens.access_token if tokens else None
        hr, pc = self._profile_flags(user_out, access)
        return LoginAuthResponse(
            user=user_out,
            tokens=tokens,
            has_profile_row=hr,
            profile_completed=pc,
            warnings=warnings or [],
        )

    def _register_session_response(
        self,
        raw: AuthResponse,
        *,
        profile_created: bool,
        warnings: list[WarningItem] | None = None,
    ) -> RegisterAuthResponse:
        user_out, tokens = self._session_user_and_tokens(raw)
        access = tokens.access_token if tokens else None
        hr, pc = self._profile_flags(user_out, access)
        return RegisterAuthResponse(
            user=user_out,
            tokens=tokens,
            has_profile_row=hr,
            profile_completed=pc,
            profile_created=profile_created,
            warnings=warnings or [],
        )

    def _profile_flags(
        self,
        user_out: UserOut,
        access_token: str | None,
    ) -> tuple[bool | None, bool | None]:
        if access_token is None:
            return None, None
        profile = self._profile_service.get_profile(user_out, access_token)
        return profile.has_profile_row, profile.profile_completed

    def _session_user_and_tokens(
        self,
        raw: AuthResponse,
    ) -> tuple[UserOut, TokensOut | None]:
        if raw.user is None:
            raise AuthServiceError(
                "Unexpected auth response",
                502,
                code="empty_user",
            )
        tokens = None
        if raw.session is not None:
            s = raw.session
            tokens = TokensOut(
                access_token=s.access_token,
                refresh_token=s.refresh_token,
                expires_in=s.expires_in,
                token_type=s.token_type,
            )
        return self._user_to_out(raw.user), tokens

    @staticmethod
    def _user_to_out(user: User) -> UserOut:
        return UserOut(
            id=user.id,
            email=user.email,
            created_at=user.created_at,
        )
