from fastapi import APIRouter

from backend.api.deps import (
    AccessTokenDep,
    AuthServiceDep,
    PredictContextDep,
    ProfileServiceDep,
)
from backend.schemas.auth import (
    LoginAuthResponse,
    LoginRequest,
    RegisterAuthResponse,
    RegisterRequest,
    UserOut,
)
from backend.schemas.errors import ErrorResponse
from backend.schemas.profile import ProfileOut, ProfilePatchRequest

router = APIRouter()

_AUTH_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse, "description": "Bad request (e.g. weak password)."},
    401: {
        "model": ErrorResponse,
        "description": "Missing/invalid bearer token or invalid credentials.",
    },
    409: {"model": ErrorResponse, "description": "Conflict (e.g. user already registered)."},
    502: {"model": ErrorResponse, "description": "Error upstream de Auth o de lectura de perfil."},
}


@router.post(
    "/register",
    response_model=RegisterAuthResponse,
    responses=_AUTH_ERROR_RESPONSES,
    summary="Register (auth user + intento de fila en profiles)",
)
def register(body: RegisterRequest, auth: AuthServiceDep) -> RegisterAuthResponse:
    return auth.register(body.email, body.password)


@router.post(
    "/login",
    response_model=LoginAuthResponse,
    responses=_AUTH_ERROR_RESPONSES,
    summary="Login",
)
def login(body: LoginRequest, auth: AuthServiceDep) -> LoginAuthResponse:
    return auth.login(body.email, body.password)


@router.get(
    "/me",
    response_model=UserOut,
    responses=_AUTH_ERROR_RESPONSES,
    summary="Current user (JWT)",
)
def me(token: AccessTokenDep, auth: AuthServiceDep) -> UserOut:
    return auth.me(token)


@router.get(
    "/me/profile",
    response_model=ProfileOut,
    responses=_AUTH_ERROR_RESPONSES,
    summary="Mi perfil (tabla profiles + email desde auth)",
)
def get_my_profile(
    ctx: PredictContextDep,
    profiles: ProfileServiceDep,
) -> ProfileOut:
    user, access_token = ctx
    return profiles.get_profile(user, access_token)


@router.patch(
    "/me/profile",
    response_model=ProfileOut,
    responses=_AUTH_ERROR_RESPONSES,
    summary="Actualizar mi perfil (crea fila si no existía)",
)
def patch_my_profile(
    body: ProfilePatchRequest,
    ctx: PredictContextDep,
    profiles: ProfileServiceDep,
) -> ProfileOut:
    user, access_token = ctx
    return profiles.update_profile(user, access_token, body)
