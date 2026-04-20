from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.core.auth_token import (
    TokenValidationError,
    parse_bearer_token,
    require_well_formed_jwt,
)
from backend.schemas.auth import UserOut
from backend.services.auth_service import AuthService
from backend.services.exceptions import AuthServiceError
from backend.services.model_evaluation_service import ModelEvaluationService
from backend.services.prediction_service import PredictionService
from backend.services.profile_service import ProfileService

http_bearer_optional = HTTPBearer(auto_error=False)


def get_access_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(http_bearer_optional),
    ],
) -> str:
    try:
        token = parse_bearer_token(credentials)
        require_well_formed_jwt(token)
    except TokenValidationError as e:
        raise AuthServiceError(e.message, 401, code=e.code) from e
    return token


def get_auth_service() -> AuthService:
    return AuthService()


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
AccessTokenDep = Annotated[str, Depends(get_access_token)]


def get_prediction_service() -> PredictionService:
    return PredictionService()


PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]


def get_model_evaluation_service() -> ModelEvaluationService:
    return ModelEvaluationService()


ModelEvaluationServiceDep = Annotated[
    ModelEvaluationService,
    Depends(get_model_evaluation_service),
]


def get_profile_service() -> ProfileService:
    return ProfileService()


ProfileServiceDep = Annotated[ProfileService, Depends(get_profile_service)]


def get_predict_context(
    token: AccessTokenDep,
    auth: AuthServiceDep,
) -> tuple[UserOut, str]:
    """Verified user + same access token for PostgREST RLS."""
    return (auth.me(token), token)


PredictContextDep = Annotated[tuple[UserOut, str], Depends(get_predict_context)]
