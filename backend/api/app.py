from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from backend.api.routes.auth import router as auth_router
from backend.api.routes.predict import router as predict_router
from backend.core.config import settings
from backend.services.exceptions import ClientHttpError

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    openapi_tags=[
        {"name": "health", "description": "Comprobación de disponibilidad."},
        {
            "name": "auth",
            "description": (
                "Supabase Auth: registro, login, /me, perfil (GET/PATCH). "
                "El email del perfil sale de auth, no de la tabla."
            ),
        },
        {
            "name": "predictions",
            "description": "Predicción mock y lectura del historial en Supabase.",
        },
    ],
)


@app.exception_handler(ClientHttpError)
async def client_http_error_handler(
    _request: Request,
    exc: ClientHttpError,
) -> JSONResponse:
    body: dict[str, str] = {"detail": exc.message}
    if exc.code:
        body["code"] = exc.code
    return JSONResponse(status_code=exc.status_code, content=body)


@app.get("/health", tags=["health"], summary="Health check")
def health():
    return {"status": "ok"}


app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(predict_router)
