"""FastAPI application factory.

Serves the JSON API and the single-page frontend from one process on one port, which
is what makes the "click-and-run" launcher a single command with no bundler or second
runtime (RESEARCH.md §4.2).

No CORS middleware is installed: the SPA is served same-origin, so cross-origin
access is unnecessary. The previous implementation combined ``allow_origins=["*"]``
with ``allow_credentials=True`` on a server handling medical images.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..config import Settings, settings as default_settings
from ..data.dataset import DatasetError
from ..logging_setup import get_logger
from . import clinical, dataset, evaluation, inference, meta, revisions
from .deps import get_repository, ml_available

logger = get_logger("api")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare output directories and log the runtime picture at startup.

    Model weights are deliberately *not* loaded here: an 8 GB load would block
    startup and would make the review tier unusable without a token. Loading is
    an explicit action via ``POST /api/model/load``.
    """
    settings: Settings = app.state.settings
    settings.ensure_output_dirs()
    logger.info("ATRIA EchoTrace %s starting", __version__)
    logger.info("Dataset directory: %s", settings.dataset_dir)
    logger.info("Output directory:  %s", settings.output_dir)

    try:
        report = get_repository().validate()
        logger.info(
            "Dataset: %d frames, %d cases, sources=%s",
            report.n_tracings,
            report.n_cases,
            report.source_counts,
        )
        if report.missing_pngs:
            logger.warning("%d tracing(s) have no PNG on disk", len(report.missing_pngs))
    except DatasetError as exc:
        logger.error("Dataset unavailable: %s", exc)

    if ml_available():
        from ..ml.runtime import configure_torch_allocator, describe_device

        configure_torch_allocator()
        device = describe_device(force_cpu=settings.force_cpu)
        logger.info(
            "AI tier available. Device=%s dtype=%s quantisation=%s",
            device.get("device"),
            device.get("compute_dtype"),
            device.get("quantization"),
        )
    else:
        logger.info(
            'AI tier not installed (review tier only). Install with: pip install -e ".[ai]"'
        )

    yield
    logger.info("ATRIA EchoTrace shutting down")


def _problem(status_code: int, title: str, detail: str) -> JSONResponse:
    """RFC 7807-style error body, used uniformly by all handlers."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": detail,
        },
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Override configuration (used by tests to point at a temporary
            dataset or output directory).
    """
    settings = settings or default_settings
    # Routers resolve configuration through api.deps; keep the two in step so an
    # explicitly supplied Settings actually takes effect.
    if settings is not default_settings:
        from .deps import set_settings

        set_settings(settings)

    app = FastAPI(
        title="ATRIA EchoTrace",
        version=__version__,
        description=(
            "MedGemma-driven echocardiographic contour tracing with human-in-the-loop "
            "revision. Research use only; not a medical device."
        ),
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings

    for router in (
        meta.router,
        dataset.router,
        clinical.router,
        revisions.router,
        inference.router,
        evaluation.router,
    ):
        app.include_router(router)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        if exc.status_code >= 500:
            logger.error("%s %s -> %s: %s", request.method, request.url.path, exc.status_code, detail)
        return _problem(exc.status_code, _TITLES.get(exc.status_code, "Error"), detail)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        messages = "; ".join(
            f"{'.'.join(str(part) for part in err.get('loc', ())[1:])}: {err.get('msg', '')}".strip(": ")
            for err in exc.errors()
        )
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Invalid request",
            messages or "Request body failed validation.",
        )

    @app.exception_handler(DatasetError)
    async def dataset_error_handler(request: Request, exc: DatasetError) -> JSONResponse:
        return _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Dataset unavailable", str(exc)
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal error",
            f"{type(exc).__name__}: {exc}",
        )

    # The SPA is mounted last so /api/* routes take precedence.
    if WEB_DIR.is_dir():
        app.mount("/", _RevalidatingStatics(directory=str(WEB_DIR), html=True), name="web")
    else:  # pragma: no cover - only when the package is installed without web assets

        @app.get("/")
        def missing_frontend() -> dict[str, Any]:
            return {
                "detail": f"Frontend assets not found at {WEB_DIR}",
                "api_docs": "/api/docs",
            }

    return app


class _RevalidatingStatics(StaticFiles):
    """Serve the SPA with ``Cache-Control: no-cache``.

    Starlette sends ``ETag`` and ``Last-Modified`` but no ``Cache-Control``, and a
    response with neither an explicit lifetime nor a ``no-cache`` directive may be
    assigned a *heuristic* freshness lifetime by the browser (RFC 9111 §4.2.2). The
    effect is that an upgraded install keeps executing the previous session's JavaScript
    until that heuristic expires. ``no-cache`` means "revalidate before reuse", not
    "do not store", so unchanged assets still answer 304 and cost nothing.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


_TITLES = {
    status.HTTP_400_BAD_REQUEST: "Bad request",
    status.HTTP_404_NOT_FOUND: "Not found",
    status.HTTP_409_CONFLICT: "Conflict",
    status.HTTP_422_UNPROCESSABLE_ENTITY: "Invalid request",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "Internal error",
    status.HTTP_503_SERVICE_UNAVAILABLE: "Service unavailable",
}


app = create_app()
