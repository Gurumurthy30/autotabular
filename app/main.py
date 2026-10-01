import os
import warnings
from contextlib import asynccontextmanager

# Suppress noisy deprecation and agent hint warnings from third-party libraries
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
warnings.filterwarnings("ignore")


from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.config import validate_config
from app.db.session import init_db
from app.utils.logger import get_logger

_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate environment and configuration
    validate_config()
    # Initialize DB tables on startup
    init_db()
    _log.info(
        "[STARTUP] Autonomous Tabular ML Platform started | DB=configured | MLFLOW=configured"
    )
    yield


app = FastAPI(
    title="Autonomous Tabular ML Engineering API",
    description="Backend API for LangGraph-orchestrated tabular machine learning workflow.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Exception Handlers ---

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handles standard HTTPExceptions cleanly without leaking internal stack traces."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "status_code": exc.status_code,
            "error": "HTTPException",
            "detail": exc.detail,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic request payload validation errors with informative messages."""
    formatted_errors = []
    for err in exc.errors():
        loc = " -> ".join(str(item) for item in err.get("loc", []))
        msg = err.get("msg", "")
        formatted_errors.append(f"{loc}: {msg}" if loc else msg)

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "status_code": 422,
            "error": "ValidationError",
            "detail": formatted_errors or exc.errors(),
        },
    )


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handles database exceptions safely and logs the traceback."""
    _log.error(f"Database error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "status_code": 500,
            "error": "DatabaseError",
            "detail": "A database error occurred while processing the request.",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all unhandled exception handler to return structured JSON and log tracebacks."""
    _log.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "status_code": 500,
            "error": "InternalServerError",
            "detail": str(exc),
        },
    )


app.include_router(router)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Autonomous Tabular ML Engineering API is running"}

