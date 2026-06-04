import logging
import sys
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings
from schemas import ErrorResponse

# Сервисы
from services.ollama_service import OllamaService
from services.qdrant_service import QdrantService
from services.redis_service import RedisSessionService
from core.router import QueryRouter

# Роутер и реестр
from api.routes import router, limiter, ServiceRegistry

def setup_logging() -> None:
    """Настраивает структурированное логирование в JSON-формате."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

setup_logging()
logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("Запуск приложения «МедСовет»...")

    ServiceRegistry.ollama_service = OllamaService(
        base_url=settings.OLLAMA_BASE_URL,
        model_name=settings.OLLAMA_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        top_p=settings.LLM_TOP_P,
        num_predict=settings.LLM_MAX_TOKENS,
        num_ctx=settings.LLM_NUM_CTX,
    )

    ollama_health = ServiceRegistry.ollama_service.health_check()
    logger.info(
        "Ollama: %s (моделей: %d, целевая загружена: %s)",
        ollama_health["status"],
        len(ollama_health.get("models", [])),
        ollama_health.get("current_model_loaded", False),
    )

    # Предупреждение о незагруженной модели
    if ollama_health["status"] == "ok" and not ollama_health.get("current_model_loaded"):
        logger.warning(
            "Целевая модель '%s' не загружена в Ollama. Загрузите командой: ollama pull %s",
            settings.OLLAMA_MODEL,
            settings.OLLAMA_MODEL,
        )

    ServiceRegistry.qdrant_service = QdrantService(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        embedding_model=settings.EMBEDDING_MODEL,
    )

    qdrant_ok = ServiceRegistry.qdrant_service.health_check()
    logger.info("Qdrant: %s", "ok" if qdrant_ok else "unavailable")

    ServiceRegistry.redis_service = RedisSessionService(
        redis_url=settings.REDIS_URL,
        ttl=settings.SESSION_TTL,
        max_history_length=settings.MAX_HISTORY_LENGTH,
    )

    redis_ok = await ServiceRegistry.redis_service.health_check()
    logger.info("Redis: %s", "ok" if redis_ok else "unavailable")

    ServiceRegistry.query_router = QueryRouter()
    logger.info("QueryRouter: загружен (классы: EMERGENCY, LIFESTYLE, OUT_OF_SCOPE, CHITCHAT)")

    logger.info("Конфигурация:")
    logger.info("  - LLM: %s [%s]", settings.OLLAMA_MODEL, settings.OLLAMA_BASE_URL)
    logger.info("  - Embedding: %s", settings.EMBEDDING_MODEL)
    logger.info("  - Reranker: %s", settings.RERANKER_MODEL)
    logger.info("  - Сессии: TTL=%dс, макс. сообщений=%d", settings.SESSION_TTL, settings.MAX_HISTORY_LENGTH)
    logger.info("  - Retrieval: top_k=%d", settings.RETRIEVAL_TOP_K)
    logger.info("  - Rate limit: %d запросов/мин", settings.RATE_LIMIT_REQUESTS)

    logger.info("Приложение готово к работе")
    logger.info("=" * 50)

    yield

    # Shutdown
    logger.info("Завершение работы приложения...")
    logger.info("Приложение остановлено")

app = FastAPI(
    title="МедСовет API",
    description=(
        "Интеллектуальный ассистент для консультирования "
        "по вопросам первой медицинской помощи и здорового образа жизни. "
        "Основан на технологии Retrieval-Augmented Generation (RAG)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/", include_in_schema=False)
async def root():
    """Редирект на документацию Swagger UI."""
    return RedirectResponse(url="/docs")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Обработчик необработанных исключений."""
    logger.error(
        "Необработанное исключение",
        path=str(request.url),
        method=request.method,
        error=str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Внутренняя ошибка сервера",
            error_code="INTERNAL_ERROR",
        ).model_dump(),
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Обработчик ошибок валидации."""
    logger.warning(
        "Ошибка валидации",
        path=str(request.url),
        error=str(exc),
    )

    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            detail=str(exc),
            error_code="VALIDATION_ERROR",
        ).model_dump(),
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )