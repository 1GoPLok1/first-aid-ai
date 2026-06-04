from pathlib import Path
from typing import Optional

from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Path:
    """Ищет .env файл в текущей и родительских директориях."""
    current = Path.cwd()
    for _ in range(4):
        env_file = current / ".env"
        if env_file.exists():
            return env_file
        current = current.parent
    return Path(".env")


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из .env файла и переменных окружения."""

    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="URL Qdrant сервера",
    )
    QDRANT_API_KEY: Optional[str] = Field(
        default=None,
        description="API-ключ Qdrant (опционально)",
    )

    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="URL Ollama сервера",
    )
    OLLAMA_MODEL: str = Field(
        default="qwen2.5:7b-instruct-q4_K_M",
        description="Имя модели в Ollama",
    )

    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="URL Redis сервера",
    )

    MINIO_ENDPOINT: str = Field(
        default="localhost:9000",
        description="Endpoint MinIO сервера",
    )
    MINIO_ACCESS_KEY: str = Field(
        default="minioadmin",
        description="Access Key для MinIO",
    )
    MINIO_SECRET_KEY: str = Field(
        default="minioadmin",
        description="Secret Key для MinIO",
    )
    MINIO_BUCKET: str = Field(
        default="medical-docs",
        description="Название бакета с PDF-документами",
    )
    MINIO_SECURE: bool = Field(
        default=False,
        description="Использовать HTTPS для MinIO",
    )

    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Название модели эмбеддингов",
    )

    RERANKER_MODEL: str = Field(
        default="DiTy/cross-encoder-russian-msmarco",
        description="Название модели реранкера",
    )

    MAX_HISTORY_LENGTH: PositiveInt = Field(
        default=10,
        description="Максимальное количество сообщений в истории диалога",
    )
    SESSION_TTL: PositiveInt = Field(
        default=1800,
        description="Время жизни сессии в секундах (30 минут)",
    )

    @property
    def qdrant_kwargs(self) -> dict:
        """Параметры для инициализации QdrantClient."""
        kwargs = {"url": self.QDRANT_URL}
        if self.QDRANT_API_KEY:
            kwargs["api_key"] = self.QDRANT_API_KEY
        return kwargs

    @property
    def ollama_kwargs(self) -> dict:
        """Параметры для инициализации ChatOllama."""
        return {
            "base_url": self.OLLAMA_BASE_URL,
            "model": self.OLLAMA_MODEL,
        }

    @property
    def redis_kwargs(self) -> dict:
        """Параметры для подключения к Redis."""
        return {
            "url": self.REDIS_URL,
            "decode_responses": True,
        }

    @property
    def minio_kwargs(self) -> dict:
        """Параметры для инициализации MinIO клиента."""
        return {
            "endpoint": self.MINIO_ENDPOINT,
            "access_key": self.MINIO_ACCESS_KEY,
            "secret_key": self.MINIO_SECRET_KEY,
            "secure": self.MINIO_SECURE,
        }

settings = Settings()