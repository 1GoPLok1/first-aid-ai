from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

class SourceCitation(BaseModel):
    """Ссылка на источник, использованный при генерации ответа."""

    source_title: str = Field(
        ...,
        description="Название документа-источника",
        examples=["Памятка МЧС РФ «Ожоги: первая помощь»"],
    )
    page: int = Field(
        ...,
        ge=1,
        description="Номер страницы в документе",
        examples=[12],
    )
    text_snippet: str = Field(
        ...,
        max_length=200,
        description="Фрагмент текста источника (первые 200 символов)",
        examples=["При термическом ожоге необходимо немедленно..."],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_title": "Приказ Минздрава РФ №477н",
                "page": 8,
                "text_snippet": "Первая помощь при кровотечении включает...",
            }
        }
    }

class ChatRequest(BaseModel):
    """Входящий запрос от пользователя."""

    session_id: str = Field(
        ...,
        description="UUID сессии диалога",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Текст запроса пользователя",
        examples=["Что делать при ожоге кипятком?"],
    )

    @field_validator("session_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        """Проверяет, что session_id является корректным UUID."""
        try:
            UUID(v)
        except ValueError:
            raise ValueError("session_id должен быть в формате UUID")
        return v

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Очищает запрос от лишних пробелов."""
        v = v.strip()
        if not v:
            raise ValueError("query не может быть пустым")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "Как правильно наложить жгут при артериальном кровотечении?",
            }
        }
    }

class ChatResponse(BaseModel):
    """Ответ ассистента на запрос пользователя."""

    session_id: str = Field(
        ...,
        description="UUID сессии диалога",
    )
    answer: str = Field(
        ...,
        description="Текст ответа ассистента",
    )
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Список источников, использованных при генерации ответа",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "answer": "При ожоге кипятком необходимо немедленно...",
                "sources": [
                    {
                        "source_title": "Памятка МЧС РФ «Ожоги»",
                        "page": 5,
                        "text_snippet": "При термическом ожоге первой степени...",
                    }
                ],
            }
        }
    }

class SessionResponse(BaseModel):
    """Информация о диалоговой сессии."""

    session_id: str = Field(
        ...,
        description="UUID сессии",
    )
    created_at: datetime = Field(
        ...,
        description="Дата и время создания сессии",
    )
    message_count: int = Field(
        ...,
        ge=0,
        description="Количество сообщений в сессии",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2025-06-04T12:00:00",
                "message_count": 5,
            }
        }
    }

class HealthResponse(BaseModel):
    """Статус доступности компонентов системы."""

    ollama: str = Field(
        ...,
        description="Статус Ollama: 'ok' или 'unavailable'",
        examples=["ok", "unavailable"],
    )
    qdrant: str = Field(
        ...,
        description="Статус Qdrant: 'ok' или 'unavailable'",
        examples=["ok", "unavailable"],
    )
    redis: str = Field(
        ...,
        description="Статус Redis: 'ok' или 'unavailable'",
        examples=["ok", "unavailable"],
    )

    @field_validator("ollama", "qdrant", "redis")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Проверяет допустимые значения статуса."""
        if v not in ("ok", "unavailable"):
            raise ValueError(f"Статус должен быть 'ok' или 'unavailable', получено: '{v}'")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "ollama": "ok",
                "qdrant": "ok",
                "redis": "ok",
            }
        }
    }

class ErrorResponse(BaseModel):
    """Стандартный ответ при возникновении ошибки."""

    detail: str = Field(
        ...,
        description="Описание ошибки",
        examples=["Сессия не найдена"],
    )
    error_code: str = Field(
        ...,
        description="Код ошибки для программной обработки",
        examples=["SESSION_NOT_FOUND", "LLM_TIMEOUT", "INVALID_REQUEST"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Сессия с указанным ID не найдена. Создайте новую сессию.",
                "error_code": "SESSION_NOT_FOUND",
            }
        }
    }