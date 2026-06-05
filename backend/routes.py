import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from typing import AsyncIterator
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain.memory import ConversationBufferMemory
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from schemas import (
    ChatRequest,
    SessionResponse,
    HealthResponse,
    ErrorResponse,
)
from prompts import OUT_OF_SCOPE_RESPONSE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")
limiter = Limiter(key_func=get_remote_address)

class ServiceRegistry:
    """Реестр сервисов приложения. Заполняется в main.py при старте."""
    ollama_service = None
    qdrant_service = None
    redis_service = None
    query_router = None

def get_ollama_service():
    """Возвращает OllamaService из реестра."""
    return ServiceRegistry.ollama_service

def get_qdrant_service():
    """Возвращает QdrantService из реестра."""
    return ServiceRegistry.qdrant_service

def get_redis_service():
    """Возвращает RedisSessionService из реестра."""
    return ServiceRegistry.redis_service

def get_query_router():
    """Возвращает QueryRouter из реестра."""
    return ServiceRegistry.query_router

def get_llm():
    service = ServiceRegistry.ollama_service
    return service.get_llm(streaming=True)

def get_llm_sync():
    service = ServiceRegistry.ollama_service
    return service.get_llm_sync()

def build_rag_chain(
    collection_name: str,
    prompt_type: str,
    qdrant_service,
    llm,
    llm_sync,
):
    from core.retriever import HybridRetriever
    from core.chain import RAGChain

    retriever = HybridRetriever(
        qdrant_service=qdrant_service,
        collection_name=collection_name,
        llm=llm_sync,
        reranker_model=settings.RERANKER_MODEL,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return RAGChain(
        llm=llm,
        retriever=retriever,
        memory=memory,
        prompt_type=prompt_type,
    )

@router.post(
    "/chat/stream",
    summary="Отправка сообщения с потоковым ответом",
)
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    redis_service=Depends(get_redis_service),
    qdrant_service=Depends(get_qdrant_service),
    query_router=Depends(get_query_router),
    llm=Depends(get_llm),
    llm_sync=Depends(get_llm_sync),
):
    session_id = body.session_id
    query = body.query

    # Проверка/создание сессии
    if not await redis_service.session_exists(session_id):
        session_id = await redis_service.create_session()

    # Сохраняем сообщение пользователя
    await redis_service.add_message(
        session_id=session_id,
        role="user",
        content=query,
    )

    # Классификация запроса
    classification = query_router.process_query(query)
    label = classification["label"]

    # Запрос вне компетенции — возвращаем заглушку
    if label == "OUT_OF_SCOPE":
        async def static_generator() -> AsyncIterator[str]:
            response_text = OUT_OF_SCOPE_RESPONSE
            await redis_service.add_message(
                session_id=session_id,
                role="assistant",
                content=response_text,
            )
            yield f"data: {json.dumps({'token': response_text})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            static_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": session_id,
            },
        )

    # Создаём RAG-цепочку
    collection_name = classification["collection"]
    prompt_type = classification["prompt"]
    rag_chain = build_rag_chain(
        collection_name=collection_name,
        prompt_type=prompt_type,
        qdrant_service=qdrant_service,
        llm=llm,
        llm_sync=llm_sync,
    )

    # Загружаем историю диалога из Redis в память цепочки
    history = await redis_service.get_session(session_id)
    for msg in history:
        if msg["role"] == "user":
            rag_chain.memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            rag_chain.memory.chat_memory.add_ai_message(msg["content"])

    # Дисклеймер для экстренных запросов
    disclaimer = classification.get("disclaimer")

    async def generate() -> AsyncIterator[str]:
        full_response = ""

        # Отправляем дисклеймер первым токеном
        if disclaimer:
            full_response = disclaimer + "\n\n"
            yield f"data: {json.dumps({'token': disclaimer + '\n\n'})}\n\n"

        try:
            async for token in rag_chain.astream(query):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Сохраняем полный ответ в Redis
            await redis_service.add_message(
                session_id=session_id,
                role="assistant",
                content=full_response,
            )

        except Exception as exc:
            logger.error("Ошибка генерации ответа: %s", exc)
            error_msg = "Произошла ошибка при генерации ответа. Попробуйте позже."
            yield f"data: {json.dumps({'token': error_msg, 'error': True})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )

@router.get(
    "/sessions",
    summary="Получение списка сессий",
)
async def list_sessions(redis_service=Depends(get_redis_service)):
    sessions = await redis_service.list_sessions()
    return sessions

@router.post(
    "/sessions",
    response_model=SessionResponse,
    summary="Создание новой сессии",
)
async def create_session(
    redis_service=Depends(get_redis_service),
):
    """Создаёт новую диалоговую сессию и возвращает её ID."""
    session_id = await redis_service.create_session()
    return SessionResponse(
        session_id=session_id,
        created_at="2025-01-01T00:00:00",
        message_count=0,
    )

@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Получение информации о сессии",
)
async def get_session(
    session_id: str,
    redis_service=Depends(get_redis_service),
):
    """Возвращает количество сообщений в сессии."""
    if not await redis_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    messages = await redis_service.get_session(session_id)
    return SessionResponse(
        session_id=session_id,
        created_at="2025-01-01T00:00:00",
        message_count=len(messages),
    )

@router.delete(
    "/sessions/{session_id}",
    summary="Удаление сессии",
)
async def delete_session(
    session_id: str,
    redis_service=Depends(get_redis_service),
):
    """Удаляет диалоговую сессию и её историю."""
    deleted = await redis_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"detail": "Сессия удалена"}

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка доступности сервисов",
)
async def health_check(
    ollama_service=Depends(get_ollama_service),
    qdrant_service=Depends(get_qdrant_service),
    redis_service=Depends(get_redis_service),
):
    """Проверяет доступность Ollama, Qdrant и Redis."""
    ollama_health = ollama_service.health_check()
    qdrant_ok = qdrant_service.health_check()
    redis_ok = await redis_service.health_check()

    return HealthResponse(
        ollama="ok" if ollama_health["status"] == "ok" else "unavailable",
        qdrant="ok" if qdrant_ok else "unavailable",
        redis="ok" if redis_ok else "unavailable",
    )