import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import uuid4

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisSessionService:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl: int = 1800,
        max_history_length: int = 10,
    ):
        self.redis_url = redis_url
        self.ttl = ttl
        self.max_history_length = max_history_length
        self.client: Optional[redis.Redis] = None

        logger.info(
            "RedisSessionService инициализирован: %s (TTL=%dс, макс. сообщений=%d)",
            redis_url,
            ttl,
            max_history_length,
        )

    async def _get_client(self) -> redis.Redis:
        """Возвращает клиент Redis, создавая его при первом обращении."""
        if self.client is None:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            logger.info("Подключение к Redis установлено: %s", self.redis_url)
        return self.client

    async def create_session(self) -> str:
        client = await self._get_client()
        session_id = str(uuid4())

        initial_data = {
            "messages": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await client.setex(
            name=f"session:{session_id}",
            time=self.ttl,
            value=json.dumps(initial_data, ensure_ascii=False),
        )

        logger.info("Сессия создана: %s", session_id)
        return session_id

    async def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        client = await self._get_client()
        data = await client.get(f"session:{session_id}")

        if data is None:
            logger.warning("Сессия не найдена: %s", session_id)
            return []

        session = json.loads(data)
        return session.get("messages", [])

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        client = await self._get_client()
        data = await client.get(f"session:{session_id}")

        if data is None:
            logger.warning("Сессия не найдена: %s", session_id)
            return False

        session = json.loads(data)
        messages = session.get("messages", [])

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if sources is not None:
            message["sources"] = sources

        messages.append(message)

        if len(messages) > self.max_history_length:
            messages = messages[-self.max_history_length:]

        session["messages"] = messages
        session["updated_at"] = datetime.now(timezone.utc).isoformat()

        await client.setex(
            name=f"session:{session_id}",
            time=self.ttl,
            value=json.dumps(session, ensure_ascii=False),
        )

        logger.info(
            "Сообщение добавлено в сессию %s: role=%s, длина=%d символов",
            session_id,
            role,
            len(content),
        )
        return True

    async def delete_session(self, session_id: str) -> bool:
        client = await self._get_client()
        result = await client.delete(f"session:{session_id}")

        if result:
            logger.info("Сессия удалена: %s", session_id)
            return True
        else:
            logger.warning("Сессия не найдена для удаления: %s", session_id)
            return False

    async def session_exists(self, session_id: str) -> bool:
        client = await self._get_client()
        exists = await client.exists(f"session:{session_id}")
        return bool(exists)

    async def refresh_ttl(self, session_id: str) -> bool:
        """Обновляет TTL сессии."""
        client = await self._get_client()
        result = await client.expire(f"session:{session_id}", self.ttl)
        return bool(result)

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as exc:
            logger.error("Redis недоступен: %s", exc)
            return False