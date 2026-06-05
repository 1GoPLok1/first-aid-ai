import logging
from typing import List, Optional

import httpx
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger(__name__)


class OllamaService:
    """Сервис Ollama"""
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:7b-instruct-q4_K_M",
        temperature: float = 0.1,
        top_p: float = 0.3,
        num_predict: int = 1024,
        num_ctx: int = 4096,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.num_ctx = num_ctx

        logger.info(
            "OllamaService инициализирован: %s [модель: %s]",
            self.base_url,
            self.model_name,
        )

    def get_llm(self, streaming: bool = True) -> ChatOllama:
        llm = ChatOllama(
            base_url=self.base_url,
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            num_predict=self.num_predict,
            num_ctx=self.num_ctx,
            streaming=streaming,
        )
        logger.info(
            "ChatOllama создан: model=%s, temperature=%.1f, streaming=%s",
            self.model_name,
            self.temperature,
            streaming,
        )
        return llm

    def get_llm_sync(self) -> ChatOllama:
        return self.get_llm(streaming=False)

    def health_check(self) -> dict:
        try:
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=5.0,
            )
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            is_loaded = self.model_name in models

            logger.info(
                "Ollama доступен. Моделей: %d. Текущая '%s': %s",
                len(models),
                self.model_name,
                "загружена" if is_loaded else "НЕ загружена",
            )

            return {
                "status": "ok",
                "models": models,
                "current_model_loaded": is_loaded,
            }

        except httpx.ConnectError:
            logger.error("Ollama недоступен: %s", self.base_url)
            return {
                "status": "unavailable",
                "models": [],
                "current_model_loaded": False,
                "error": f"Не удалось подключиться к {self.base_url}",
            }

        except httpx.TimeoutException:
            logger.error("Ollama недоступен: тайм-аут")
            return {
                "status": "unavailable",
                "models": [],
                "current_model_loaded": False,
                "error": "Тайм-аут подключения к Ollama",
            }

        except Exception as exc:
            logger.error("Ошибка проверки Ollama: %s", exc)
            return {
                "status": "unavailable",
                "models": [],
                "current_model_loaded": False,
                "error": str(exc),
            }

    def list_models(self) -> List[str]:
        """Возвращает список загруженных моделей."""
        health = self.health_check()
        return health.get("models", [])

    def is_model_loaded(self) -> bool:
        """Проверяет, загружена ли целевая модель."""
        health = self.health_check()
        return health.get("current_model_loaded", False)