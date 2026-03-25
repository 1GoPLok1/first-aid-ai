import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Параметры подключения к Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

    # Параметры коллекции
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "first_aid_knowledge_base")
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 768))
    DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "COSINE")

    # Модель для эмбеддингов
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")

    # Пути к данным
    DATA_PATH = os.getenv("DATA_PATH", "./data/knowledge_base")


    @classmethod
    def validate(cls):
        """Проверка наличия обязательных переменных"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY не установлен в .env файле")