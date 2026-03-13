import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Vector Store
    QDRANT_DB_PATH = os.getenv("QDRANT_DB_PATH", "./data/embeddings/qdrant_db")
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))


    @classmethod
    def validate(cls):
        """Проверка наличия обязательных переменных"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY не установлен в .env файле")