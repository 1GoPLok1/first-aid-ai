"""
Скрипт векторизации чанков и загрузки в Qdrant.
Приложение А.1.4 — Векторизация и загрузка в Qdrant.

Требования:
    pip install llama-index llama-index-vector-stores-qdrant qdrant-client sentence-transformers python-dotenv
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
CHUNKS_DIR = Path("./data/chunks")
BATCH_SIZE = 50  # Размер батча для загрузки в Qdrant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("index_qdrant.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

def import_dependencies():
    """Проверяет и импортирует все необходимые библиотеки."""
    libs = {}

    # Qdrant client
    try:
        from qdrant_client import QdrantClient
        libs["QdrantClient"] = QdrantClient
        logger.info("✅ qdrant-client загружен")
    except ImportError:
        logger.error("❌ qdrant-client не установлен. Установите: pip install qdrant-client")
        sys.exit(1)

    # LlamaIndex core
    try:
        from llama_index.core import Document, Settings
        from llama_index.core.node_parser import SentenceSplitter
        libs["Document"] = Document
        libs["Settings"] = Settings
        logger.info("✅ llama-index-core загружен")
    except ImportError:
        logger.error("❌ llama-index не установлен. Установите: pip install llama-index")
        sys.exit(1)

    # HuggingFace Embeddings
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        libs["HuggingFaceEmbedding"] = HuggingFaceEmbedding
        logger.info("✅ llama-index-embeddings-huggingface загружен")
    except ImportError:
        logger.error(
            "❌ llama-index-embeddings-huggingface не установлен. "
            "Установите: pip install llama-index-embeddings-huggingface"
        )
        sys.exit(1)

    # Qdrant Vector Store для LlamaIndex
    try:
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        libs["QdrantVectorStore"] = QdrantVectorStore
        logger.info("✅ llama-index-vector-stores-qdrant загружен")
    except ImportError:
        logger.error(
            "❌ llama-index-vector-stores-qdrant не установлен. "
            "Установите: pip install llama-index-vector-stores-qdrant"
        )
        sys.exit(1)

    # sentence-transformers (для проверки доступности модели)
    try:
        import sentence_transformers
        libs["sentence_transformers"] = sentence_transformers
        logger.info("✅ sentence-transformers загружен")
    except ImportError:
        logger.error("❌ sentence-transformers не установлен. Установите: pip install sentence-transformers")
        sys.exit(1)

    return libs

def load_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """
    Загружает все чанки из JSON-файла.

    Args:
        chunks_dir: Папка с результатами чанкинга.

    Returns:
        Список чанков.
    """
    chunks_file = chunks_dir / "all_chunks.json"

    if not chunks_file.exists():
        logger.error("❌ Файл с чанками не найден: %s", chunks_file.resolve())
        logger.error("   Сначала запустите скрипт чанкинга: python scripts/03_chunk_documents.py")
        sys.exit(1)

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("📂 Загружено чанков из файла: %d", len(chunks))

    # Разделяем на child и parent
    child_chunks = [c for c in chunks if c.get("metadata", {}).get("chunk_type") == "child"]
    parent_chunks = [c for c in chunks if c.get("metadata", {}).get("chunk_type") == "parent"]

    logger.info("   - Дочерних (child):  %d", len(child_chunks))
    logger.info("   - Родительских (parent): %d", len(parent_chunks))

    return chunks

def create_qdrant_client(url: str, api_key: str = None) -> "QdrantClient":
    logger.info("🔗 Подключение к Qdrant: %s", url)

    try:
        client_args = {"url": url}
        if api_key:
            client_args["api_key"] = api_key

        client = __import__("qdrant_client").QdrantClient(**client_args)

        # Проверка подключения
        collections = client.get_collections()
        logger.info("✅ Подключение установлено. Существующих коллекций: %d", len(collections.collections))

        return client
    except Exception as exc:
        logger.error("❌ Не удалось подключиться к Qdrant: %s", exc)
        logger.error("   Убедитесь, что Qdrant запущен: docker compose up -d qdrant")
        sys.exit(1)

def create_embedding_model(libs: dict) -> "HuggingFaceEmbedding":
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    logger.info("🧠 Загрузка модели эмбеддингов: %s", model_name)

    try:
        embed_model = libs["HuggingFaceEmbedding"](
            model_name=model_name,
            trust_remote_code=True,
        )
        logger.info("✅ Модель загружена. Размерность: 384")
        return embed_model
    except Exception as exc:
        logger.error("❌ Ошибка загрузки модели: %s", exc)
        logger.error("   Проверьте интернет-соединение и доступность Hugging Face.")
        sys.exit(1)

COLLECTIONS_CONFIG = {
    "first_aid_protocols": {
        "description": "Протоколы первой помощи (МЧС, Минздрав)",
        "source_type": "first_aid",
    },
    "healthy_lifestyle": {
        "description": "Рекомендации по ЗОЖ и профилактике",
        "source_type": "healthy_lifestyle",
    },
}

def create_collections(
    client: "QdrantClient",
    embed_model: "HuggingFaceEmbedding",
    vector_size: int = 384,
) -> Dict[str, "QdrantVectorStore"]:

    from qdrant_client.models import VectorParams, Distance

    vector_stores = {}

    for collection_name, config in COLLECTIONS_CONFIG.items():
        logger.info("-" * 50)
        logger.info("📦 Создание коллекции: %s", collection_name)
        logger.info("   Описание: %s", config["description"])

        # Удаляем коллекцию, если существует (для чистоты эксперимента)
        try:
            client.delete_collection(collection_name)
            logger.info("   🗑️  Старая коллекция удалена.")
        except Exception:
            pass

        # Создаём новую коллекцию
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("   ✅ Коллекция создана (размерность: %d, метрика: Cosine)", vector_size)

        # Создаём QdrantVectorStore для LlamaIndex
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
        )
        vector_stores[collection_name] = vector_store

    return vector_stores

# Импорт QdrantVectorStore глобально для использования в функции
from llama_index.vector_stores.qdrant import QdrantVectorStore

def filter_chunks_by_source(
    chunks: List[Dict[str, Any]],
    source_type: str,
) -> List[Dict[str, Any]]:
    return [
        c for c in chunks
        if c.get("metadata", {}).get("source_type") == source_type
    ]


def chunk_to_llama_document(chunk: Dict[str, Any]) -> "Document":
    from llama_index.core import Document

    metadata = chunk.get("metadata", {})

    # Формируем метаданные для хранения в Qdrant
    doc_metadata = {
        "chunk_id": chunk.get("chunk_id", ""),
        "parent_id": chunk.get("parent_id", ""),
        "source": metadata.get("source", "unknown"),
        "section_header": metadata.get("section_header", ""),
        "chunk_type": metadata.get("chunk_type", "child"),
        "token_count": metadata.get("token_count", 0),
        "source_type": metadata.get("source_type", ""),
    }

    return Document(
        text=chunk.get("text", ""),
        metadata=doc_metadata,
        doc_id=chunk.get("chunk_id", ""),
    )

def index_chunks(
    chunks: List[Dict[str, Any]],
    vector_stores: Dict[str, "QdrantVectorStore"],
    embed_model: "HuggingFaceEmbedding",
    batch_size: int = 50,
) -> Dict[str, int]:
    from llama_index.core import VectorStoreIndex, StorageContext

    stats = {}

    for collection_name, config in COLLECTIONS_CONFIG.items():
        source_type = config["source_type"]
        logger.info("-" * 50)
        logger.info("📤 Загрузка в коллекцию: %s", collection_name)

        # Фильтруем чанки
        collection_chunks = filter_chunks_by_source(chunks, source_type)
        logger.info("   Чанков для загрузки: %d", len(collection_chunks))

        if not collection_chunks:
            logger.warning("   ⚠️  Нет чанков для этого типа источника!")
            stats[collection_name] = 0
            continue

        # Преобразуем в LlamaIndex Documents
        documents = [chunk_to_llama_document(c) for c in collection_chunks]

        # Разбиваем на батчи
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(documents))
            batch = documents[start_idx:end_idx]

            logger.info(
                "   🔄 Батч %d/%d (%d чанков)...",
                batch_idx + 1, total_batches, len(batch),
            )

            # Создаём индекс для батча
            vector_store = vector_stores[collection_name]
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            try:
                VectorStoreIndex.from_documents(
                    batch,
                    storage_context=storage_context,
                    embed_model=embed_model,
                    show_progress=False,
                )
            except Exception as exc:
                logger.error("   ❌ Ошибка загрузки батча: %s", exc)
                raise

        stats[collection_name] = len(documents)
        logger.info("   ✅ Загружено: %d чанков", len(documents))

    return stats

def print_collection_stats(
    client: "QdrantClient",
    stats: Dict[str, int],
) -> None:
    """
    Выводит статистику по коллекциям Qdrant.

    Args:
        client: Клиент Qdrant.
        stats: Словарь с количеством загруженных чанков.
    """
    logger.info("=" * 60)
    logger.info("📊 Статистика загрузки в Qdrant:")
    logger.info("")

    total_vectors = 0
    total_size_bytes = 0

    for collection_name in COLLECTIONS_CONFIG:
        try:
            collection_info = client.get_collection(collection_name)
            vectors_count = collection_info.vectors_count
            # Оцениваем размер: 384 float32 = 1536 байт на вектор
            estimated_size_mb = (vectors_count * 1536) / (1024 * 1024)

            logger.info("   📦 Коллекция: %s", collection_name)
            logger.info("      - Векторов:     %d", vectors_count)
            logger.info("      - Примерный размер: %.2f МБ", estimated_size_mb)
            logger.info("      - Статус:       %s", collection_info.status)

            total_vectors += vectors_count
            total_size_bytes += vectors_count * 1536
        except Exception as exc:
            logger.warning("   ⚠️  Не удалось получить информацию о коллекции '%s': %s", collection_name, exc)

    total_size_mb = total_size_bytes / (1024 * 1024)
    logger.info("")
    logger.info("   📊 Общая статистика:")
    logger.info("      - Всего векторов:   %d", total_vectors)
    logger.info("      - Общий размер:     %.2f МБ", total_size_mb)
    logger.info("      - Размерность:      384")
    logger.info("      - Метрика:          Cosine")
    logger.info("=" * 60)

def main() -> None:
    """Основная функция."""
    logger.info("🚀 Запуск векторизации и индексации в Qdrant...")
    logger.info("   Qdrant URL:     %s", QDRANT_URL)
    logger.info("   Папка с чанками: %s", CHUNKS_DIR.resolve())
    logger.info("   Модель:         sentence-transformers/all-MiniLM-L6-v2")
    logger.info("   Размерность:    384")
    logger.info("   Метрика:        Cosine")

    # Импорт зависимостей
    libs = import_dependencies()

    # Загрузка чанков
    chunks = load_chunks(CHUNKS_DIR)

    # Подключение к Qdrant
    client = create_qdrant_client(QDRANT_URL, QDRANT_API_KEY)

    # Загрузка модели эмбеддингов
    embed_model = create_embedding_model(libs)

    # Создание коллекций
    vector_stores = create_collections(client, embed_model, vector_size=384)

    # Индексация
    start_time = time.time()
    stats = index_chunks(chunks, vector_stores, embed_model, batch_size=BATCH_SIZE)
    elapsed_time = time.time() - start_time

    # Статистика
    print_collection_stats(client, stats)

    logger.info("⏱️  Время индексации: %.1f сек", elapsed_time)
    logger.info("✅ Векторизация и загрузка в Qdrant завершены.")

if __name__ == "__main__":
    main()