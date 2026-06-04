import logging
from typing import List, Optional

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, QdrantRetriever
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexConfig,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.url = url
        self.api_key = api_key
        self.embedding_model = embedding_model

        client_kwargs = {"url": url}
        if api_key:
            client_kwargs["api_key"] = api_key

        self.client = QdrantClient(**client_kwargs)
        logger.info("QdrantService инициализирован: %s", url)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"trust_remote_code": True},
        )
        logger.info("Модель эмбеддингов загружена: %s", embedding_model)

    def collection_exists(self, collection_name: str) -> bool:
        """Проверяет существование коллекции."""
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception:
            return False

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: Distance = Distance.COSINE,
    ) -> None:
        if self.collection_exists(collection_name):
            logger.info("Коллекция '%s' уже существует", collection_name)
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
            sparse_vectors_config={"text": SparseVectorParams(index=SparseIndexConfig())},
        )
        logger.info(
            "Коллекция '%s' создана (dense=%d, distance=%s)",
            collection_name,
            vector_size,
            distance,
        )

    def delete_collection(self, collection_name: str) -> None:
        """Удаляет коллекцию."""
        try:
            self.client.delete_collection(collection_name)
            logger.info("Коллекция '%s' удалена", collection_name)
        except Exception as exc:
            logger.warning("Ошибка удаления коллекции '%s': %s", collection_name, exc)

    def get_vector_store(self, collection_name: str) -> QdrantVectorStore:
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name)

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        logger.info("QdrantVectorStore создан для '%s'", collection_name)
        return vector_store

    def get_retriever(self, collection_name: str, top_k: int = 20) -> QdrantRetriever:
        vector_store = self.get_vector_store(collection_name)
        retriever = QdrantRetriever(vectorstore=vector_store, k=top_k)
        logger.info("QdrantRetriever (dense) создан для '%s', k=%d", collection_name, top_k)
        return retriever

    def get_hybrid_retriever(
        self,
        collection_name: str,
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> QdrantRetriever:
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name)

        vector_store = self.get_vector_store(collection_name)

        retriever = QdrantRetriever(
            vectorstore=vector_store,
            k=top_k,
            search_type="hybrid",
            search_kwargs={
                "k": top_k,
                "fetch_k": top_k * 2,
                "score_threshold": None,
                "hybrid_k": rrf_k,
            },
        )
        logger.info(
            "QdrantRetriever (hybrid) создан для '%s', k=%d, rrf_k=%d",
            collection_name,
            top_k,
            rrf_k,
        )
        return retriever

    def search_dense(
        self,
        query: str,
        collection_name: str,
        top_k: int = 20,
    ) -> List[Document]:
        """Выполняет dense (семантический) поиск."""
        retriever = self.get_retriever(collection_name, top_k)
        return retriever.invoke(query)

    def search_hybrid(
        self,
        query: str,
        collection_name: str,
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> List[Document]:
        """Выполняет гибридный поиск (dense + BM25) с RRF-слиянием."""
        retriever = self.get_hybrid_retriever(collection_name, top_k, rrf_k)
        return retriever.invoke(query)

    def get_document_by_chunk_id(
        self,
        collection_name: str,
        chunk_id: str,
    ) -> Optional[Document]:
        try:
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.chunk_id",
                            match=MatchValue(value=chunk_id),
                        )
                    ]
                ),
                limit=1,
            )

            points, _ = results
            if points:
                point = points[0]
                payload = point.payload or {}
                return Document(
                    page_content=payload.get("text", payload.get("page_content", "")),
                    metadata=payload.get("metadata", {}),
                )

            logger.debug("Документ с chunk_id '%s' не найден в '%s'", chunk_id, collection_name)
            return None

        except Exception as exc:
            logger.warning(
                "Ошибка поиска документа по chunk_id '%s' в '%s': %s",
                chunk_id,
                collection_name,
                exc,
            )
            return None

    def add_documents(self, collection_name: str, documents: List[Document]) -> None:
        """Добавляет документы в коллекцию."""
        vector_store = self.get_vector_store(collection_name)
        vector_store.add_documents(documents)
        logger.info("Добавлено %d документов в '%s'", len(documents), collection_name)

    def get_collection_info(self, collection_name: str) -> dict:
        """Возвращает информацию о коллекции."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            }
        except Exception as exc:
            logger.error("Ошибка получения информации о коллекции: %s", exc)
            return {"name": collection_name, "error": str(exc)}

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False