import logging
from typing import List, Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        qdrant_service: QdrantService,
        collection_name: str,
        llm,
        reranker_model: str = "DiTy/cross-encoder-russian-msmarco",
    ):
        self.qdrant_service = qdrant_service
        self.collection_name = collection_name
        self.llm = llm

        # Базовый гибридный ретривер (dense + BM25)
        self.base_retriever = qdrant_service.get_hybrid_retriever(
            collection_name=collection_name,
            top_k=20,
        )

        # MultiQueryRetriever — генерирует 3 переформулировки запроса
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=llm,
            include_original=True,
        )

        # CrossEncoderReranker — сжимает до top_n
        self.reranker_model_name = reranker_model
        self.reranker = HuggingFaceCrossEncoder(model_name=reranker_model)
        self.compressor = CrossEncoderReranker(model=self.reranker, top_n=3)

        # Итоговый ретривер: MultiQuery -> Hybrid -> RRF -> Rerank
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.multi_query_retriever,
        )

        logger.info(
            "HybridRetriever инициализирован: collection=%s, reranker=%s",
            collection_name,
            reranker_model,
        )

    def _get_parent_document(self, doc: Document) -> Optional[Document]:
        parent_id = doc.metadata.get("parent_id")
        if not parent_id:
            logger.debug("У чанка нет parent_id, пропускаем PDR")
            return None

        parent_doc = self.qdrant_service.get_document_by_chunk_id(
            collection_name=self.collection_name,
            chunk_id=parent_id,
        )

        if parent_doc:
            logger.debug(
                "PDR: замена child '%s' -> parent '%s'",
                doc.metadata.get("chunk_id", "?"),
                parent_id,
            )
        else:
            logger.debug(
                "PDR: родительский документ '%s' не найден в коллекции '%s'",
                parent_id,
                self.collection_name,
            )

        return parent_doc

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        top_m: int = 10,
        top_n: int = 3,
    ) -> List[Document]:
        # Обновляем параметры
        self.base_retriever.k = top_k
        self.compressor.top_n = top_n

        logger.info(
            "Retrieval: query='%s', top_k=%d, top_m=%d, top_n=%d",
            query[:100],
            top_k,
            top_m,
            top_n,
        )

        # MultiQuery -> Hybrid -> RRF -> Rerank
        compressed_docs = self.compression_retriever.invoke(query)

        logger.info("После реранкинга: %d документов", len(compressed_docs))

        # Parent Document Retriever
        final_docs = []
        seen_parents = set()

        for doc in compressed_docs:
            chunk_type = doc.metadata.get("chunk_type", "")

            if chunk_type == "parent":
                # Уже родительский документ
                parent_id = doc.metadata.get("chunk_id", "")
                if parent_id not in seen_parents:
                    final_docs.append(doc)
                    seen_parents.add(parent_id)
                    logger.debug("PDR: родительский чанк уже в выдаче: %s", parent_id)
            else:
                # Дочерний чанк — ищем родительский
                parent_doc = self._get_parent_document(doc)
                if parent_doc:
                    parent_id = parent_doc.metadata.get("chunk_id", "")
                    if parent_id not in seen_parents:
                        final_docs.append(parent_doc)
                        seen_parents.add(parent_id)
                else:
                    # Родитель не найден — оставляем дочерний
                    doc_id = doc.metadata.get("chunk_id", "")
                    if doc_id not in seen_parents:
                        final_docs.append(doc)
                        seen_parents.add(doc_id)
                        logger.debug("PDR: родитель не найден, оставлен child: %s", doc_id)

        logger.info(
            "Retrieval завершён: найдено %d -> после PDR: %d",
            len(compressed_docs),
            len(final_docs),
        )

        return final_docs[:top_n]

    def format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "Информация по данному запросу не найдена в официальных источниках."

        parts = []

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Неизвестный источник")
            section = doc.metadata.get("section_header", "")
            chunk_type = doc.metadata.get("chunk_type", "")

            header = f"Источник {i}: {source}"
            if section:
                header += f" — {section}"
            if chunk_type == "parent":
                header += " [полный раздел]"

            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def retrieve_with_context(
        self,
        query: str,
        top_n: int = 3,
    ) -> tuple:
        docs = self.retrieve(query, top_n=top_n)
        context = self.format_context(docs)
        return docs, context