import os
import sys
import logging
from typing import AsyncIterator, List

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_PROMPTS, build_chat_prompt

logger = logging.getLogger(__name__)


def get_system_prompt(prompt_type: str) -> str:
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["chitchat_prompt"])


class RAGChain:
    def __init__(
        self,
        llm,
        retriever,
        memory: ConversationBufferMemory,
        prompt_type: str = "lifestyle_prompt",
    ):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.prompt_type = prompt_type

        system_prompt = get_system_prompt(prompt_type)
        self.prompt = build_chat_prompt(system_prompt)
        self.chain = self._build_chain()

        self.last_result = None

        logger.info("RAGChain инициализирована: prompt_type=%s", prompt_type)

    def _build_chain(self) -> ConversationalRetrievalChain:
        ret = self.retriever.base_retriever if hasattr(self.retriever, 'base_retriever') else self.retriever

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=ret,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=False,
        )
        return chain

    def update_prompt(self, prompt_type: str) -> None:
        if prompt_type == self.prompt_type:
            return

        self.prompt_type = prompt_type
        system_prompt = get_system_prompt(prompt_type)
        self.prompt = build_chat_prompt(system_prompt)
        self.chain = self._build_chain()

        logger.info("Промпт обновлён: %s", prompt_type)

    def invoke(self, query: str) -> dict:
        self.last_result = self.chain.invoke({"question": query})
        return self.last_result

    async def astream(self, query: str) -> AsyncIterator[str]:
        full_response = ""

        async for chunk in self.chain.astream({"question": query}):
            if "answer" in chunk:
                token = chunk["answer"]
                full_response += token
                yield token

        logger.info("Генерация завершена. Длина ответа: %d символов", len(full_response))

    def get_sources(self) -> List[dict]:
        sources = []

        if self.last_result and "source_documents" in self.last_result:
            for doc in self.last_result["source_documents"]:
                sources.append({
                    "source_title": doc.metadata.get("source", "Неизвестный источник"),
                    "section_header": doc.metadata.get("section_header", ""),
                    "chunk_type": doc.metadata.get("chunk_type", ""),
                    "text_snippet": doc.page_content[:200] if doc.page_content else "",
                })

        return sources

    def reset_memory(self) -> None:
        self.memory.clear()
        logger.info("Память диалога очищена")

    def get_memory_messages(self) -> list:
        return self.memory.chat_memory.messages