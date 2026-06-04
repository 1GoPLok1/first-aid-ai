"""
Скрипт семантического чанкинга документов с Parent Document Retriever.
Приложение А.1.3 — Семантический чанкинг с Parent Document Retriever.

Требования:
    pip install llama-index python-dotenv tiktoken
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

PARSED_DIR = Path("./data/parsed")
CHUNKS_OUTPUT_DIR = Path("./data/chunks")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("chunk_documents.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

def import_llamaindex():
    """Проверяет и импортирует компоненты LlamaIndex."""
    libs = {}
    
    try:
        from llama_index.core import Document
        libs["Document"] = Document
        logger.info("✅ LlamaIndex Document загружен")
    except ImportError:
        logger.error("❌ llama-index не установлен. Установите: pip install llama-index")
        sys.exit(1)
    
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        libs["SemanticSplitterNodeParser"] = SemanticSplitterNodeParser
        logger.info("✅ SemanticSplitterNodeParser загружен")
    except ImportError:
        logger.error("❌ SemanticSplitterNodeParser не найден. Установите: pip install llama-index")
        sys.exit(1)
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        libs["HuggingFaceEmbedding"] = HuggingFaceEmbedding
        logger.info("✅ HuggingFaceEmbedding загружен")
    except ImportError:
        logger.warning("⚠️  HuggingFaceEmbedding не загружен. Будет использован SimpleNodeParser.")
        libs["HuggingFaceEmbedding"] = None
    
    try:
        import tiktoken
        libs["tiktoken"] = tiktoken
        logger.info("✅ tiktoken загружен")
    except ImportError:
        logger.warning("⚠️  tiktoken не установлен. Подсчёт токенов будет приблизительным.")
        libs["tiktoken"] = None
    
    return libs

def load_parsed_documents(parsed_dir: Path) -> List[Dict[str, Any]]:
    if not parsed_dir.exists():
        logger.error("❌ Папка '%s' не существует.", parsed_dir.resolve())
        sys.exit(1)
    
    json_files = list(parsed_dir.glob("*_parsed.json"))
    if not json_files:
        logger.error("❌ Нет файлов *_parsed.json в '%s'.", parsed_dir.resolve())
        sys.exit(1)
    
    documents = []
    
    for json_path in json_files:
        logger.info("📂 Загрузка: %s", json_path.name)
        
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Загружаем полный Markdown
        md_path = parsed_dir / f"{json_path.stem.replace('_parsed', '')}.md"
        if not md_path.exists():
            logger.warning("   ⚠️  Markdown-файл не найден: %s", md_path.name)
            continue
        
        with open(md_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        documents.append({
            "filename": metadata["filename"],
            "parsed_filename": metadata.get("parsed_filename", metadata["filename"]),
            "full_text": full_text,
            "headers": metadata.get("headers", []),
            "pages_estimate": metadata.get("pages_estimate", 1),
            "tables_count": metadata.get("tables_count", 0),
            "text_blocks_count": metadata.get("text_blocks_count", 0),
            "source_type": get_source_type(metadata["filename"]),
        })
        
        logger.info("   ✅ Загружено. Символов: %d, Заголовков: %d",
                     len(full_text), len(metadata.get("headers", [])))
    
    logger.info("📊 Всего загружено документов: %d", len(documents))
    return documents


def get_source_type(filename: str) -> str:
    filename_lower = filename.lower()
    
    first_aid_keywords = [
        "мчс", "ожог", "кровотеч", "слр", "реанимац", "травм",
        "перелом", "обморож", "помощь", "477н", "первая",
    ]
    
    for keyword in first_aid_keywords:
        if keyword in filename_lower:
            return "first_aid"
    
    return "healthy_lifestyle"

def count_tokens(text: str, tiktoken_lib=None) -> int:
    if tiktoken_lib is not None:
        try:
            encoding = tiktoken_lib.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    
    # Приблизительный подсчёт: 1 токен ≈ 4 символа для русского языка
    return len(text) // 4

def split_by_h1_sections(markdown_text: str) -> List[Dict[str, Any]]:
    lines = markdown_text.split("\n")
    sections = []
    current_section = {
        "header": "Начало документа",
        "level": 0,
        "content": [],
    }
    
    for line in lines:
        stripped = line.strip()
        
        # Определяем заголовки H1
        if stripped.startswith("# ") and not stripped.startswith("## "):
            # Сохраняем предыдущий раздел
            if current_section["content"]:
                current_section["content"] = "\n".join(current_section["content"])
                sections.append(current_section)
            
            # Начинаем новый раздел
            current_section = {
                "header": stripped.lstrip("#").strip(),
                "level": 1,
                "content": [line],
            }
        else:
            current_section["content"].append(line)
    
    # Последний раздел
    if current_section["content"]:
        current_section["content"] = "\n".join(current_section["content"])
        sections.append(current_section)
    
    # Если H1 не найдены — весь документ как один раздел
    if len(sections) == 1 and sections[0]["level"] == 0:
        sections[0]["content"] = markdown_text
    
    return sections

def generate_chunks(
    document_data: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    libs: dict,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    full_text = document_data["full_text"]
    source_name = document_data["filename"]
    source_type = document_data["source_type"]
    
    # Разбиваем на родительские разделы по H1
    h1_sections = split_by_h1_sections(full_text)
    
    parent_documents = []
    child_chunks = []
    
    for section_idx, section in enumerate(h1_sections):
        section_text = section["content"]
        section_header = section["header"]
        
        if not section_text.strip():
            continue
        
        # Создаём родительский документ
        parent_id = hashlib.md5(
            f"{source_name}_parent_{section_idx}".encode()
        ).hexdigest()[:12]
        
        # Обрезаем родительский документ до ~1500 токенов
        parent_text = section_text
        if count_tokens(parent_text, libs.get("tiktoken")) > 1500:
            # Оставляем первые 1500 токенов + многоточие
            words = parent_text.split()
            truncated = []
            token_count = 0
            for word in words:
                word_tokens = count_tokens(word, libs.get("tiktoken"))
                if token_count + word_tokens > 1500:
                    break
                truncated.append(word)
                token_count += word_tokens
            parent_text = " ".join(truncated) + "\n\n[Текст сокращён до 1500 токенов]"
        
        parent_doc = {
            "chunk_id": parent_id,
            "text": parent_text,
            "parent_id": None,  # Это родитель
            "metadata": {
                "source": source_name,
                "source_type": source_type,
                "section_header": section_header,
                "section_index": section_idx,
                "chunk_type": "parent",
                "token_count": count_tokens(parent_text, libs.get("tiktoken")),
            },
        }
        parent_documents.append(parent_doc)

        if libs["HuggingFaceEmbedding"] is not None and libs.get("SemanticSplitterNodeParser") is not None:
            # Используем семантический сплиттер
            try:
                embed_model = libs["HuggingFaceEmbedding"](
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                splitter = libs["SemanticSplitterNodeParser"](
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    min_chunk_size=min_chunk_size,
                )
                
                # Создаём временный Document для сплиттера
                Document = libs["Document"]
                temp_doc = Document(text=section_text)
                nodes = splitter.get_nodes_from_documents([temp_doc])
                
                for node_idx, node in enumerate(nodes):
                    chunk_text = node.get_content()
                    if not chunk_text.strip():
                        continue
                    
                    chunk_id = hashlib.md5(
                        f"{source_name}_child_{section_idx}_{node_idx}".encode()
                    ).hexdigest()[:12]
                    
                    child_chunk = {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "parent_id": parent_id,
                        "metadata": {
                            "source": source_name,
                            "source_type": source_type,
                            "section_header": section_header,
                            "section_index": section_idx,
                            "chunk_index": node_idx,
                            "chunk_type": "child",
                            "token_count": count_tokens(chunk_text, libs.get("tiktoken")),
                        },
                    }
                    child_chunks.append(child_chunk)
                    
            except Exception as exc:
                logger.warning("   ⚠️  SemanticSplitter не сработал: %s. Использую фиксированный чанкинг.", exc)
                child_chunks.extend(
                    fixed_size_chunking(
                        section_text, source_name, source_type,
                        section_header, section_idx, parent_id, chunk_size, chunk_overlap, min_chunk_size, libs
                    )
                )
        else:
            # Фиксированный чанкинг как запасной вариант
            child_chunks.extend(
                fixed_size_chunking(
                    section_text, source_name, source_type,
                    section_header, section_idx, parent_id, chunk_size, chunk_overlap, min_chunk_size, libs
                )
            )
    
    return child_chunks, parent_documents


def fixed_size_chunking(
    text: str,
    source_name: str,
    source_type: str,
    section_header: str,
    section_idx: int,
    parent_id: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    libs: dict,
) -> List[Dict[str, Any]]:

    # Разбиваем на предложения (грубо)
    sentences = []
    for delimiter in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
        if delimiter in text:
            sentences = text.split(delimiter)
            # Восстанавливаем разделители
            sentences = [s + delimiter.replace("\n", " ") for s in sentences[:-1]] + [sentences[-1]]
            break
    
    if not sentences or len(sentences) == 1:
        sentences = text.split("\n\n")
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_index = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, libs.get("tiktoken"))
        
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Сохраняем текущий чанк
            chunk_text = " ".join(current_chunk)
            if count_tokens(chunk_text, libs.get("tiktoken")) >= min_chunk_size:
                chunk_id = hashlib.md5(
                    f"{source_name}_child_{section_idx}_{chunk_index}".encode()
                ).hexdigest()[:12]
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "parent_id": parent_id,
                    "metadata": {
                        "source": source_name,
                        "source_type": source_type,
                        "section_header": section_header,
                        "section_index": section_idx,
                        "chunk_index": chunk_index,
                        "chunk_type": "child",
                        "token_count": count_tokens(chunk_text, libs.get("tiktoken")),
                    },
                })
                chunk_index += 1
                
            # Начинаем новый чанк с перекрытием
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                s_tokens = count_tokens(s, libs.get("tiktoken"))
                if overlap_tokens + s_tokens > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens
            
            current_chunk = overlap_sentences
            current_tokens = overlap_tokens
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Последний чанк
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if count_tokens(chunk_text, libs.get("tiktoken")) >= min_chunk_size // 2:
            chunk_id = hashlib.md5(
                f"{source_name}_child_{section_idx}_{chunk_index}".encode()
            ).hexdigest()[:12]
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "parent_id": parent_id,
                "metadata": {
                    "source": source_name,
                    "source_type": source_type,
                    "section_header": section_header,
                    "section_index": section_idx,
                    "chunk_index": chunk_index,
                    "chunk_type": "child",
                    "token_count": count_tokens(chunk_text, libs.get("tiktoken")),
                },
            })
    
    return chunks

def main() -> None:
    """Основная функция."""
    logger.info("🚀 Запуск семантического чанкинга документов...")
    logger.info("   Папка с результатами парсинга: %s", PARSED_DIR.resolve())
    logger.info("   Папка для чанков:              %s", CHUNKS_OUTPUT_DIR.resolve())
    
    # Параметры чанкинга
    CHUNK_SIZE = 512       # токенов
    CHUNK_OVERLAP = 50     # токенов (≈10%)
    MIN_CHUNK_SIZE = 100   # токенов
    
    logger.info("   Параметры чанкинга:")
    logger.info("   - chunk_size:      %d токенов", CHUNK_SIZE)
    logger.info("   - chunk_overlap:   %d токенов", CHUNK_OVERLAP)
    logger.info("   - min_chunk_size:  %d токенов", MIN_CHUNK_SIZE)
    logger.info("   - стратегия:       Parent Document Retriever")
    
    # Импорт библиотек
    libs = import_llamaindex()
    
    # Загрузка документов
    documents = load_parsed_documents(PARSED_DIR)
    
    # Создание выходной папки
    CHUNKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Обработка
    all_child_chunks = []
    all_parent_docs = []
    stats_by_source = defaultdict(lambda: {"child": 0, "parent": 0, "total_tokens": 0})
    
    for doc_data in documents:
        logger.info("-" * 50)
        logger.info("📄 Чанкинг: %s", doc_data["filename"])
        
        child_chunks, parent_docs = generate_chunks(
            doc_data,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            MIN_CHUNK_SIZE,
            libs,
        )
        
        all_child_chunks.extend(child_chunks)
        all_parent_docs.extend(parent_docs)
        
        source_key = doc_data["source_type"]
        stats_by_source[source_key]["child"] += len(child_chunks)
        stats_by_source[source_key]["parent"] += len(parent_docs)
        stats_by_source[source_key]["total_tokens"] += sum(
            c["metadata"]["token_count"] for c in child_chunks
        )
        
        logger.info("   📊 Родительских разделов: %d", len(parent_docs))
        logger.info("   📊 Дочерних чанков:      %d", len(child_chunks))
    
    # Сохранение результатов
    all_chunks = all_child_chunks + all_parent_docs
    
    output_path = CHUNKS_OUTPUT_DIR / "all_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    # Отдельно сохраняем маппинг parent-child
    parent_child_map = {}
    for child in all_child_chunks:
        parent_id = child["parent_id"]
        if parent_id not in parent_child_map:
            parent_child_map[parent_id] = []
        parent_child_map[parent_id].append(child["chunk_id"])
    
    map_path = CHUNKS_OUTPUT_DIR / "parent_child_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(parent_child_map, f, ensure_ascii=False, indent=2)
    
    # Итоговая статистика
    child_sizes = [c["metadata"]["token_count"] for c in all_child_chunks]
    avg_child_size = sum(child_sizes) / len(child_sizes) if child_sizes else 0
    
    logger.info("=" * 60)
    logger.info("📊 Итоговая статистика чанкинга:")
    logger.info("   Всего документов:         %d", len(documents))
    logger.info("   Родительских разделов:    %d", len(all_parent_docs))
    logger.info("   Дочерних чанков:          %d", len(all_child_chunks))
    logger.info("   Всего чанков (суммарно):  %d", len(all_chunks))
    logger.info("   Средний размер чанка:     %.0f токенов", avg_child_size)
    logger.info("")
    logger.info("   Распределение по источникам:")
    for source_type, stats in sorted(stats_by_source.items()):
        label = "Первая помощь" if source_type == "first_aid" else "ЗОЖ"
        logger.info("   - %s:", label)
        logger.info("       Дочерних чанков: %d", stats["child"])
        logger.info("       Родительских:    %d", stats["parent"])
        logger.info("       Суммарно токенов: %d", stats["total_tokens"])
    logger.info("")
    logger.info("   💾 Все чанки сохранены в:  %s", output_path.resolve())
    logger.info("   💾 Маппинг сохранён в:     %s", map_path.resolve())
    logger.info("=" * 60)
    
    logger.info("✅ Чанкинг завершён.")


if __name__ == "__main__":
    main()