import re
import json
import hashlib
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum

# NLTK для токенизации
import nltk
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize

# === Надёжная инициализация NLTK ===
def _ensure_nltk_resources():
    """Гарантированная загрузка необходимых NLTK-ресурсов для русского языка"""
    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),  # Новая версия для NLTK 3.9+
        ('tokenizers/punkt', 'punkt'),           # Старая версия для совместимости
    ]
    
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
            return  # Уже установлено
        except LookupError:
            continue
    
    # Пытаемся скачать (может потребовать интернет)
    try:
        nltk.download('punkt_tab', quiet=True, raise_on_error=False)
    except:
        try:
            nltk.download('punkt', quiet=True, raise_on_error=False)
        except:
            logging.warning("⚠️ Не удалось загрузить NLTK-модели. Будет использован fallback-токенизатор.")

# Вызываем при импорте модуля
_ensure_nltk_resources()


# === Fallback-токенизатор для русского языка (если NLTK не работает) ===
def _regex_sent_tokenize(text: str) -> List[str]:
    """
    Простой токенизатор на основе регулярных выражений.
    Используется как запасной вариант, если NLTK недоступен.
    """
    # Разбиваем по концам предложений, сохраняя разделители
    # Учитываем русские окончания: . ! ? ... !? ?!
    pattern = r'([^.!?]+[.!?]+(?:\s+|(?=\s*[А-Я]|$)))'
    parts = re.findall(pattern, text, flags=re.UNICODE)
    
    # Если паттерн не сработал — разбиваем по точкам
    if not parts:
        parts = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    # Очистка и фильтрация
    sentences = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]
    
    return sentences if sentences else [text]


def safe_sent_tokenize(text: str, language: str = 'russian') -> List[str]:
    """
    Безопасная токенизация предложений с авто-фолбэком.
    
    Args:
        text: Исходный текст
        language: Язык ('russian' или 'english')
    
    Returns:
        List[str]: Список предложений
    """
    try:
        # Пробуем NLTK с явным указанием языка
        return nltk_sent_tokenize(text, language=language)
    except (LookupError, NotImplementedError, ValueError) as e:
        logging.debug(f"NLTK токенизация не удалась ({e}), используем fallback")
        return _regex_sent_tokenize(text)
    except Exception as e:
        logging.warning(f"Неожиданная ошибка токенизации: {e}, используем fallback")
        return _regex_sent_tokenize(text)


# ==================== DATA CLASSES ====================

class UrgencyLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ChunkType(Enum):
    ALGORITHM = "algorithm"
    WARNING = "warning"
    DOSAGE = "dosage"
    SYMPTOM = "symptom"
    PREVENTIVE = "preventive"
    GENERAL = "general"


@dataclass
class RAGChunk:
    chunk_id: str
    doc_id: str
    text: str
    text_hash: str
    metadata: Dict
    score: float = 1.0
    schema_version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['metadata']['urgency'] = data['metadata'].get('urgency', 'low')
        data['metadata']['chunk_type'] = data['metadata'].get('chunk_type', 'general')
        return data
    
    def to_embedding_input(self) -> Dict:
        prefix = ""
        chunk_type = self.metadata.get('chunk_type', 'general')
        if chunk_type == 'algorithm':
            prefix = "[Инструкция] "
        elif chunk_type == 'warning':
            prefix = "[Важно] "
        elif chunk_type == 'dosage':
            prefix = "[Дозировка] "
        return {
            'id': self.chunk_id,
            'text': prefix + self.text,
            'metadata': self.metadata
        }


# ==================== CHUNKER CLASS ====================

class AdaptiveSemanticChunker:
    """
    Адаптивный чанкер для медицинских текстов в RAG-системах.
    Использует безопасную токенизацию с поддержкой русского языка.
    """
    
    SCHEMA_VERSION = "1.0"
    URGENCY_THRESHOLDS = {'critical': 5, 'high': 3, 'medium': 1}
    MIN_CHUNK_CHARS = 100
    MAX_CHUNK_CHARS = 2000

    def __init__(self,
                 chunk_size: int = 800,
                 chunk_overlap: int = 150,
                 min_chunk_chars: int = MIN_CHUNK_CHARS,
                 respect_headings: bool = True,
                 preserve_algorithms: bool = True,
                 doc_id_prefix: str = "chunk",
                 language: str = 'russian',
                 log_level: int = logging.WARNING):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.respect_headings = respect_headings
        self.preserve_algorithms = preserve_algorithms
        self.doc_id_prefix = doc_id_prefix
        self.language = language  # 'russian' или 'english'
        
        self._compile_patterns()
        
        # Словари для классификации
        self.urgency_keywords = {
            UrgencyLevel.CRITICAL: {
                'остановка сердца', 'клиническая смерть', 'асфиксия', 'анафилактический',
                'массивное кровотечение', 'артериальное кровотечение', 'потеря сознания',
                'реанимация', 'немедленно', 'угроза жизни', '103', '112'
            },
            UrgencyLevel.HIGH: {
                'сильная боль', 'высокая температура', 'одышка', 'травма', 'перелом',
                'ожог', 'аллергическая реакция', 'срочно', 'неотложная помощь'
            },
            UrgencyLevel.MEDIUM: {
                'рекомендуется', 'желательно', 'профилактика', 'контроль', 'обследование'
            },
        }
        
        self.chunk_type_keywords = {
            ChunkType.ALGORITHM: {'алгоритм', 'порядок действий', 'последовательность', 'шаг', 'этап'},
            ChunkType.WARNING: {'противопоказание', 'предупреждение', 'осторожно', 'запрещено'},
            ChunkType.DOSAGE: {'дозировка', 'доза', 'принимать', 'мг', 'мл', 'таб', 'кап'},
            ChunkType.SYMPTOM: {'симптом', 'признак', 'проявление', 'жалоба'},
            ChunkType.PREVENTIVE: {'профилактика', 'рекомендация', 'совет', 'ЗОЖ'},
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self._processed_count = 0

    def _compile_patterns(self):
        self.patterns = {
            'heading': re.compile(
                r'^\s*(#{1,3}\s+|\d+[\.\)]\s+|⚠️?\s*|[А-Я]{3,}\s*)[А-Я][^\n]+', re.M
            ),
            'algorithm_step': re.compile(r'^\s*(\d+[\.\)]|Шаг\s*\d+|Этап\s*\d+)\s+', re.M),
            'bullet': re.compile(r'^\s*[•\-*◦▪]\s+', re.M),
            'context_break': re.compile(
                r'(противопоказан|не рекомендуется|запрещен|при наличии|в случае|если|'
                r'следует|необходимо|обязательно|алгоритм|последовательность)', re.I
            ),
            'dosage': re.compile(r'(\d+[.,]?\d*)\s*([мк]г|мл|таб|кап|%)', re.I),
            'vital_signs': re.compile(r'(\d+[/\s]\d+)\s*мм\s*рт\.?\s*ст\.?', re.I),
            'temperature': re.compile(r'(\d+[.,]\d+)\s*°?\s*C', re.I),
        }

    # ==================== ОСНОВНОЙ МЕТОД ====================
    
    def create_rag_chunks(self,
                         text: str,
                         doc_id: Optional[str] = None,
                         source_meta: Optional[Dict] = None) -> List[RAGChunk]:
        if not text or len(text.strip()) < self.min_chunk_chars:
            return []
        
        doc_id = doc_id or f"{self.doc_id_prefix}_{self._processed_count}"
        sections = self._split_by_headings(text) if self.respect_headings else [text]
        
        all_chunks: List[RAGChunk] = []
        
        for section_idx, section in enumerate(sections):
            section_chunks = self._chunk_section(section, doc_id, section_idx, source_meta or {})
            all_chunks.extend(section_chunks)
        
        if self.chunk_overlap > 0 and len(all_chunks) > 1:
            all_chunks = self._add_smart_overlap(all_chunks)
        
        self._processed_count += len(all_chunks)
        self.logger.debug(f"Создано {len(all_chunks)} чанков для {doc_id}")
        
        return all_chunks

    def _chunk_section(self,
                      section: str,
                      doc_id: str,
                      section_idx: int,
                      source_meta: Dict) -> List[RAGChunk]:
        # ✅ ИСПОЛЬЗУЕМ БЕЗОПАСНУЮ ТОКЕНИЗАЦИЮ
        sentences = safe_sent_tokenize(section, language=self.language)
        chunks: List[RAGChunk] = []
        
        current_sentences: List[str] = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sent_length = len(sentence)
            
            if sent_length > self.chunk_size:
                if current_sentences:
                    chunks.append(self._finalize_chunk(
                        current_sentences, doc_id, section_idx, len(chunks), source_meta
                    ))
                    current_sentences = []
                    current_length = 0
                
                words = sentence.split()
                temp_sentences: List[str] = []
                temp_length = 0
                
                for word in words:
                    if temp_length + len(word) + 1 > self.chunk_size and temp_sentences:
                        chunks.append(self._finalize_chunk(
                            temp_sentences, doc_id, section_idx, len(chunks), source_meta,
                            note="forced_split"
                        ))
                        temp_sentences = [word]
                        temp_length = len(word)
                    else:
                        temp_sentences.append(word)
                        temp_length += len(word) + 1
                
                if temp_sentences:
                    current_sentences = temp_sentences
                    current_length = temp_length
                continue
            
            if current_length + sent_length + 1 > self.chunk_size:
                can_split = (i >= len(sentences) - 1 or 
                           self._is_safe_boundary(sentence, sentences[i + 1]))
                
                if can_split:
                    chunks.append(self._finalize_chunk(
                        current_sentences, doc_id, section_idx, len(chunks), source_meta
                    ))
                    current_sentences = [sentence]
                    current_length = sent_length
                else:
                    current_sentences.append(sentence)
                    current_length += sent_length + 1
            else:
                current_sentences.append(sentence)
                current_length += sent_length + 1
        
        if current_sentences:
            chunks.append(self._finalize_chunk(
                current_sentences, doc_id, section_idx, len(chunks), source_meta
            ))
        
        return chunks

    def _finalize_chunk(self,
                       sentences: List[str],
                       doc_id: str,
                       section_idx: int,
                       chunk_idx: int,
                       source_meta: Dict,
                       note: Optional[str] = None) -> RAGChunk:
        text = ' '.join(sentences).strip()
        if len(text) < self.min_chunk_chars and len(sentences) > 1:
            text = text + ' ' + ' '.join(sentences[:1])
        
        chunk_id = f"{doc_id}_{section_idx}_{chunk_idx}"
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        
        metadata = {
            'content_type': self._detect_chunk_type(text).value,
            'urgency': self._detect_urgency(text).value,
            'section_index': section_idx,
            'sentence_count': len(sentences),
            'char_count': len(text),
            'entities': self._extract_entities(text),
            'key_phrases': self._extract_key_phrases(text),
            'section_title': self._extract_section_title(' '.join(sentences[:3])),
            **source_meta
        }
        if note:
            metadata['note'] = note
        
        score = self._calculate_priority_score(metadata)
        
        return RAGChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            text_hash=text_hash,
            metadata=metadata,
            score=score,
            schema_version=self.SCHEMA_VERSION
        )

    # ==================== СМАРТ-ПЕРЕКРЫТИЕ ====================
    
    def _add_smart_overlap(self, chunks: List[RAGChunk], overlap_chars: int = None) -> List[RAGChunk]:
        if len(chunks) <= 1:
            return chunks
        
        overlap_chars = overlap_chars or self.chunk_overlap
        result = []
        
        for i, chunk in enumerate(chunks):
            new_text = chunk.text
            
            if i > 0 and overlap_chars > 0:
                prev_text = chunks[i-1].text
                overlap = self._extract_overlap_suffix(prev_text, overlap_chars)
                if overlap:
                    new_text = overlap + ' ' + new_text
            
            if i < len(chunks) - 1 and overlap_chars > 0:
                next_text = chunks[i+1].text
                overlap = self._extract_overlap_prefix(next_text, overlap_chars)
                if overlap:
                    new_text = new_text + ' ' + overlap
            
            updated_chunk = RAGChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=new_text.strip(),
                text_hash=hashlib.sha256(new_text.encode('utf-8')).hexdigest()[:16],
                metadata={**chunk.metadata, 'char_count': len(new_text), 'has_overlap': True},
                score=chunk.score,
                schema_version=chunk.schema_version
            )
            result.append(updated_chunk)
        
        return result

    def _extract_overlap_suffix(self, text: str, max_chars: int) -> Optional[str]:
        if len(text) <= max_chars:
            return text
        suffix = text[-max_chars:]
        for sep in ['. ', '! ', '? ']:
            idx = suffix.find(sep)
            if idx != -1:
                return suffix[idx + 2:].strip()
        return suffix.strip()

    def _extract_overlap_prefix(self, text: str, max_chars: int) -> Optional[str]:
        if len(text) <= max_chars:
            return text
        prefix = text[:max_chars]
        for sep in ['. ', '! ', '? ']:
            idx = prefix.rfind(sep)
            if idx != -1:
                return prefix[:idx + 1].strip()
        return prefix.strip()

    # ==================== КЛАССИФИКАЦИЯ ====================
    
    def _detect_urgency(self, text: str) -> UrgencyLevel:
        text_lower = text.lower()
        for level, keywords in self.urgency_keywords.items():
            if sum(1 for kw in keywords if kw in text_lower) >= self.URGENCY_THRESHOLDS.get(level, 1):
                return level
        return UrgencyLevel.LOW

    def _detect_chunk_type(self, text: str) -> ChunkType:
        text_lower = text.lower()
        scores = {ct: sum(1 for kw in kws if kw in text_lower) 
                 for ct, kws in self.chunk_type_keywords.items()}
        if not scores or max(scores.values()) == 0:
            return ChunkType.GENERAL
        best = max(scores, key=scores.get)
        return best if scores[best] >= 2 else ChunkType.GENERAL

    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        for match in self.patterns['dosage'].finditer(text):
            entities.append(f"dosage:{match.group(0)}")
        for match in self.patterns['vital_signs'].finditer(text):
            entities.append(f"vital:{match.group(0)}")
        for match in self.patterns['temperature'].finditer(text):
            entities.append(f"temperature:{match.group(0)}")
        medical_terms = ['СЛР', 'ИВЛ', 'жгут', 'повязка', 'антисептик', 'адреналин']
        for term in medical_terms:
            if term.lower() in text.lower():
                entities.append(f"term:{term}")
        return list(set(entities))

    def _extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        stop_words = {'и', 'в', 'не', 'на', 'с', 'по', 'для', 'от', 'при', 'как', 'что', 'это', 'а', 'но'}
        words = re.findall(r'[а-яёa-z]{4,}', text.lower(), re.I)
        freq = {}
        for w in words:
            if w not in stop_words:
                freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]

    def _extract_section_title(self, text: str) -> Optional[str]:
        match = self.patterns['heading'].match(text)
        if match:
            return match.group(0).strip().lstrip('#').strip()
        return None

    def _calculate_priority_score(self, metadata: Dict) -> float:
        score = 1.0
        if metadata['urgency'] == 'critical':
            score *= 2.0
        elif metadata['urgency'] == 'high':
            score *= 1.5
        if metadata['content_type'] in ['algorithm', 'warning']:
            score *= 1.3
        if metadata['char_count'] < 200:
            score *= 0.8
        return min(score, 3.0)

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def _split_by_headings(self, text: str) -> List[str]:
        lines = text.split('\n')
        sections = []
        current = []
        for line in lines:
            if self.patterns['heading'].match(line.strip()) and current:
                section = '\n'.join(current).strip()
                if section:
                    sections.append(section)
                current = [line]
            else:
                current.append(line)
        if current:
            section = '\n'.join(current).strip()
            if section:
                sections.append(section)
        return sections if sections else [text]

    def _is_safe_boundary(self, sentence: str, next_sentence: str) -> bool:
        if re.search(r'^Однако|^Поэтому|^Следовательно|^Кроме того|^Также|^При этом', 
                    next_sentence, re.I):
            return False
        if self.patterns['context_break'].search(sentence) and len(next_sentence) < 150:
            return False
        if re.search(r':\s*$', sentence) or next_sentence.lstrip().startswith(('- ', '• ', '* ', '1.', '1)')):
            return False
        if self.preserve_algorithms:
            if self.patterns['algorithm_step'].match(sentence) or self.patterns['algorithm_step'].match(next_sentence):
                return False
        return True

    # ==================== ФИЛЬТРАЦИЯ И ЭКСПОРТ ====================
    
    def filter_by_urgency(self, chunks: List[RAGChunk], min_urgency: Union[UrgencyLevel, str]) -> List[RAGChunk]:
        target = min_urgency.value if isinstance(min_urgency, UrgencyLevel) else min_urgency
        urgency_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        min_level = urgency_order.get(target, 0)
        return [c for c in chunks if urgency_order.get(c.metadata['urgency'], 0) >= min_level]

    def filter_by_type(self, chunks: List[RAGChunk], chunk_type: Union[ChunkType, str]) -> List[RAGChunk]:
        target = chunk_type.value if isinstance(chunk_type, ChunkType) else chunk_type
        return [c for c in chunks if c.metadata['content_type'] == target]

    def filter_by_entity(self, chunks: List[RAGChunk], entity: str) -> List[RAGChunk]:
        return [c for c in chunks if any(entity.lower() in e.lower() for e in c.metadata['entities'])]

    def deduplicate(self, chunks: List[RAGChunk], known_hashes: Optional[set] = None) -> List[RAGChunk]:
        seen = known_hashes or set()
        unique = []
        for chunk in chunks:
            if chunk.text_hash not in seen:
                seen.add(chunk.text_hash)
                unique.append(chunk)
        return unique

    def to_jsonl(self, chunks: List[RAGChunk], output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')

    def to_embedding_batch(self, chunks: List[RAGChunk]) -> List[Dict]:
        return [chunk.to_embedding_input() for chunk in chunks]

    # ==================== АСИНХРОННАЯ ОБРАБОТКА ====================
    
    async def create_rag_chunks_async(self,
                                     text: str,
                                     doc_id: Optional[str] = None,
                                     source_meta: Optional[Dict] = None) -> List[RAGChunk]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.create_rag_chunks, 
            text, doc_id, source_meta
        )

    async def process_files_async(self,
                                  file_paths: List[str],
                                  doc_ids: Optional[List[str]] = None) -> AsyncGenerator[RAGChunk, None]:
        for i, path in enumerate(file_paths):
            doc_id = doc_ids[i] if doc_ids else None
            text = Path(path).read_text(encoding='utf-8')
            chunks = await self.create_rag_chunks_async(text, doc_id)
            for chunk in chunks:
                yield chunk
            await asyncio.sleep(0)

    def get_stats(self) -> Dict:
        return {
            'total_chunks': self._processed_count,
            'schema_version': self.SCHEMA_VERSION,
            'config': {
                'chunk_size': self.chunk_size,
                'overlap': self.chunk_overlap,
                'min_chars': self.min_chunk_chars,
            }
        }


# ========================
# Пример использования
# ========================
if __name__ == '__main__':
    chunker = AdaptiveSemanticChunker(
        chunk_size=800,
        chunk_overlap=150,
        doc_id_prefix="first_aid",
        language='russian',  # ✅ Явно указываем язык
        log_level=logging.INFO
    )
    
    sample = """
    ### Остановка артериального кровотечения
    
    ⚠️ Признаки: кровь алого цвета, бьёт пульсирующей струёй.
    
    Алгоритм действий:
    1. Немедленно прижмите артерию пальцем выше раны.
    2. Наложите жгут на 3-5 см выше места повреждения.
    3. Зафиксируйте время: максимум 1.5 часа летом, 1 час зимой.
    4. Вызовите скорую помощь (103 или 112).
    
    Противопоказания:
    - Не накладывайте жгут на голое тело.
    - Не ослабляйте жгут до прибытия медиков.
    """
    
    chunks = chunker.create_rag_chunks(
        sample,
        doc_id="bleeding_algorithm",
        source_meta={"source": "МЧС", "category": "emergency", "year": 2024}
    )
    
    print(f"📊 Создано {len(chunks)} чанков")
    for i, chunk in enumerate(chunks[:3]):  # Показать первые 3
        print(f"\n📝 Чанк {i+1} [{chunk.metadata['urgency']}]:")
        print(f"   Тип: {chunk.metadata['content_type']}")
        print(f"   Текст: {chunk.text[:150]}...")
    
    chunker.to_jsonl(chunks, "data/vector_store/bleeding_chunks.jsonl")
    print(f"\n📈 Статистика: {chunker.get_stats()}")