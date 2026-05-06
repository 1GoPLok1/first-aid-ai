"""
MedicalTextCleaner для RAG-бэкенда
Оптимизирован для индексации медицинских текстов в векторные хранилища.
"""

import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, asdict, field

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Типы контента для фильтрации в RAG"""
    EMERGENCY = "emergency"      # Экстренная помощь
    PREVENTIVE = "preventive"    # Профилактика, ЗОЖ
    MIXED = "mixed"              # Смешанный
    UNKNOWN = "unknown"


@dataclass
class RAGChunkMetadata:
    """Метаданные чанка для векторного хранилища"""
    chunk_id: str
    doc_id: str
    text_hash: str  # SHA-256 для дедупликации
    content_type: str
    urgency_level: str  # critical/high/medium/low
    entities: List[str]  # Извлечённые медицинские сущности
    key_phrases: List[str]  # Ключевые фразы для гибридного поиска
    char_count: int
    sentence_count: int
    section_title: Optional[str] = None
    schema_version: str = "1.0"
    processed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_meta: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Конвертация в dict для JSON-сериализации"""
        return asdict(self)


class MedicalTextCleaner:
    """
    Клинер для подготовки медицинских текстов к индексации в RAG-систему.
    Фокус: сохранение семантики + извлечение метаданных для улучшения поиска.
    """

    # Версия схемы метаданных (для отслеживания изменений при реиндексации)
    SCHEMA_VERSION = "1.0"
    
    # Пороги для определения срочности (настраиваются)
    URGENCY_THRESHOLDS = {
        'critical': 5,  # 5+ ключевых слов экстренности
        'high': 3,
        'medium': 1,
    }

    def __init__(self,
                 doc_id_prefix: str = "doc",
                 extract_entities: bool = True,
                 extract_keyphrases: bool = True,
                 min_chunk_chars: int = 100,
                 log_level: int = logging.WARNING):
        
        self.doc_id_prefix = doc_id_prefix
        self.extract_entities = extract_entities
        self.extract_keyphrases = extract_keyphrases
        self.min_chunk_chars = min_chunk_chars
        
        # === Регулярные выражения (скомпилированы для скорости) ===
        self._compile_patterns()
        
        # === Словари для извлечения сущностей ===
        self.medical_entities = {
            'procedures': ['СЛР', 'ИВЛ', 'НМС', 'наложение жгута', 'давящая повязка'],
            'conditions': ['кровотечение', 'ожог', 'перелом', 'инфаркт', 'инсульт', 'аллергия'],
            'medications': ['адреналин', 'нитроглицерин', 'аспирин', 'хлоргексидин'],
            'measurements': ['мм рт. ст.', '°C', 'уд/мин', 'мг', 'мл'],
        }
        
        # Ключевые слова для определения срочности
        self.urgency_keywords = {
            'critical': {'остановка сердца', 'артериальное кровотечение', 'анафилактический', 
                        'реанимация', 'немедленно', 'угроза жизни', '103', '112'},
            'high': {'сильная боль', 'высокая температура', 'потеря сознания', 
                    'травма', 'перелом', 'ожог', 'срочно'},
            'medium': {'рекомендуется', 'профилактика', 'контроль', 'обследование'},
        }
        
        # Ключевые слова для типа контента
        self.emergency_keywords = {
            'кровотечение', 'ожог', 'перелом', 'реаним', '103', '112', 'немедлен', 
            'травм', 'рана', 'шок', 'инфаркт', 'инсульт'
        }
        self.preventive_keywords = {
            'профилактик', 'рекоменд', 'питани', 'физическ', 'сон', 'стресс', 
            'иммунитет', 'витамин', 'курени', 'алкоголь', 'диспансеризац'
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self._processed_count = 0

    def _compile_patterns(self):
        """Компиляция регулярных выражений"""
        self.patterns = {
            'html': re.compile(r'<[^>]+>'),
            'references': re.compile(r'\[\d+\]|\(\d{4}\)|\[Электронный ресурс\]|https?://\S+', re.I),
            'copyright': re.compile(r'©\s*\d{4}.*?\.|\bВсе права защищены\b', re.I),
            'extra_spaces': re.compile(r'\s+'),
            'punctuation_space': re.compile(r'\s+([,.;:!?])'),
            'heading': re.compile(r'^(#{1,3}\s+|\d+[\.\)]\s+|⚠️?\s*)[А-Я][^\n]+', re.M),
            'bullet': re.compile(r'^\s*[•\-*◦▪]\s+', re.M),
            'step_marker': re.compile(r'^\s*(\d+[\.\)]|Шаг\s*\d+|Этап\s*\d+)\s+', re.M),
            'dosage': re.compile(r'(\d+[.,]?\d*)\s*([мк]г|мл|таб|кап|%)', re.I),
            'vital_signs': re.compile(r'(\d+[/\s]\d+)\s*мм\s*рт\.?\s*ст\.?', re.I),
            'temperature': re.compile(r'(\d+[.,]\d+)\s*°?\s*C', re.I),
            'phone': re.compile(r'\+?7[\s\-\(]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'),
        }

    # ==================== ОСНОВНОЙ ПЛАЙПЛАЙН ====================
    
    def prepare_for_rag(self, 
                       text: str, 
                       doc_id: Optional[str] = None,
                       source_meta: Optional[Dict] = None) -> Optional[Dict]:
        """
        Подготовка текста для индексации в RAG.
        
        Returns:
            Dict с полями 'text' и 'metadata' для векторной БД,
            или None если текст слишком короткий/невалидный.
        """
        if not text or len(text.strip()) < self.min_chunk_chars:
            return None
        
        # Базовая очистка
        cleaned = self._basic_clean(text)
        if len(cleaned) < self.min_chunk_chars:
            return None
        
        # Извлечение метаданных
        metadata = self._extract_metadata(cleaned, doc_id, source_meta or {})
        
        # Хеш для дедупликации
        text_hash = hashlib.sha256(cleaned.encode('utf-8')).hexdigest()[:16]
        
        result = {
            'text': cleaned,
            'metadata': metadata,
            'text_hash': text_hash,
            'chunk_id': metadata['chunk_id']
        }
        
        self._processed_count += 1
        self.logger.debug(f"Обработан чанк {metadata['chunk_id']} ({len(cleaned)} симв.)")
        
        return result

    def _basic_clean(self, text: str) -> str:
        """Базовая очистка с сохранением семантики"""
        # Удаление шума
        text = self.patterns['html'].sub('', text)
        text = self.patterns['references'].sub('', text)
        text = self.patterns['copyright'].sub('', text)
        
        # Нормализация пробелов (но сохранение структуры абзацев)
        text = self.patterns['extra_spaces'].sub(' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = self.patterns['punctuation_space'].sub(r'\1', text)
        
        # Unicode-нормализация
        text = text.replace('«', '"').replace('»', '"')
        text = re.sub(r'[–—−]', '-', text)
        
        return text.strip()

    def _extract_metadata(self, text: str, doc_id: Optional[str], source_meta: Dict) -> Dict:
        """Извлечение метаданных для улучшения поиска"""
        # Генерация ID
        doc_id = doc_id or f"{self.doc_id_prefix}_{self._processed_count}"
        chunk_id = f"{doc_id}_{self._processed_count}"
        
        # Определение типа контента
        content_type = self._categorize_content(text)
        
        # Определение срочности
        urgency = self._detect_urgency(text)
        
        # Извлечение сущностей (опционально)
        entities = self._extract_entities(text) if self.extract_entities else []
        
        # Извлечение ключевых фраз (опционально)
        key_phrases = self._extract_key_phrases(text) if self.extract_keyphrases else []
        
        # Извлечение заголовка раздела
        section_title = self._extract_section_title(text)
        
        return RAGChunkMetadata(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text_hash="",  # Заполняется позже
            content_type=content_type.value,
            urgency_level=urgency,
            entities=entities,
            key_phrases=key_phrases,
            char_count=len(text),
            sentence_count=text.count('.') + text.count('!') + text.count('?'),
            section_title=section_title,
            schema_version=self.SCHEMA_VERSION,
            source_meta={k: v for k, v in source_meta.items() if k != 'doc_id'}
        ).to_dict()

    def _categorize_content(self, text: str) -> ContentType:
        """Определение типа контента"""
        text_lower = text.lower()
        emergency_score = sum(1 for kw in self.emergency_keywords if kw in text_lower)
        preventive_score = sum(1 for kw in self.preventive_keywords if kw in text_lower)
        
        if emergency_score > preventive_score + 2:
            return ContentType.EMERGENCY
        elif preventive_score > emergency_score + 2:
            return ContentType.PREVENTIVE
        elif emergency_score > 0 or preventive_score > 0:
            return ContentType.MIXED
        return ContentType.UNKNOWN

    def _detect_urgency(self, text: str) -> str:
        """Определение уровня срочности"""
        text_lower = text.lower()
        for level, keywords in self.urgency_keywords.items():
            if sum(1 for kw in keywords if kw in text_lower) >= self.URGENCY_THRESHOLDS.get(level, 1):
                return level
        return 'low'

    def _extract_entities(self, text: str) -> List[str]:
        """Извлечение медицинских сущностей"""
        entities = []
        for category, terms in self.medical_entities.items():
            for term in terms:
                if term.lower() in text.lower():
                    entities.append(f"{category}:{term}")
        
        # Извлечение дозировок и показателей
        for match in self.patterns['dosage'].finditer(text):
            entities.append(f"dosage:{match.group(0)}")
        for match in self.patterns['vital_signs'].finditer(text):
            entities.append(f"vital:{match.group(0)}")
        for match in self.patterns['temperature'].finditer(text):
            entities.append(f"temperature:{match.group(0)}")
        
        return list(set(entities))

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Простое извлечение ключевых фраз (можно заменить на YAKE/Rake)"""
        # Удаляем стоп-слова и короткие слова
        stop_words = {'и', 'в', 'не', 'на', 'с', 'по', 'для', 'от', 'при', 'как', 'что', 'это'}
        words = re.findall(r'[а-яёa-z]{4,}', text.lower(), re.I)
        
        # Подсчёт частоты
        freq = {}
        for w in words:
            if w not in stop_words:
                freq[w] = freq.get(w, 0) + 1
        
        # Топ-10 фраз
        return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:10]]

    def _extract_section_title(self, text: str) -> Optional[str]:
        """Извлечение заголовка раздела"""
        match = self.patterns['heading'].match(text)
        if match:
            return match.group(0).strip().lstrip('#').strip()
        return None

    # ==================== ПАКЕТНАЯ ОБРАБОТКА ====================
    
    def prepare_batch(self, 
                     texts: List[str], 
                     doc_ids: Optional[List[str]] = None,
                     source_metas: Optional[List[Dict]] = None) -> List[Dict]:
        """Пакетная подготовка текстов для индексации"""
        results = []
        for i, text in enumerate(texts):
            doc_id = doc_ids[i] if doc_ids else None
            meta = source_metas[i] if source_metas else None
            result = self.prepare_for_rag(text, doc_id, meta)
            if result:
                results.append(result)
        return results

    def prepare_from_file(self, 
                         file_path: str, 
                         doc_id: Optional[str] = None,
                         encoding: str = 'utf-8',
                         split_by_paragraphs: bool = True) -> List[Dict]:
        """Загрузка и подготовка текста из файла"""
        text = Path(file_path).read_text(encoding=encoding)
        
        if split_by_paragraphs:
            # Разбиение на абзацы как отдельные чанки
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            return self.prepare_batch(paragraphs, doc_ids=[doc_id] * len(paragraphs) if doc_id else None)
        else:
            result = self.prepare_for_rag(text, doc_id)
            return [result] if result else []

    # ==================== УТИЛИТЫ ====================
    
    def get_stats(self) -> Dict:
        """Статистика обработки"""
        return {
            'processed_count': self._processed_count,
            'schema_version': self.SCHEMA_VERSION,
        }
    
    def is_duplicate(self, text: str, known_hashes: set) -> bool:
        """Проверка на дубликат по хешу"""
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return text_hash in known_hashes
    
    def filter_by_urgency(self, chunks: List[Dict], min_urgency: str) -> List[Dict]:
        """Фильтрация чанков по минимальному уровню срочности"""
        urgency_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        min_level = urgency_order.get(min_urgency, 0)
        return [c for c in chunks if urgency_order.get(c['metadata']['urgency_level'], 0) >= min_level]
    
    def filter_by_content_type(self, chunks: List[Dict], content_type: Union[ContentType, str]) -> List[Dict]:
        """Фильтрация по типу контента"""
        target = content_type.value if isinstance(content_type, ContentType) else content_type
        return [c for c in chunks if c['metadata']['content_type'] == target]


# ========================
# Пример использования в RAG-пайплайне
# ========================
if __name__ == '__main__':
    # Инициализация клинера
    cleaner = MedicalTextCleaner(
        doc_id_prefix="first_aid",
        extract_entities=True,
        extract_keyphrases=True,
        log_level=logging.INFO
    )
    
    # Пример текста
    sample = """
    ### Остановка кровотечения
    
    1. Прижмите рану стерильной салфеткой.
    2. Наложите давящую повязку.
    3. Если кровь алого цвета и бьёт струёй — это артериальное кровотечение. 
       Немедленно вызовите 103!
    
    Дозировка: хлоргексидин 0.05% для обработки.
    """
    
    # Подготовка для RAG
    result = cleaner.prepare_for_rag(
        sample, 
        doc_id="bleeding_guide",
        source_meta={"source": "МЧС", "year": 2024}
    )
    
    if result:
        print(f"✅ Чанк готов к индексации:")
        print(f"   ID: {result['chunk_id']}")
        print(f"   Хеш: {result['text_hash']}")
        print(f"   Тип: {result['metadata']['content_type']}")
        print(f"   Срочность: {result['metadata']['urgency_level']}")
        print(f"   Сущности: {result['metadata']['entities']}")
        print(f"   Ключевые фразы: {result['metadata']['key_phrases']}")
        print(f"\n📝 Текст ({result['metadata']['char_count']} симв.):")
        print(result['text'][:200] + "...")
    
    # Статистика
    print(f"\n📊 Статистика: {cleaner.get_stats()}")