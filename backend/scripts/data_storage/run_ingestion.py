"""
DataIngestionPipeline для RAG-бэкенда медицинских данных
Полный пайплайн: извлечение → очистка → чанкинг → эмбеддинги → векторная БД
"""

import os
import sys
import json
import hashlib
import logging
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Union, AsyncGenerator, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import glob

# === Опциональные зависимости ===
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("⚠️ pdfplumber не установлен: pip install pdfplumber")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_SUPPORT = True
except ImportError:
    EMBEDDING_SUPPORT = False
    logging.warning("⚠️ sentence-transformers не установлен: pip install sentence-transformers")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_SUPPORT = True
except ImportError:
    QDRANT_SUPPORT = False
    logging.warning("⚠️ qdrant-client не установлен: pip install qdrant-client")

# Локальные модули (ваши классы)
from cleaner import MedicalTextCleaner
from chunker import AdaptiveSemanticChunker, RAGChunk, UrgencyLevel, ChunkType

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/ingestion.log', encoding='utf-8', mode='a', delay=True)
    ]
)
logger = logging.getLogger(__name__)


# ==================== КОНФИГУРАЦИЯ ====================

@dataclass
class PipelineConfig:
    """Конфигурация пайплайна"""
    # Пути
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    chunks_dir: str = "data/chunks"
    embeddings_dir: str = "data/embeddings"
    vector_store_path: str = "data/vector_store"  # для локального кэша, если нужно
    
    # Обработка файлов
    supported_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.txt', '.md', '.html'])
    max_file_size_mb: int = 50
    encoding: str = 'utf-8'
    
    # Чанкинг
    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_chars: int = 100
    
    # === ЭМБЕДДИНГИ ===
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # ✅ Обновлено
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"  # или "cuda" если есть GPU
    cache_embeddings: bool = True
    save_embeddings_to_file: bool = True  # ✅ Сохранять векторы в файл
    
    # === ВЕКТОРНОЕ ХРАНИЛИЩЕ ===
    vector_store_type: str = "qdrant"  # ✅ Только Qdrant
    collection_name: str = "medical_rag"
    
    # Качество
    min_chunk_quality_score: float = 0.5
    remove_duplicates: bool = True
    
    # Инкрементальная обработка
    incremental_mode: bool = True
    state_file: str = "data/.ingestion_state.json"
    
    # Асинхронность
    max_concurrent_files: int = 4
    use_async: bool = False
    
    # === QDRANT CLOUD CONFIG ===
    qdrant_config: Dict = field(default_factory=lambda: {
        # ✅ ОБЯЗАТЕЛЬНО заполните эти поля перед запуском:
        'url': 'https://YOUR-CLUSTER-ID.aws.cloud.qdrant.io',  # из Qdrant Cloud Console
        'api_key': 'YOUR-API-KEY-HERE',                         # из Qdrant Cloud Console
        
        # Опциональные настройки:
        'timeout': 60,           # таймаут запросов (сек)
        'grpc_port': 6334,       # для gRPC (быстрее)
        'http_port': 6333,       # для HTTP API
        'prefer_grpc': True,     # использовать gRPC если доступен
    })

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}

# ==================== МЕТРИКИ ====================

@dataclass
class PipelineMetrics:
    """Метрики выполнения пайплайна"""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    files_processed: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    chunks_filtered: int = 0
    embeddings_generated: int = 0
    duplicates_removed: int = 0
    total_chars_processed: int = 0
    errors: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['start_time'] = data['start_time'].isoformat()
        data['end_time'] = datetime.now(timezone.utc).isoformat()
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        data['processing_time_sec'] = elapsed
        data['throughput_chars_per_sec'] = self.total_chars_processed / elapsed if elapsed > 0 else 0
        return data
    
    def log_summary(self) -> None:
        logger.info("📊 Метрики пайплайна:")
        for k, v in self.to_dict().items():
            if k not in ['start_time', 'end_time', 'errors']:
                logger.info(f"   {k}: {v}")
        if self.errors:
            logger.warning(f"   Ошибок: {len(self.errors)}")


# ==================== СЕРВИС ЭМБЕДДИНГОВ ====================

class EmbeddingService:
    """Сервис генерации эмбеддингов с кэшированием"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.cache: Dict[str, List[float]] = {}
        self._cache_file = Path(config.embeddings_dir) / "embedding_cache.json"
        self._load_cache()
        self._init_model()
    
    def _init_model(self):
        """Инициализация модели эмбеддингов"""
        if not EMBEDDING_SUPPORT:
            logger.warning("Используются заглушки эмбеддингов. Установите sentence-transformers.")
            return
        
        try:
            logger.info(f"Загрузка модели: {self.config.embedding_model}")
            self.model = SentenceTransformer(
                self.config.embedding_model,
                device=self.config.embedding_device
            )
            logger.info("✅ Модель загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.model = None
    
    def get_dimension(self) -> int:
        """Возвращает размерность эмбеддинга"""
        model_dims = {
            "all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-MiniLM-L6-v2": 384,  # ✅ с префиксом
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
            "paraphrase-multilingual-mpnet-base-v2": 768,
            "intfloat/multilingual-e5-large": 1024,
            "sentence-transformers/laBSE": 768,
        }
        # Поддержка как с префиксом, так и без
        model_key = self.config.embedding_model
        if model_key in model_dims:
            return model_dims[model_key]
        # Пробуем без префикса
        simple_key = model_key.split('/')[-1]
        return model_dims.get(simple_key, 384)  # fallback на 384

    def _load_cache(self):
        """Загрузка кэша эмбеддингов"""
        if self.config.cache_embeddings and self._cache_file.exists():
            try:
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Загружено {len(self.cache)} эмбеддингов из кэша")
            except:
                pass
    
    def _save_cache(self):
        """Сохранение кэша"""
        if self.config.cache_embeddings:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False)
    
    def _hash_text(self, text: str) -> str:
        """Хеш текста для кэширования"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Генерация эмбеддингов"""
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        results = []
        batch = []
        batch_keys = []
        
        for text in texts:
            text_hash = self._hash_text(text)
            
            # Проверка кэша
            if self.config.cache_embeddings and text_hash in self.cache:
                results.append(self.cache[text_hash])
                continue
            
            batch.append(text)
            batch_keys.append(text_hash)
        
        # Генерация для новых текстов
        if batch:
            if self.model and EMBEDDING_SUPPORT:
                # Пакетная генерация
                for i in range(0, len(batch), self.config.embedding_batch_size):
                    batch_slice = batch[i:i + self.config.embedding_batch_size]
                    embeddings = self.model.encode(
                        batch_slice, 
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    ).tolist()
                    
                    for emb, key in zip(embeddings, batch_keys[i:i + len(batch_slice)]):
                        self.cache[key] = emb
                        results.append(emb)
            else:
                # Заглушки для тестов
                import numpy as np
                for _ in batch:
                    emb = np.random.randn(384).astype(np.float32).tolist()
                    results.append(emb)
        
        self._save_cache()
        
        return results[0] if single else results
    
    async def encode_async(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Асинхронная версия"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, texts)


# ==================== ВЕКТОРНОЕ ХРАНИЛИЩЕ ====================

class VectorStoreManager:
    """Менеджер векторного хранилища (абстракция)"""
    
    def __init__(self, config: PipelineConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self.store = None
        self._init_store()
    
    def _init_store(self):
        """Инициализация векторного хранилища"""
        vector_size = self.embedding_service.get_dimension() if hasattr(self.embedding_service, 'get_dimension') else 384

        if self.config.vector_store_type == "chromadb":
            # ... существующий код ChromaDB ...
            pass
        
        elif self.config.vector_store_type == "qdrant" and QDRANT_SUPPORT:
            # Поддержка локального или удалённого Qdrant
            qdrant_config = getattr(self.config, 'qdrant_config', {})

            if qdrant_config.get('url'):
                # Подключение к удалённому Qdrant (Cloud или self-hosted)
                self.client = QdrantClient(
                    url=qdrant_config['url'],
                    api_key=qdrant_config.get('api_key'),
                    prefer_grpc=True  # gRPC быстрее для больших данных
                )
                logger.info(f"✅ Подключено к Qdrant: {qdrant_config['url']}")
            else:
                # Локальный Qdrant (on-disk)
                qdrant_path = Path(self.config.vector_store_path) / "qdrant_storage"
                qdrant_path.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(qdrant_path))
                logger.info(f"✅ Локальный Qdrant: {qdrant_path}")

            # Создание или получение коллекции
            collection_name = self.config.collection_name

            try:
                self.client.get_collection(collection_name)
                logger.info(f"✅ Коллекция '{collection_name}' найдена")
            except (UnexpectedResponse, ValueError):
                # Коллекция не существует — создаём
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=vector_size,
                        distance=qdrant_models.Distance.COSINE
                    ),
                    optimizers_config=qdrant_models.OptimizersConfigDiff(
                        default_segment_number=2,
                        memmap_threshold=20000  # оптимизация для больших данных
                    ),
                    hnsw_config=qdrant_models.HnswConfigDiff(
                        m=16,  # количество связей для поиска
                        ef_construct=100  # точность построения графа
                    )
                )
                logger.info(f"✅ Создана коллекция '{collection_name}'")

            self.collection_name = collection_name

        elif self.config.vector_store_type == "memory":
            # ... существующий код in-memory ...
            pass
        else:
            logger.warning(f"Векторное хранилище '{self.config.vector_store_type}' не поддерживается")

# === Новые методы для Qdrant ===

def _prepare_qdrant_payload(self, chunk: Dict) -> Dict:
    """Подготовка метаданных для Qdrant (с фильтрацией для payload indexing)"""
    meta = chunk.get('metadata', {})
    
    # Поля, по которым будем фильтровать в поиске
    filterable_fields = ['urgency', 'content_type', 'doc_id', 'source']
    
    payload = {
        'text': chunk.get('text', ''),
        'chunk_id': chunk.get('chunk_id'),
        **{k: v for k, v in meta.items() if k in filterable_fields}
    }
    
    # Остальные метаданные храним в nested-поле (не индексируются, но доступны)
    payload['_meta'] = {k: v for k, v in meta.items() if k not in filterable_fields}
    
    return payload

def add_chunks(self, chunks: List[Dict], batch_size: int = 100) -> int:
    """Добавление чанков в индекс (универсальный метод)"""
    if not chunks:
        return 0
    
    added = 0
    
    if self.config.vector_store_type == "qdrant" and QDRANT_SUPPORT:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Подготовка данных
            ids = [c['chunk_id'] for c in batch]
            payloads = [self._prepare_qdrant_payload(c) for c in batch]
            texts = [c['text'] for c in batch]
            
            # Генерация эмбеддингов
            embeddings = self.embedding_service.encode(texts)
            
            # Upsert в Qdrant (автоматически обновляет существующие)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=idx,  # Qdrant принимает int или UUID
                        vector=emb,
                        payload=payload
                    )
                    for idx, emb, payload in zip(ids, embeddings, payloads)
                ],
                wait=True  # ждать подтверждения записи
            )
            added += len(batch)
            logger.debug(f"Добавлено {len(batch)} чанков в Qdrant")
        
        logger.info(f"✅ Всего добавлено {added} чанков в Qdrant")
        
    elif self.config.vector_store_type == "chromadb":
        # ... существующий код ChromaDB ...
        pass
    
    # ... остальные хранилища ...
    
    return added

def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
    """Поиск с поддержкой фильтров Qdrant"""
    if self.config.vector_store_type == "qdrant" and QDRANT_SUPPORT:
        # Генерация эмбеддинга запроса
        query_embedding = self.embedding_service.encode(query)
        
        # Построение фильтра Qdrant
        qdrant_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # Любой из значений (OR)
                    must_conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchAny(any=value)
                        )
                    )
                else:
                    # Точное совпадение
                    must_conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=value)
                        )
                    )
            if must_conditions:
                qdrant_filter = qdrant_models.Filter(must=must_conditions)
        
        # Поиск
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False  # не возвращать вектора для экономии трафика
        )
        
        # Форматирование ответа
        return [
            {
                'chunk_id': hit.payload.get('chunk_id'),
                'text': hit.payload.get('text', ''),
                'metadata': {
                    **hit.payload,
                    **(hit.payload.get('_meta', {}))  # раскрываем вложенные метаданные
                },
                'score': hit.score,
                'vector_id': hit.id
            }
            for hit in results
        ]
    
    elif self.config.vector_store_type == "chromadb":
        # ... существующий код ChromaDB ...
        pass
    
    # ... остальные хранилища ...
    
    return []

def get_stats(self) -> Dict:
    """Статистика хранилища"""
    if self.config.vector_store_type == "qdrant" and QDRANT_SUPPORT:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status,
                "name": self.collection_name
            }
        except:
            return {"error": "Не удалось получить статистику Qdrant"}
    
    # ... остальные хранилища ...
    return {}


# ==================== ВАЛИДАТОР КАЧЕСТВА ====================

class QualityValidator:
    """Валидация качества чанков перед индексацией"""
    
    MIN_WORDS = 15
    MAX_WORDS = 300
    MIN_SENTENCES = 1
    
    @classmethod
    def validate(cls, chunk: Dict) -> Tuple[bool, List[str]]:
        """
        Проверка качества чанка.
        Returns: (is_valid, warnings)
        """
        warnings = []
        text = chunk.get('text', '')
        meta = chunk.get('metadata', {})
        
        # Проверка длины
        word_count = len(text.split())
        if word_count < cls.MIN_WORDS:
            warnings.append(f"too_short:{word_count} words")
        if word_count > cls.MAX_WORDS:
            warnings.append(f"too_long:{word_count} words")
        
        # Проверка на "мусор"
        if text.count(' ') / max(len(text), 1) > 0.5:
            warnings.append("excessive_spaces")
        
        # Проверка медицинских сущностей
        if not meta.get('entities') and meta.get('content_type') in ['dosage', 'algorithm']:
            warnings.append("missing_entities")
        
        # Проверка срочности для экстренного контента
        if meta.get('urgency') == 'critical' and word_count < 30:
            warnings.append("critical_too_short")
        
        is_valid = len([w for w in warnings if 'too_short' in w or 'excessive' in w]) == 0
        return is_valid, warnings


# ==================== ИНКРЕМЕНТАЛЬНЫЙ ПРОЦЕССОР ====================

class IncrementalProcessor:
    """Обработка только новых/изменённых файлов"""
    
    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Загрузка состояния"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"files": {}, "last_run": None}
    
    def _save_state(self):
        """Сохранение состояния"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state["last_run"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def _get_file_hash(self, path: str) -> str:
        """Хеш файла для детекции изменений"""
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def needs_processing(self, file_path: str) -> bool:
        """Проверка, нужно ли обрабатывать файл"""
        file_hash = self._get_file_hash(file_path)
        stored = self.state["files"].get(file_path)
        
        if stored is None:
            return True  # Новый файл
        if stored["hash"] != file_hash:
            return True  # Файл изменён
        return False  # Файл уже обработан
    
    def mark_processed(self, file_path: str, chunk_count: int):
        """Отметить файл как обработанный"""
        self.state["files"][file_path] = {
            "hash": self._get_file_hash(file_path),
            "chunk_count": chunk_count,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        self._save_state()
    
    def get_unprocessed_files(self, file_paths: List[str]) -> List[str]:
        """Получить список файлов, требующих обработки"""
        return [f for f in file_paths if self.needs_processing(f)]


# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================

class DataIngestionPipeline:
    """Полный RAG-пайплайн для медицинских данных"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Создаём директории
        for dir_path in [
            self.config.processed_dir,
            self.config.chunks_dir,
            self.config.embeddings_dir,
            self.config.vector_store_path
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Компоненты
        self.cleaner = MedicalTextCleaner()
        self.chunker = AdaptiveSemanticChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_chars=self.config.min_chunk_chars
        )
        self.embedding_service = EmbeddingService(self.config)
        self.vector_store = VectorStoreManager(self.config, self.embedding_service)
        self.incremental = IncrementalProcessor(self.config.state_file) if self.config.incremental_mode else None
        
        self.metrics = PipelineMetrics()
        logger.info("🚀 Пайплайн инициализирован")
    
    def _extract_text(self, file_path: str) -> Optional[str]:
        """Извлечение текста из файла"""
        ext = Path(file_path).suffix.lower()
        
        try:
            if ext == '.pdf':
                if not PDF_SUPPORT:
                    logger.error(f"PDF поддержка не установлена: {file_path}")
                    return None
                with pdfplumber.open(file_path) as pdf:
                    return '\n'.join(p.extract_text() or '' for p in pdf.pages)
            
            elif ext in ['.txt', '.md', '.html']:
                with open(file_path, 'r', encoding=self.config.encoding) as f:
                    return f.read()
            
            else:
                logger.warning(f"Неподдерживаемый формат: {ext}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из {file_path}: {e}")
            self.metrics.errors.append({"file": file_path, "error": str(e)})
            return None
    
    def _process_file(self, file_path: str) -> Optional[List[Dict]]:
        """Обработка одного файла: извлечение → очистка → чанкинг"""
        logger.info(f"📄 Обработка: {Path(file_path).name}")
        
        # Извлечение текста
        raw_text = self._extract_text(file_path)
        if not raw_text:
            return None
        
        self.metrics.total_chars_processed += len(raw_text)
        
        # Очистка
        cleaned = self.cleaner.clean(raw_text)
        
        # Подготовка метаданных
        doc_id = Path(file_path).stem
        doc_metadata = {
            'doc_id': doc_id,
            'source': str(file_path),
            'file_type': Path(file_path).suffix,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'original_length': len(raw_text),
            'cleaned_length': len(cleaned)
        }
        
        # Чанкинг (используем RAG-версию чанкера)
        chunks = self.chunker.create_rag_chunks(
            cleaned,
            doc_id=doc_id,
            source_meta=doc_metadata
        )
        
        # Конвертация в dict для совместимости
        chunk_dicts = [c.to_dict() for c in chunks]
        
        # Валидация качества
        if self.config.remove_duplicates or self.config.min_chunk_quality_score > 0:
            validated = []
            for chunk in chunk_dicts:
                is_valid, warnings = QualityValidator.validate(chunk)
                if is_valid:
                    validated.append(chunk)
                else:
                    self.metrics.chunks_filtered += 1
                    if warnings:
                        logger.debug(f"Отфильтрован чанк {chunk['chunk_id']}: {warnings}")
            chunk_dicts = validated
        
        # Сохранение очищенного текста
        processed_path = Path(self.config.processed_dir) / f"{doc_id}.txt"
        with open(processed_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        logger.info(f"✅ Создано {len(chunk_dicts)} чанков из {Path(file_path).name}")
        return chunk_dicts
    
    async def _process_file_async(self, file_path: str) -> Optional[List[Dict]]:
        """Асинхронная обработка файла"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_file, file_path)
    
    def _collect_files(self) -> List[str]:
        """Сбор файлов для обработки"""
        files = []
        for ext in self.config.supported_extensions:
            pattern = os.path.join(self.config.raw_dir, f'*{ext}')
            files.extend(glob.glob(pattern))
        
        # Фильтрация по инкрементальному режиму
        if self.incremental:
            files = self.incremental.get_unprocessed_files(files)
            logger.info(f"Найдено {len(files)} файлов для обработки (инкрементально)")
        
        return files
    
    def _export_chunks(self, chunks: List[Dict]):
        """Экспорт чанков в различные форматы"""
        # JSON (все чанки)
        chunks_path = Path(self.config.chunks_dir) / "all_chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # JSONL (для индексации)
        indexing_path = Path(self.config.embeddings_dir) / "chunks_for_indexing.jsonl"
        with open(indexing_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                line = {
                    'id': chunk['chunk_id'],
                    'text': chunk['text'][:2000],  # обрезка для моделей
                    'metadata': {
                        k: v for k, v in chunk['metadata'].items()
                        if k in ['urgency', 'content_type', 'entities', 'doc_id']
                    }
                }
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        # Статистика
        stats_path = Path(self.config.chunks_dir) / "processing_stats.json"
        stats = {
            'total_chunks': len(chunks),
            'by_urgency': {},
            'by_type': {},
            'avg_length': sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0
        }
        for chunk in chunks:
            meta = chunk['metadata']
            stats['by_urgency'][meta.get('urgency', 'unknown')] = \
                stats['by_urgency'].get(meta.get('urgency', 'unknown'), 0) + 1
            stats['by_type'][meta.get('content_type', 'unknown')] = \
                stats['by_type'].get(meta.get('content_type', 'unknown'), 0) + 1
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📦 Экспортировано: {chunks_path.name}, {indexing_path.name}")
    
    def run(self) -> List[Dict]:
        """Синхронный запуск пайплайна"""
        logger.info("🚀 Запуск пайплайна...")
        start_time = datetime.now(timezone.utc)
        
        all_chunks = []
        files = self._collect_files()
        
        for file_path in files:
            try:
                chunks = self._process_file(file_path)
                if chunks:
                    all_chunks.extend(chunks)
                    self.metrics.files_processed += 1
                    self.metrics.chunks_created += len(chunks)
                    
                    # Отметка в инкрементальном процессоре
                    if self.incremental:
                        self.incremental.mark_processed(file_path, len(chunks))
                        
            except Exception as e:
                logger.error(f"❌ Ошибка обработки {file_path}: {e}")
                self.metrics.files_failed += 1
                self.metrics.errors.append({"file": file_path, "error": str(e)})
        
        # Индексация в векторном хранилище
        if all_chunks and self.config.vector_store_type != "none":
            logger.info("🔗 Индексация в векторном хранилище...")
            added = self.vector_store.add_chunks(all_chunks)
            self.metrics.embeddings_generated += added
        
        # Экспорт
        self._export_chunks(all_chunks)
        
        # Финализация метрик
        self.metrics.processing_time_sec = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.metrics.log_summary()
        
        logger.info("✅ Пайплайн завершён")
        return all_chunks
    
    async def run_async(self) -> List[Dict]:
        """Асинхронный запуск пайплайна"""
        logger.info("🚀 Запуск асинхронного пайплайна...")
        start_time = datetime.now(timezone.utc)
        
        all_chunks = []
        files = self._collect_files()
        
        # Ограничение параллелизма
        semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        
        async def process_with_semaphore(file_path: str):
            async with semaphore:
                return await self._process_file_async(file_path)
        
        tasks = [process_with_semaphore(fp) for fp in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for file_path, result in zip(files, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Ошибка {file_path}: {result}")
                self.metrics.files_failed += 1
                self.metrics.errors.append({"file": file_path, "error": str(result)})
            elif result:
                all_chunks.extend(result)
                self.metrics.files_processed += 1
                self.metrics.chunks_created += len(result)
                
                if self.incremental:
                    self.incremental.mark_processed(file_path, len(result))
        
        # Индексация
        if all_chunks and self.config.vector_store_type != "none":
            logger.info("🔗 Индексация...")
            added = self.vector_store.add_chunks(all_chunks)
            self.metrics.embeddings_generated += added
        
        self._export_chunks(all_chunks)
        self.metrics.processing_time_sec = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.metrics.log_summary()
        
        return all_chunks
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Поиск по векторному хранилищу"""
        return self.vector_store.search(query, top_k, filters)
    
    def get_stats(self) -> Dict:
        """Получение статистики"""
        return {
            'metrics': self.metrics.to_dict(),
            'vector_store': self.vector_store.get_stats(),
            'config': self.config.to_dict()
        }


# ========================
# Пример использования
# ========================
if __name__ == "__main__":
    # Конфигурация
    config = PipelineConfig(
        raw_dir="data/raw",
        chunk_size=800,
        chunk_overlap=150,
        embedding_model="backend/models/all-MiniLM-L6-v2",
        vector_store_type="qdrant",
        incremental_mode=True,
        remove_duplicates=True,
        vector_store_type="qdrant",
        collection_name="medical_rag",
        qdrant_config={
            'url': 'https://your-cluster.cloud.qdrant.io',
            'api_key': 'your-secret-key'
        }
    )
    
    # Инициализация пайплайна
    pipeline = DataIngestionPipeline(config)
    
    # Запуск (синхронный или асинхронный)
    if config.use_async:
        chunks = asyncio.run(pipeline.run_async())
    else:
        chunks = pipeline.run()
    
    # Тестовый поиск
    if chunks:
        print("\n🔍 Тестовый поиск: 'что делать при кровотечении'")
        results = pipeline.search("что делать при кровотечении", top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. [{r['metadata'].get('urgency')}] Score: {r['score']:.3f}")
            print(f"   {r['text'][:200]}...")
    
    # Статистика
    print(f"\n📊 Статистика: {json.dumps(pipeline.get_stats(), ensure_ascii=False, indent=2)}")