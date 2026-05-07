import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json
import os

class JSONFormatter(logging.Formatter):
    """
    Кастомный форматтер для записи логов в JSON.
    Позволяет легко парсить логи для аналитики.
    """
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Добавление дополнительных полей если есть
        if hasattr(record, 'session_id'):
            log_data['session_id'] = record.session_id
        if hasattr(record, 'query_length'):
            log_data['query_length'] = record.query_length
        if hasattr(record, 'processing_time'):
            log_data['processing_time_ms'] = record.processing_time
            
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging():
    """
    Настройка многоуровневой системы логирования:
    - Console: для разработки
    - File: все логи
    - Error File: только ошибки
    - Access Log: HTTP запросы
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 1. Консольный обработчик (для разработки)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # 2. Файловый обработчик (все логи)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        logs_dir / "app.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # 3. Обработчик ошибок (только ERROR и выше)
    error_handler = logging.handlers.TimedRotatingFileHandler(
        logs_dir / "errors.log",
        when="midnight",
        interval=1,
        backupCount=90,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)
    
    # 4. Лог доступа (HTTP запросы)
    access_logger = logging.getLogger("uvicorn.access")
    access_handler = logging.handlers.TimedRotatingFileHandler(
        logs_dir / "access.log",
        when="hourly",
        interval=1,
        backupCount=168,  # Неделя
        encoding="utf-8"
    )
    access_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))
    access_logger.addHandler(access_handler)
    
    logging.info("Logging system initialized")

# Специализированный логгер для аудита запросов
class QueryAuditLogger:
    """Логгер для аудита пользовательских запросов"""
    
    def __init__(self):
        self.logger = logging.getLogger("query_audit")
        self.audit_file = Path("logs/queries.jsonl")
        self.audit_file.parent.mkdir(exist_ok=True)
    
    def log_query(self, session_id: str, query: str, 
                  category: str, response_time: float,
                  results_count: int, success: bool):
        """Запись информации о запросе в JSONL формат"""
        audit_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "query_preview": query[:100],
            "query_length": len(query),
            "category": category,
            "response_time_ms": round(response_time * 1000, 2),
            "results_count": results_count,
            "success": success
        }
        
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_record, ensure_ascii=False) + "\n")

# Использование
audit_logger = QueryAuditLogger()