import logging
import sys
import os
from typing import Optional, Tuple

# Добавляем корень backend в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import SYSTEM_PROMPTS, OUT_OF_SCOPE_RESPONSE, EMERGENCY_DISCLAIMER

logger = logging.getLogger(__name__)

# =========================================================================
# Константы
# =========================================================================

LABELS = ["EMERGENCY", "LIFESTYLE", "OUT_OF_SCOPE", "CHITCHAT"]

ROUTING_MAP = {
    "EMERGENCY": {
        "collection": "first_aid_protocols",
        "prompt": "emergency_prompt",
        "add_disclaimer": True,
    },
    "LIFESTYLE": {
        "collection": "healthy_lifestyle",
        "prompt": "lifestyle_prompt",
        "add_disclaimer": False,
    },
    "OUT_OF_SCOPE": {
        "collection": None,
        "prompt": None,
        "add_disclaimer": False,
        "static_response": OUT_OF_SCOPE_RESPONSE,
    },
    "CHITCHAT": {
        "collection": None,
        "prompt": "chitchat_prompt",
        "add_disclaimer": False,
    },
}

# =========================================================================
# Ключевые слова для классификации (без ML-модели)
# =========================================================================

EMERGENCY_KEYWORDS = [
    "кровь", "кровотечение", "ожог", "перелом", "слр", "реанимац",
    "не дышит", "нет дыхания", "сердце остановилось", "инсульт",
    "инфаркт", "удар током", "потерял сознание", "без сознания",
    "задыхается", "тонет", "обморожение", "судороги", "травма",
    "рана", "головокружение", "упал", "сломал", "вывих", "растяжение",
    "открытый перелом", "закрытый перелом", "ушиб", "гематома",
    "аллергия", "отек квинке", "анафилактический шок",
    "тепловой удар", "солнечный удар", "отравление", "укус",
    "ожог кипятком", "термический ожог", "химический ожог",
    "скорая", "помощь", "первая помощь", "что делать если",
    "как остановить кровь", "как наложить жгут", "как наложить шину",
    "искусственное дыхание", "непрямой массаж сердца",
]

LIFESTYLE_KEYWORDS = [
    "питание", "диета", "спорт", "упражнение", "бег", "ходьба",
    "витамин", "сон", "стресс", "вес", "калория", "зож",
    "здоровый образ", "профилактика", "зарядка", "тренировка",
    "сколько нужно спать", "норма шагов", "сколько пить воды",
    "правильное питание", "режим дня", "закаливание", "йога",
    "растяжка", "фитнес", "плавание", "велосипед",
    "сколько калорий", "белки", "жиры", "углеводы", "клетчатка",
    "медитация", "релаксация", "отдых", "гигиена сна",
]

OUT_OF_SCOPE_KEYWORDS = [
    "лекарство", "таблетка", "антибиотик", "рецепт", "дозировка",
    "пропиши", "вылечи", "диагноз", "чем лечить", "какое лекарство",
    "мазь", "укол", "прививка", "вакцина", "операция",
    "онкология", "рак", "диабет", "давление повышенное",
    "хроническое заболевание", "беременность", "роды",
]


# =========================================================================
# Классификатор
# =========================================================================

class QueryRouter:
    """
    Классификатор запросов на основе ключевых слов.

    Определяет тип запроса пользователя и возвращает:
        - метку класса (EMERGENCY, LIFESTYLE, OUT_OF_SCOPE, CHITCHAT)
        - имя коллекции Qdrant для поиска
        - имя промпта для генерации
        - готовый ответ-заглушку для запросов вне компетенции
        - флаг необходимости дисклеймера
    """

    def __init__(self):
        """Инициализация классификатора."""
        logger.info("QueryRouter инициализирован (режим: ключевые слова)")

    def classify(self, query: str) -> str:
        """
        Классифицирует запрос пользователя по ключевым словам.

        Args:
            query: Текст запроса.

        Returns:
            Одна из меток: EMERGENCY, LIFESTYLE, OUT_OF_SCOPE, CHITCHAT.
        """
        query_lower = query.lower().strip()

        # Проверка на OUT_OF_SCOPE (должна быть первой — самые жёсткие критерии)
        for word in OUT_OF_SCOPE_KEYWORDS:
            if word in query_lower:
                logger.info("Запрос классифицирован как OUT_OF_SCOPE: %.100s", query)
                return "OUT_OF_SCOPE"

        # Проверка на EMERGENCY
        for word in EMERGENCY_KEYWORDS:
            if word in query_lower:
                logger.info("Запрос классифицирован как EMERGENCY: %.100s", query)
                return "EMERGENCY"

        # Проверка на LIFESTYLE
        for word in LIFESTYLE_KEYWORDS:
            if word in query_lower:
                logger.info("Запрос классифицирован как LIFESTYLE: %.100s", query)
                return "LIFESTYLE"

        # Короткие приветствия — chitchat
        if len(query_lower) < 10 and any(
            w in query_lower for w in ["привет", "здравствуй", "пока", "как дела", "спасибо"]
        ):
            logger.info("Запрос классифицирован как CHITCHAT: %.100s", query)
            return "CHITCHAT"

        # По умолчанию — первая помощь (лучше перестраховаться)
        logger.info("Запрос классифицирован как EMERGENCY (по умолчанию): %.100s", query)
        return "EMERGENCY"

    def get_collection_and_prompt(
        self, label: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Возвращает имя коллекции Qdrant и промпта для заданной метки.

        Args:
            label: Метка класса.

        Returns:
            Кортеж (collection_name, prompt_name).
        """
        route = ROUTING_MAP.get(label, ROUTING_MAP["CHITCHAT"])
        return route.get("collection"), route.get("prompt")

    def get_static_response(self, label: str) -> Optional[str]:
        """
        Возвращает готовый ответ-заглушку для запросов вне компетенции.

        Args:
            label: Метка класса.

        Returns:
            Строка с ответом или None.
        """
        route = ROUTING_MAP.get(label, {})
        return route.get("static_response")

    def needs_retrieval(self, label: str) -> bool:
        """
        Проверяет, требуется ли поиск по базе знаний.

        Args:
            label: Метка класса.

        Returns:
            True, если нужен retrieval.
        """
        route = ROUTING_MAP.get(label, {})
        return route.get("collection") is not None

    def get_emergency_disclaimer(self) -> str:
        """
        Возвращает текст экстренного дисклеймера.

        Returns:
            Строка с требованием вызвать скорую.
        """
        return EMERGENCY_DISCLAIMER

    def should_add_disclaimer(self, label: str) -> bool:
        """
        Проверяет, нужно ли добавлять дисклеймер.

        Args:
            label: Метка класса.

        Returns:
            True, если дисклеймер нужен.
        """
        route = ROUTING_MAP.get(label, {})
        return route.get("add_disclaimer", False)

    def process_query(self, query: str) -> dict:
        """
        Полный цикл обработки запроса: классификация + маршрутизация.

        Args:
            query: Текст запроса пользователя.

        Returns:
            Словарь с результатами.
        """
        label = self.classify(query)
        collection, prompt = self.get_collection_and_prompt(label)
        static_response = self.get_static_response(label)
        needs_retrieval = self.needs_retrieval(label)
        add_disclaimer = self.should_add_disclaimer(label)
        disclaimer = self.get_emergency_disclaimer() if add_disclaimer else None

        result = {
            "query": query,
            "label": label,
            "collection": collection,
            "prompt": prompt,
            "needs_retrieval": needs_retrieval,
            "static_response": static_response,
            "disclaimer": disclaimer,
        }

        logger.info(
            "Маршрутизация: label=%s, collection=%s, prompt=%s, retrieval=%s",
            label,
            collection,
            prompt,
            needs_retrieval,
        )

        return result