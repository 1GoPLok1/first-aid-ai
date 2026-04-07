from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import backend.scripts.config as config

client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

def detect_category(text, source):
    text_lower = text.lower()
    source_lower = source.lower()

    if "first_aid" in source_lower or "кровотечени" in text_lower or "обморок" in text_lower:
        return "first_aid"
    elif "nutrition" in source_lower or "healthy" in source_lower or "питан" in text_lower:
        return "nutrition"
    elif "physical" in source_lower or "exercise" in source_lower or "трениров" in text_lower:
        return "physical_activity"
    else:
        return "other"

def update_all_points():
    offset = None
    updated_count = 0

    while True:

        records, offset = client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=100,
            offset=offset
        )

        if not records:
            break

        # Обновление каждой точки
        points_to_update = []
        for record in records:
            # Определяем категорию
            category = detect_category(
                record.payload.get("text", ""),
                record.payload.get("source", "")
            )

            # Добавляем теги (пример)
            tags = []
            if "first_aid" in category:
                tags = ["экстренная", "медицина"]
            elif "nutrition" in category:
                tags = ["питание", "здоровье"]

            # Обновляем payload (сохраняем старые поля + добавляем новые)
            record.payload["category"] = category
            record.payload["tags"] = tags
            record.payload["last_updated"] = "2024-03-17"

            points_to_update.append(record)

        # Загружаем обновленные точки
        if points_to_update:
            client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points_to_update
            )
            updated_count += len(points_to_update)
            print(f"Обновлено {updated_count} точек...")

    print(f"Обновление завершено. Всего обновлено точек: {updated_count}")

if __name__ == "__main__":
    update_all_points()