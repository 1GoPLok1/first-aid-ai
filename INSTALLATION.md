# =============================================================================
# РУКОВОДСТВО ПО ИНСТАЛЛЯЦИИ ИНТЕЛЛЕКТУАЛЬНОГО АССИСТЕНТА ПМП/ЗОЖ
# =============================================================================

# Шаг 1: Клонирование репозитория
git clone https://github.com/1GoPLok1/first-aid-ai.git
cd first-aid-ai

# Шаг 2: Настройка бэкенда (Python/FastAPI)
cd server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Настройка переменных окружения
cd ..

# Шаг 3: Настройка фронтенда (React)
cd client
npm install
cp .env.example .env      # Настройка API_URL
cd ..

# Шаг 4: Запуск Qdrant через Docker
docker run -d \
  --name qdrant-medical \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest

# Проверка запуска Qdrant
curl http://localhost:6333/

# Шаг 5: Инициализация коллекции (однократно)
python server/scripts/init_collection.py

# Шаг 6: Запуск сервера разработки
# Терминал 1 — Бэкенд:
cd server && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Терминал 2 — Фронтенд:
cd client && npm run dev

# Шаг 7: Проверка работоспособности
# Открыть в браузере: http://localhost:5173
# Или протестировать API:
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Как остановить кровотечение?", "category": "pmp", "session_id": "test"}'