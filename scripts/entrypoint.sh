set -e

MODEL_NAME="${OLLAMA_MODEL:-qwen2.5:7b-instruct-q4_K_M}"

echo "=========================================="
echo "  Ollama Entrypoint"
echo "  Модель: ${MODEL_NAME}"
echo "=========================================="

echo "[1/4] Запуск ollama serve..."
ollama serve &
OLLAMA_PID=$!

echo "[2/4] Ожидание готовности сервера (5 сек)..."
sleep 5

# Дополнительная проверка — ждём, пока API станет доступен
MAX_RETRIES=10
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "      Сервер Ollama готов."
        break
    fi
    RETRY=$((RETRY + 1))
    echo "      Ожидание... (попытка ${RETRY}/${MAX_RETRIES})"
    sleep 3
done

if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "ОШИБКА: Сервер Ollama не запустился за отведённое время."
    exit 1
fi

echo "[3/4] Проверка модели '${MODEL_NAME}'..."

if ollama list | grep -q "${MODEL_NAME}"; then
    echo "      Модель уже загружена."
else
    echo "      Модель не найдена. Загрузка..."
    ollama pull "${MODEL_NAME}"

    if [ $? -eq 0 ]; then
        echo "      Модель успешно загружена."
    else
        echo "ОШИБКА: Не удалось загрузить модель '${MODEL_NAME}'."
        echo "Проверьте интернет-соединение и доступность реестра Ollama."
        exit 1
    fi
fi

echo "[4/4] Готово."
echo ""
echo "=========================================="
echo "  Ollama запущен и готов к работе"
echo "  Модель: ${MODEL_NAME}"
echo "  API:    http://localhost:11434"
echo "=========================================="

wait $OLLAMA_PID