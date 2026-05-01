import os
import glob
import json
from sentence_transformers import SentenceTransformer

# Пути к папкам
processed_dir = r'backend/data/processed'
embeddings_dir = r'backend/data/embeddings'

# Создаем папку для эмбеддингов, если не существует
os.makedirs(embeddings_dir, exist_ok=True)

# Инициализация модели
model = SentenceTransformer('backend/models/all-MiniLM-L6-v2')

# Обрабатываем все txt файлы
for file_path in glob.glob(os.path.join(processed_dir, '*.txt')):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    # Создаем эмбеддинги
    embeddings = model.encode(texts)

    # Получаем базовое имя файла без расширения
    filename = os.path.splitext(os.path.basename(file_path))[0] + '_embeddings.json'
    save_path = os.path.join(embeddings_dir, filename)

    # Сохраняем как JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings.tolist(), f)

    print(f'Эмбеддинги для {os.path.basename(file_path)} сохранены в {save_path}')