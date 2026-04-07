import os
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

def main():
    # Пути и модель из config.py
    processed_dir = Path(r'backend/data/processed')
    embeddings_dir = Path(r'backend/data/embeddings')

    # Создаем папку для эмбеддингов, если не существует
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Инициализация модели
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Обрабатываем все txt файлы
    for file_path in processed_dir.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()

        # Создаем эмбеддинги
        embeddings = model.encode(texts)

        # Название файла для сохранения
        filename = file_path.stem + '_embeddings.json'
        save_path = embeddings_dir / filename

        # Сохраняем как JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings.tolist(), f)

        print(f'Эмбеддинги для {file_path.name} сохранены в {save_path}')
