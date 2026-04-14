import os
import glob
import numpy as np
from fastembed import FastEmbed

# Папки
input_dir = r'backend/data/processed'
output_dir = r'backend/data/embeddings'

# Проверка директорий
if not os.path.exists(input_dir):
    print(f"[ОШИБКА] Папка с текстами '{input_dir}' не найдена.")
    exit(1)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"[INFO] Папка для эмбеддингов '{output_dir}' создана.")

# Инициализация эмбеддера
print("[INFO] Инициализация FastEmbed...")
embedder = FastEmbed()

# Поиск файлов с текстами (например, .txt)
text_files = glob.glob(os.path.join(input_dir, '*.txt'))
if not text_files:
    print(f"[ОШИБКА] Не найдено файлов .txt в '{input_dir}'")
    exit(1)

print(f"[INFO] Найдено файлов: {len(text_files)}")
for idx, file_path in enumerate(text_files, 1):
    print(f"\n[ШАГ {idx}] Обработка файла: {os.path.basename(file_path)}")
    
    # Чтение текста
    with open(file_path, encoding='utf-8') as f:
        text = f.read()
    print(f"[INFO] Длина текста: {len(text)} символов")
    
    # Получение эмбеддинга
    embedding = embedder.encode([text])
    embedding = np.array(embedding[0])  # Вытаскиваем из списка

    # Имя файла для эмбеддинга (например, example.txt → example.npy)
    emb_filename = os.path.splitext(os.path.basename(file_path))[0] + '.npy'
    emb_path = os.path.join(output_dir, emb_filename)

    # Сохранение эмбеддинга
    np.save(emb_path, embedding)
    print(f"[INFO] Эмбеддинг сохранён: {emb_path}")

print("\n[УСПЕХ] Все эмбеддинги созданы и сохранены!")