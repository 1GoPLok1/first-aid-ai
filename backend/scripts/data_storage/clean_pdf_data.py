import os
import re
import pdfplumber

raw_path = r'backend/data/raw'
processed_path = r'backend/data/processed'
os.makedirs(processed_path, exist_ok=True)

# Размер чанка
CHUNK_SIZE = 2000

# Ключевые слова для пропуска страниц
skip_keywords = ['Содержание', 'Введение', 'Introduction', 'Оглавление', 'Обзор']

def should_skip_page(text):
    # Проверка, содержит ли страница ключевые слова
    for keyword in skip_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
            return True
    return False

def split_text_into_chunks(text):
    # Разбитие текста на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
            current_chunk += sentence + ' '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

for filename in os.listdir(raw_path):
    if filename.lower().endswith('.pdf'):
        file_path = os.path.join(raw_path, filename)
        with pdfplumber.open(file_path) as pdf:
            full_text = ''
            for page in pdf.pages:
                text = page.extract_text()
                if text and not should_skip_page(text):
                    full_text += text + '\n'

        # Деление на чанки
        chunks = split_text_into_chunks(full_text)
        
        # Сохраняем каждую часть
        for idx, chunk in enumerate(chunks):
            filename_out = f"{os.path.splitext(filename)[0]}_chunk_{idx+1}.txt"
            output_path = os.path.join(processed_path, filename_out)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(chunk)