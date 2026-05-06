import pypdf
import pdfplumber
from tqdm import tqdm

def extract_from_pdf(pdf_path, method='pypdf'):
    """Извлечение текста из PDF разными методами"""
    
    if method == 'pypdf':
        reader = pypdf.PdfReader(pdf_path)
        text = []
        for page in tqdm(reader.pages, desc=f"Обработка {pdf_path}"):
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return '\n'.join(text)
    
    elif method == 'pdfplumber':
        # pdfplumber лучше работает со сложной версткой и таблицами
        with pdfplumber.open(pdf_path) as pdf:
            text = []
            for page in tqdm(pdf.pages, desc=f"Обработка {pdf_path}"):
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return '\n'.join(text)
    
    else:
        raise ValueError(f"Неизвестный метод: {method}")

# Пакетная обработка всех PDF в папке
import os
from glob import glob

pdf_files = glob('data/raw/*.pdf')
for pdf_path in pdf_files:
    try:
        text = extract_from_pdf(pdf_path, method='pdfplumber')
        # Сохраняем как txt для дальнейшей обработки
        txt_path = pdf_path.replace('.pdf', '.txt').replace('raw', 'processed')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✅ Обработан: {pdf_path}")
    except Exception as e:
        print(f"❌ Ошибка {pdf_path}: {e}")