import os
import pdfplumber

input_dir = r'data\raw'
output_dir = r'data\processed'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(input_dir, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"

        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

        print(f'Обработан: {filename} -> {txt_filename}')