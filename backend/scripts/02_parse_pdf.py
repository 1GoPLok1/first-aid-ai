import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

PDF_SOURCE_DIR = Path("./data/raw")
PARSED_OUTPUT_DIR = Path("./data/parsed")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("parse_pdfs.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

def import_parsing_libraries():
    """Проверяет наличие и импортирует библиотеки парсинга."""
    libs = {}
    
    # PyMuPDF (fitz) — базовое извлечение текста
    try:
        import fitz
        libs["fitz"] = fitz
        logger.info("✅ PyMuPDF (fitz) загружен")
    except ImportError:
        logger.error("❌ PyMuPDF не установлен. Установите: pip install PyMuPDF")
        sys.exit(1)
    
    # pymupdf4llm — конвертация в Markdown с сохранением структуры
    try:
        import pymupdf4llm
        libs["pymupdf4llm"] = pymupdf4llm
        logger.info("✅ pymupdf4llm загружен")
    except ImportError:
        logger.warning("⚠️  pymupdf4llm не установлен. Будет использован базовый парсинг.")
        logger.warning("   Установите: pip install pymupdf4llm")
        libs["pymupdf4llm"] = None
    
    # pdfplumber — извлечение таблиц
    try:
        import pdfplumber
        libs["pdfplumber"] = pdfplumber
        logger.info("✅ pdfplumber загружен")
    except ImportError:
        logger.warning("⚠️  pdfplumber не установлен. Таблицы извлекаться не будут.")
        logger.warning("   Установите: pip install pdfplumber")
        libs["pdfplumber"] = None
    
    # pytesseract — OCR для сканов
    try:
        import pytesseract
        from pdf2image import convert_from_path
        libs["pytesseract"] = pytesseract
        libs["pdf2image"] = convert_from_path
        logger.info("✅ pytesseract + pdf2image загружены")
    except ImportError:
        logger.warning("⚠️  OCR-библиотеки не установлены. Сканы PDF не будут распознаны.")
        logger.warning("   Установите: pip install pytesseract pdf2image Pillow")
        libs["pytesseract"] = None
        libs["pdf2image"] = None
    
    return libs

def has_text_layer(pdf_path: Path, fitz) -> bool:
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page_num in range(min(3, doc.page_count)):
            text += doc[page_num].get_text()
        doc.close()
        return len(text.strip()) > 50
    except Exception as exc:
        logger.warning("⚠️  Ошибка проверки текстового слоя: %s", exc)
        return True  # При ошибке считаем, что слой есть

def ocr_pdf(input_path: Path, output_path: Path, libs: dict) -> Optional[Path]:
    if libs["pytesseract"] is None or libs["pdf2image"] is None:
        logger.error("❌ OCR невозможен: библиотеки не установлены.")
        return None
    
    try:
        from PIL import Image
        
        logger.info("🔍 Запуск OCR для: %s", input_path.name)
        images = libs["pdf2image"](str(input_path), dpi=300)
        logger.info("   📄 Страниц для OCR: %d", len(images))
        
        fitz = libs["fitz"]
        doc = fitz.open()
        
        for i, image in enumerate(images, start=1):
            text = libs["pytesseract"].image_to_string(image, lang="rus")
            
            # Создаём страницу
            img_bytes = image.tobytes("ppm")
            img_pixmap = fitz.Pixmap(img_bytes)
            page = doc.new_page(width=img_pixmap.width, height=img_pixmap.height)
            page.insert_image(page.rect, pixmap=img_pixmap)
            
            # Вставляем распознанный текст
            page.insert_text(
                fitz.Point(0, 0),
                text,
                fontsize=1,
                color=(0.99, 0.99, 0.99),
            )
            
            if i % 5 == 0:
                logger.info("   ⏳ Обработано: %d/%d", i, len(images))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        doc.close()
        
        logger.info("   ✅ OCR завершён: %s", output_path.name)
        return output_path
        
    except Exception as exc:
        logger.error("❌ Ошибка OCR: %s", exc)
        return None

def extract_tables(pdf_path: Path, pdfplumber) -> List[Dict[str, Any]]:
    tables = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables):
                    if table:
                        tables.append({
                            "page": page_num,
                            "table_index": table_idx,
                            "rows": len(table),
                            "cols": len(table[0]) if table else 0,
                            "data": table,
                        })
    except Exception as exc:
        logger.warning("⚠️  Ошибка извлечения таблиц: %s", exc)
    return tables

def analyze_structure(markdown_text: str) -> Dict[str, Any]:
    lines = markdown_text.split("\n")
    headers = []
    tables_count = 0
    text_blocks_count = 0
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Заголовки H1-H3
        if stripped.startswith("# ") or stripped.startswith("## ") or stripped.startswith("### "):
            level = 1 if stripped.startswith("# ") else (2 if stripped.startswith("## ") else 3)
            headers.append({"level": level, "text": stripped.lstrip("#").strip()})
            in_table = False

        # Таблицы
        elif "|" in stripped and "---" in stripped:
            tables_count += 1
            in_table = True

        # Текстовые блоки
        elif stripped and not in_table:
            text_blocks_count += 1
        elif not stripped:
            in_table = False

    return {
        "headers": headers,
        "tables_count": tables_count,
        "text_blocks_count": text_blocks_count,
        "headers_count": len(headers),
    }

def parse_pdfs(source_dir: Path, output_dir: Path, libs: dict) -> None:
    if not source_dir.exists():
        logger.error("❌ Папка '%s' не существует.", source_dir.resolve())
        sys.exit(1)

    pdf_files = list(source_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("❌ PDF-файлы не найдены в '%s'.", source_dir.resolve())
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("📄 Найдено PDF-файлов: %d", len(pdf_files))

    total_text_blocks = 0
    total_tables = 0
    processed = 0
    failed = 0

    for pdf_path in pdf_files:
        logger.info("-" * 50)
        logger.info("🔍 Обработка: %s", pdf_path.name)

        try:
            file_to_parse = pdf_path

            # Проверка текстового слоя
            if not has_text_layer(pdf_path, libs["fitz"]):
                logger.warning("   ⚠️  Текстовый слой отсутствует. Запуск OCR...")
                ocr_result = ocr_pdf(
                    pdf_path,
                    Path("./data/ocr_temp") / f"ocr_{pdf_path.name}",
                    libs,
                )
                if ocr_result:
                    file_to_parse = ocr_result
                else:
                    logger.error("   ❌ OCR не удался, пропускаем файл.")
                    failed += 1
                    continue

            # Парсинг в Markdown
            if libs["pymupdf4llm"] is not None:
                logger.info("   📝 Парсинг через pymupdf4llm...")
                full_markdown = libs["pymupdf4llm"].to_markdown(str(file_to_parse))
            else:
                logger.info("   📝 Базовое извлечение текста через PyMuPDF...")
                doc = libs["fitz"].open(str(file_to_parse))
                full_markdown = ""
                for page in doc:
                    full_markdown += page.get_text() + "\n\n---\n\n"
                doc.close()

            # Извлечение таблиц
            tables = []
            if libs["pdfplumber"] is not None:
                logger.info("   📊 Извлечение таблиц...")
                tables = extract_tables(file_to_parse, libs["pdfplumber"])
            else:
                logger.info("   ⚠️  pdfplumber не доступен, таблицы пропущены.")

            # Анализ структуры
            structure = analyze_structure(full_markdown)
            structure["tables_count"] = max(structure["tables_count"], len(tables))
            page_count = len(full_markdown.split("\n---\n"))
            structure["pages_estimate"] = max(page_count, 1)

            # Сохранение JSON
            output_data = {
                "filename": pdf_path.name,
                "parsed_filename": file_to_parse.name,
                "had_text_layer": True,
                "pages_estimate": structure["pages_estimate"],
                "headers": structure["headers"],
                "tables_count": structure["tables_count"],
                "text_blocks_count": structure["text_blocks_count"],
                "tables_data": tables,
                "markdown_preview": full_markdown[:500] + "...",
            }

            json_path = output_dir / f"{pdf_path.stem}_parsed.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            # Сохранение Markdown
            md_path = output_dir / f"{pdf_path.stem}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(full_markdown)

            logger.info("   📊 Результаты:")
            logger.info("      Страниц (оценка): %d", structure["pages_estimate"])
            logger.info("      Текстовых блоков: %d", structure["text_blocks_count"])
            logger.info("      Таблиц:           %d", structure["tables_count"])
            logger.info("      Заголовков:       %d", structure["headers_count"])
            logger.info("   💾 JSON:     %s", json_path.name)
            logger.info("   💾 Markdown: %s", md_path.name)

            total_text_blocks += structure["text_blocks_count"]
            total_tables += structure["tables_count"]
            processed += 1

        except Exception as exc:
            logger.error("   ❌ Ошибка: %s", exc, exc_info=True)
            failed += 1

    # Итоги
    logger.info("=" * 50)
    logger.info("📊 Итоговая статистика:")
    logger.info("   Всего PDF:          %d", len(pdf_files))
    logger.info("   Успешно:            %d", processed)
    logger.info("   Ошибок:             %d", failed)
    logger.info("   Текстовых блоков:   %d", total_text_blocks)
    logger.info("   Таблиц:             %d", total_tables)
    logger.info("   Папка результатов:  %s", output_dir.resolve())
    logger.info("=" * 50)

def main() -> None:
    """Основная функция."""
    logger.info("🚀 Запуск ЛОКАЛЬНОГО парсинга PDF...")
    logger.info("   Исходная папка: %s", PDF_SOURCE_DIR.resolve())
    logger.info("   Выходная папка: %s", PARSED_OUTPUT_DIR.resolve())
    logger.info("   ⚠️  LlamaCloud НЕ используется (недоступен в РФ)")

    libs = import_parsing_libraries()
    parse_pdfs(PDF_SOURCE_DIR, PARSED_OUTPUT_DIR, libs)

    logger.info("✅ Парсинг завершён.")


if __name__ == "__main__":
    main()