import sys
import importlib


def check_package(package_name):
    """Проверяет, установлен ли пакет"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def main():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("\nПроверка пакетов:")

    packages = [
        "langchain",
        "chromadb",
        "sentence_transformers",
        "torch",
        "openai",
        "pypdf",
        "beautifulsoup4",
        "dotenv",
        "fastapi",
        "aiogram"
    ]

    all_ok = True
    for package in packages:
        installed = check_package(package)
        status = "✅" if installed else "❌"
        print(f"  {status} {package}")
        if not installed:
            all_ok = False

    # Проверка переменных окружения
    from dotenv import load_dotenv
    import os

    load_dotenv()
    print("\nПроверка переменных окружения:")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "sk-xxxxxx":
        print(f"  ✅ OPENAI_API_KEY настроен")
    else:
        print(f"  ⚠️ OPENAI_API_KEY не настроен (нужен для работы с OpenAI)")

    if all_ok:
        print("\n✅ Все базовые пакеты установлены корректно!")
    else:
        print("\n❌ Некоторые пакеты отсутствуют. Установите их через pip install -r requirements.txt")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())