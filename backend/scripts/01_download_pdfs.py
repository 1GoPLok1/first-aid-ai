import os
import sys
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "medical-docs")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

LOCAL_DIR = Path("./data/raw")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_pdf.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def create_s3_client() -> boto3.client:
    try:
        client = boto3.client(
            "s3",
            endpoint_url=f"{'https' if MINIO_SECURE else 'http'}://{MINIO_ENDPOINT}",
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            verify=MINIO_SECURE,
        )
        # Проверка подключения
        client.list_buckets()
        logger.info("✅ Подключение к MinIO установлено: %s", MINIO_ENDPOINT)
        return client
    except NoCredentialsError:
        logger.error("❌ Неверные Access Key / Secret Key. Проверьте .env файл.")
        sys.exit(1)
    except EndpointConnectionError:
        logger.error("❌ Не удалось подключиться к MinIO по адресу: %s. Проверьте, запущен ли сервер.", MINIO_ENDPOINT)
        sys.exit(1)

def ensure_bucket_exists(client: boto3.client, bucket_name: str) -> None:
    try:
        client.head_bucket(Bucket=bucket_name)
        logger.info("📦 Бакет '%s' найден.", bucket_name)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code == "404":
            logger.error("❌ Бакет '%s' не существует. Создайте его в веб-консоли MinIO (http://localhost:9001).", bucket_name)
            sys.exit(1)
        else:
            logger.error("❌ Ошибка доступа к бакету '%s': %s", bucket_name, exc)
            sys.exit(1)

def download_pdfs(client: boto3.client, bucket_name: str, local_dir: Path) -> None:
    # Создаём целевую папку, если её нет
    local_dir.mkdir(parents=True, exist_ok=True)

    # Получаем список всех объектов в бакете
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name)

    total_files = 0
    downloaded = 0
    skipped = 0
    failed = 0

    for page in pages:
        if "Contents" not in page:
            logger.warning("⚠️  Бакет '%s' пуст.", bucket_name)
            return

        for obj in page["Contents"]:
            object_name = obj["Key"]

            # Фильтруем только PDF
            if not object_name.lower().endswith(".pdf"):
                logger.debug("⏭️  Пропущен (не PDF): %s", object_name)
                continue

            total_files += 1
            local_path = local_dir / object_name

            # Создаём подпапки, если файл внутри "директории" в бакете
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Пропускаем, если файл уже скачан
            if local_path.exists():
                logger.info("⏭️  Уже существует: %s", object_name)
                skipped += 1
                continue

            # Скачиваем файл
            try:
                client.download_file(bucket_name, object_name, str(local_path))
                file_size_kb = round(obj["Size"] / 1024, 1)
                logger.info("📥 Скачан: %s (%s КБ)", object_name, file_size_kb)
                downloaded += 1
            except ClientError as exc:
                logger.error("❌ Ошибка загрузки %s: %s", object_name, exc)
                failed += 1

    logger.info("=" * 50)
    logger.info("📊 Статистика загрузки:")
    logger.info("   Всего PDF в бакете: %d", total_files)
    logger.info("   Успешно скачано:    %d", downloaded)
    logger.info("   Пропущено (сущ.):   %d", skipped)
    logger.info("   Ошибок:             %d", failed)
    logger.info("   Файлы сохранены в:  %s", local_dir.resolve())
    logger.info("=" * 50)

def main() -> None:
    """Основная функция запуска скрипта."""
    logger.info("🚀 Запуск загрузки PDF из MinIO...")
    logger.info("   Эндпоинт: %s", MINIO_ENDPOINT)
    logger.info("   Бакет:    %s", MINIO_BUCKET)
    logger.info("   Папка:    %s", LOCAL_DIR.resolve())

    s3_client = create_s3_client()
    ensure_bucket_exists(s3_client, MINIO_BUCKET)
    download_pdfs(s3_client, MINIO_BUCKET, LOCAL_DIR)

    logger.info("✅ Загрузка завершена.")


if __name__ == "__main__":
    main()