import requests
import config
import os

snapshot_path = "C:\\backups\\qdrant\\qdrant_snapshot_20240317_020000.tar.gz"
snapshot_filename = os.path.basename(snapshot_path)

os.system(f'docker cp "{snapshot_path}" qdrant_first_aid:/qdrant/storage/{snapshot_filename}')

response = requests.post(
    f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/collections/{config.COLLECTION_NAME}/snapshots/recover",
    json={"location": f"/qdrant/storage/{snapshot_filename}"}
)

if response.status_code == 200:
    print("Коллекция успешно восстановлена из снапшота.")
else:
    print(f"Ошибка восстановления: {response.status_code} - {response.text}")