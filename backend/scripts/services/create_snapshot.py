import requests
import backend.scripts.config as config

response = requests.post(
    f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/collections/{config.COLLECTION_NAME}/snapshots"
)

if response.status_code == 200:
    snapshot_info = response.json()
    print(f"Снапшот создан: {snapshot_info}")
else:
    print(f"Ошибка создания снапшота: {response.status_code}")