param(
    [string]$BackupPath = "C:\backups\qdrant",
    [string]$CollectionName = "first_aid_knowledge_base"
)

# Создаем папку для бэкапов, если её нет
if (!(Test-Path $BackupPath)) {
    New-Item -ItemType Directory -Path $BackupPath -Force
}

# Формируем имя файла с датой
$dateStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$snapshotFile = "$BackupPath\qdrant_snapshot_$dateStamp.tar.gz"

Write-Host "Создание снапшота коллекции $CollectionName..." -ForegroundColor Green

# Создаем снапшот через API
$body = @{}
$jsonBody = $body | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "http://localhost:6333/collections/$CollectionName/snapshots" `
                                  -Method Post `
                                  -ContentType "application/json" `
                                  -Body $jsonBody

    if ($response.result) {
        $snapshotName = $response.result.name
        Write-Host "Снапшот создан: $snapshotName" -ForegroundColor Green

        # Копируем снапшот из контейнера
        docker cp "qdrant_first_aid:/qdrant/storage/snapshots/$snapshotName" "$snapshotFile"

        Write-Host "Снапшот сохранен: $snapshotFile" -ForegroundColor Green

        # Очищаем старые снапшоты в контейнере (оставляем только последний)
        docker exec qdrant_first_aid rm -rf /qdrant/storage/snapshots/*
    }
}
catch {
    Write-Host "Ошибка создания снапшота: $_" -ForegroundColor Red
}