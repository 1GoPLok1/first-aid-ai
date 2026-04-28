from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional

app = FastAPI()

@app.post("/process")
async def process_audio_or_text(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    answer = ""
    if audio:
        # Здесь можно сохранить файл и выполнить распознавание голоса
        contents = await audio.read()
        
        return {"message": "Голосовое сообщение получено"}
    elif text:
        # Обработка текстового запроса
        print("Get text:" + {text})
        return {"message": f"Получен текст: {answer}"}
    else:
        return {"message": "Нет данных"}