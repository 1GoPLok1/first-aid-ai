from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
from scripts.answer_processing import gen_answer as GenAnswer, get_prompt as GetPrompt

app = FastAPI()

@app.post("/process")
async def process_audio_or_text(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    if not isinstance(audio, None) or not isinstance(text, None):
        answer = ""
        if audio:
            contents = await audio.read()
            user_prompt = GetPrompt.get_prompt(contents)
        elif text:
            user_prompt = text
        ##Connect to qdrant

        ##Gen answer
        answer = GenAnswer.gen_answer(user_prompt)
        
        return answer
    else:
        return {"message": "Нет данных"}