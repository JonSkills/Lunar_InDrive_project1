from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

from analyzer import analyze_text

# Загружаем переменные из .env
load_dotenv()

app = FastAPI(title="AI Legal Analyzer")

class TextRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text: str
    issue_type: str
    is_correct: bool
    comment: str = ""

# Если появилась папка static и там есть index.html - отдаем UI на главной
@app.get("/")
def read_root():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"message": "AI Legal Analyzer API is running. Go to /docs for Swagger"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/analyze")
def analyze(request: TextRequest):
    result = analyze_text(request.text)
    return result

@app.post("/api/v1/feedback")
def feedback(request: FeedbackRequest):
    # В реальном проекте здесь будет запись в БД (PostgreSQL/SQLite)
    # Для конкурса сохраняем в локальный JSON файл
    import json
    feedback_file = "feedback.json"
    data = []
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    
    data.append(request.dict())
    
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    return {"status": "success", "message": "Feedback received and saved for training"}

# Подключаем статику после роутов
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")