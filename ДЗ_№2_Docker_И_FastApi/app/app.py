from fastapi import FastAPI
from pydantic import BaseModel

from ml.model import load_model

model = None
app = FastAPI()

# формат ответа модели
class PapResponse(BaseModel):
    text: str
    pap_label: str
    pap_score: float


@app.get("/")
def index():
    return {"text": "Pap Classification"}


# загрузка модели во время старта приложения
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


# GET-запрос для получения предсказания по заданному запросу
@app.get("/predict")
def predict_pap(text: str):
    pap = model(text)

    response = PapResponse(
        text=text,
        pap_label=pap.label,
        pap_score=pap.score,
    )

    return response
