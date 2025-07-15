from fastapi import FastAPI
from pydantic import BaseModel
from app.model.predictor import predict_topic_and_level

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/predict")
def predict(request: QueryRequest):
    topic, level = predict_topic_and_level(request.query)
    return {"topic": topic, "knowledge_level": level}

@app.get("/")
def read_root():
    return {"message": "Multi-Task BERT Knowledge Level Predictor is running!"}
