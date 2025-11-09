from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from backend.api.model import prediction
from feature_extraction import get_complete_features

app = FastAPI()

# CORS para permitir frontend local o remoto
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
model = joblib.load("backend/api/models/model.joblib")
preprocessor = joblib.load("backend/api/models/preprocessor.joblib")
@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    genre: str | None = Form(None)
):
    audio_path = f"temp_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    
    features = get_complete_features(audio_path)
    if not features:
        return {"error": "Failed to extract features"}
    
    if genre:
        features['genre'] = genre

    # Usar tu función de predicción con preprocesador
    popularity_pred = prediction(preprocessor, model, features)
    
    return {"prediction": popularity_pred, "features": features}
