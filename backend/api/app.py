from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
import os
from pydantic import BaseModel
import requests
from backend.api.model import prediction
from backend.api.feature_extraction import get_complete_features

app = FastAPI()

# CORS para permitir frontend local o remoto
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y preprocesador
model = joblib.load("backend/api/models/model.joblib")
preprocessor_route = "backend/api/models/preprocessor.joblib"

# -------------------
# Endpoint: predict desde features
# -------------------
@app.post("/predict")
async def predict_from_features(features: dict):
    """
    Predecir popularidad usando features ya extraidos
    """
    popularity_pred = prediction(preprocessor_route, model, features)
    return {"prediction": popularity_pred, "features": features}

# -------------------
# Endpoint: extraer features desde audio
# -------------------
@app.post("/extract-features")
async def extract_features(
    audio: UploadFile = File(...),
    genre: str | None = Form(None)
):
    """
    Recibir un audio y opcionalmente genero, devolver features
    """
    audio_path = f"temp_{audio.filename}"
    print(genre)
    try:
        # Guardar temporalmente
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        
        features = get_complete_features(audio_path)
        if not features:
            return {"error": "Failed to extract features"}
        
        if genre:
            features["genre"] = genre
        
        return {"features": features}
    finally:
        # Borrar archivo temporal
        if os.path.exists(audio_path):
            os.remove(audio_path)

# -------------------
# Endpoint: search-track (ReccoBeats)
# -------------------
RECCO_BASE = "https://api.reccobeats.com/v1"


@app.get("/search-track")
def search_track(ids: str = Query(..., description="IDs separados por coma")):
    id_list = ids.split(",")
    params = [("ids", i) for i in id_list]
    response = requests.get(RECCO_BASE + "/audio-features", params=params)

    if response.status_code != 200:
        return {"error": f"Error {response.status_code}", "detail": response.text}

    data = response.json()
    print(data)
    # Verificar que content exista y tenga elementos
    if not data.get("content") or len(data["content"]) == 0:
        return {
            "error": "No se ha podido obtener las caracteristicas de esta cancion, intenta con el mp3"
        }

    # Tomar solo el primer track devuelto
    track_features = data["content"][0]

    return track_features.get("id")


# -------------------
# Endpoint: track-features (ReccoBeats)
# -------------------
@app.get("/track-features")
def track_features(track_id: str = Query(..., description="ID de ReccoBeats"), genre: str | None = Query(None, description="Genero opcional")):
    url = f"{RECCO_BASE}/track/{track_id}/audio-features"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": f"Error {response.status_code}", "detail": response.text}

    features = response.json()

    # Completar columnas faltantes con valores por defecto
    features.setdefault("time_signature", 4)
    features.setdefault("genre", genre if genre else "unknown")
    
    url = f"{RECCO_BASE}/track/{track_id}"
    duration_ms = requests.get(url).json().get("durationMs")
    features.setdefault("duration_ms", duration_ms)
    
    # Conservar solo columnas necesarias para el modelo
    model_features = [
        "acousticness", "danceability", "energy", "instrumentalness",
        "key", "liveness", "loudness", "mode", "speechiness",
        "tempo", "valence", "time_signature", "genre", "duration_ms"
    ]
    filtered_features = {k: features[k] for k in model_features if k in features}

    return filtered_features

class WizardRequest(BaseModel):
    features: dict
    locked_features: list[str] = []

@app.post("/wizard-optimize")
def wizard_optimize(request: WizardRequest):
    base_features = request.features
    locked = set(request.locked_features)
    print(base_features)
    current_best = base_features.copy()
    best_score = prediction(preprocessor_route, model, current_best)
    
    # Features que deben ser enteras (solo las categóricas/discretas)
    integer_features = {"duration_ms", "mode", "key", "time_signature"}
    
    # Rango de variación simple para features numéricas
    numeric_ranges = {
        "acousticness": (0, 1),
        "danceability": (0, 1),
        "energy": (0, 1),
        "instrumentalness": (0, 1),
        "key": (0, 11),
        "liveness": (0, 1),
        "loudness": (-60, 0),
        "mode": (0, 1),
        "speechiness": (0, 1),
        "tempo": (60, 200),
        "valence": (0, 1),
        "time_signature": (3, 5),
        "duration_ms": (60000, 600000)
    }
    
    for feature, (low, high) in numeric_ranges.items():
        if feature in locked or feature not in current_best:
            continue
        
        # Generar valores de prueba
        if feature in integer_features:
            # Para enteros, usar linspace y luego redondear, limitando a 15 valores
            test_values = np.round(np.linspace(low, high, min(15, int(high - low + 1)))).astype(int)
            # Eliminar duplicados manteniendo el orden
            test_values = np.unique(test_values)
        else:
            test_values = np.linspace(low, high, 15)
        
        for val in test_values:
            candidate = current_best.copy()
            # Asignar como int o float según corresponda
            candidate[feature] = int(val) if feature in integer_features else float(val)
            score = prediction(preprocessor_route, model, candidate)
            if score > best_score:
                best_score = score
                current_best = candidate.copy()
    
    return {
        "optimized_features": current_best,
        "predicted_popularity": round(best_score, 2)
    }