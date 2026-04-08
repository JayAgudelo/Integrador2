from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import joblib
import os
import time
import logging
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

from backend.api.model import prediction
from backend.api.feature_extraction import get_complete_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading for model
_model = None
_preprocessor_route = "backend/api/models/preprocessor.joblib"

def get_model():
    global _model
    if _model is None:
        try:
            logging.info("Loading model...")
            _model = joblib.load("backend/api/models/model.joblib")
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    return _model

RECCO_BASE = "https://api.reccobeats.com/v1"


def error_response(stage: str, message: str, status_code: int = 400, detail: str | None = None):
    payload = {"error": message, "stage": stage}
    if detail:
        payload["detail"] = detail
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/predict")
async def predict_from_features(features: dict):
    try:
        model = get_model()
        popularity_pred = prediction(_preprocessor_route, model, features)
        return {"prediction": popularity_pred, "features": features}
    except Exception as exc:
        logging.error(f"Prediction failed: {exc}")
        return error_response("predict", "Prediction failed for the provided feature set.", 500, str(exc))


@app.post("/extract-features")
async def extract_features(audio: UploadFile = File(...), genre: str | None = Form(None)):
    audio_path = f"temp_{audio.filename}"

    try:
        with open(audio_path, "wb") as temp_audio:
            temp_audio.write(await audio.read())

        features = get_complete_features(audio_path)

        if not features:
            return error_response("extract", "Failed to extract features from the uploaded audio.", 422)

        if genre:
            features["genre"] = genre

        return {"features": features}
    except Exception as exc:
        return error_response("extract", "Unexpected error while extracting audio features.", 500, str(exc))
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.get("/search-track")
def search_track(ids: str = Query(..., description="IDs separados por coma")):
    id_list = ids.split(",")
    params = [("ids", item) for item in id_list]

    try:
        response = requests.get(RECCO_BASE + "/audio-features", params=params)
    except requests.RequestException as exc:
        return error_response("search-track", "Spotify lookup service is unavailable.", 502, str(exc))

    if response.status_code != 200:
        return error_response("search-track", "Spotify lookup failed.", response.status_code, response.text)

    data = response.json()

    if not data.get("content") or len(data["content"]) == 0:
        return error_response(
            "search-track",
            "Track ID not found. Verify the Spotify track ID or try the upload route.",
            404,
        )

    track_features = data["content"][0]
    return track_features.get("id")


@app.get("/track-features")
def track_features(
    track_id: str = Query(..., description="ID de ReccoBeats"),
    genre: str | None = Query(None, description="Genero opcional"),
):
    try:
        response = requests.get(f"{RECCO_BASE}/track/{track_id}/audio-features")
    except requests.RequestException as exc:
        return error_response("track-features", "Track feature service is unavailable.", 502, str(exc))

    if response.status_code != 200:
        return error_response("track-features", "Track feature lookup failed.", response.status_code, response.text)

    features = response.json()
    features.setdefault("time_signature", 4)
    features.setdefault("genre", genre if genre else "unknown")

    try:
        duration_ms = requests.get(f"{RECCO_BASE}/track/{track_id}").json().get("durationMs")
    except requests.RequestException as exc:
        return error_response("track-features", "Track duration lookup failed.", 502, str(exc))

    features.setdefault("duration_ms", duration_ms)

    model_features = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "key",
        "liveness",
        "loudness",
        "mode",
        "speechiness",
        "tempo",
        "valence",
        "time_signature",
        "genre",
        "duration_ms",
    ]
    filtered_features = {key: features[key] for key in model_features if key in features}

    if not filtered_features:
        return error_response("track-features", "Track features could not be mapped to the model schema.", 422)

    return filtered_features


class WizardRequest(BaseModel):
    features: dict
    locked_features: list[str] = []


class BeatGenerationRequest(BaseModel):
    features: dict
    duration_seconds: int = 15
    prompt_text: str | None = None


load_dotenv(dotenv_path="backend/.env")
BEATOVEN_API_KEY = os.environ.get("BEATOVEN_API_KEY")
BEATOVEN_BASE_URL = "https://public-api.beatoven.ai"


def build_beatoven_prompt(features: dict, duration_seconds: int, prompt_text: str | None = None) -> str:
    if prompt_text:
        return prompt_text

    genre = features.get("genre") or "modern"
    tempo = int(features.get("tempo", 120))
    energy = float(features.get("energy", 0.5))
    danceability = float(features.get("danceability", 0.5))
    valence = float(features.get("valence", 0.5))
    loudness = float(features.get("loudness", -12))

    energy_text = "high-energy" if energy >= 0.65 else "moderate-energy" if energy >= 0.4 else "subtle"
    dance_text = "danceable" if danceability >= 0.55 else "groovy" if danceability >= 0.35 else "laid-back"
    mood_text = "bright and uplifting" if valence >= 0.6 else "warm and moody" if valence >= 0.35 else "dark and atmospheric"
    loudness_text = "punchy" if loudness >= -12 else "thick" if loudness >= -24 else "soft"

    return (
        f"{duration_seconds} second instrumental {genre} beat base, tempo {tempo} BPM, {energy_text}, "
        f"{dance_text}, {mood_text}, {loudness_text} drums and bass. No vocals, only the instrumental foundation."
    )


def compose_beatoven_track(prompt_text: str, duration_seconds: int = 15, output_format: str = "mp3") -> str:
    if not BEATOVEN_API_KEY:
        raise ValueError("Beatoven API key is not configured in the backend environment.")

    headers = {
        "Authorization": f"Bearer {BEATOVEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": {"text": f"{duration_seconds} seconds {prompt_text}"},
        "format": output_format,
        "looping": False,
    }

    response = requests.post(f"{BEATOVEN_BASE_URL}/api/v1/tracks/compose", json=payload, headers=headers)
    response.raise_for_status()
    compose_result = response.json()
    task_id = compose_result.get("task_id")

    if not task_id:
        raise RuntimeError(f"Failed to get task_id from compose response: {compose_result}")

    status = "composing"
    status_response = {}
    for _ in range(12):
        status_response = requests.get(f"{BEATOVEN_BASE_URL}/api/v1/tasks/{task_id}", headers=headers)
        status_response.raise_for_status()
        status_data = status_response.json()
        status = status_data.get("status")
        if status == "composed":
            return status_data.get("meta", {}).get("track_url")
        if status == "failed":
            raise RuntimeError(f"Beatoven task failed: {status_data}")
        time.sleep(5)

    raise RuntimeError(f"Beatoven task did not complete in time: {status_response.json()}")


@app.post("/wizard-optimize")
def wizard_optimize(request: WizardRequest):
    try:
        model = get_model()
        base_features = request.features
        locked = set(request.locked_features)
        current_best = base_features.copy()
        best_score = prediction(_preprocessor_route, model, current_best)

        integer_features = {"duration_ms", "mode", "key", "time_signature"}
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
            "duration_ms": (60000, 600000),
        }

        for feature, (low, high) in numeric_ranges.items():
            if feature in locked or feature not in current_best:
                continue

            if feature in integer_features:
                test_values = np.round(np.linspace(low, high, min(15, int(high - low + 1)))).astype(int)
                test_values = np.unique(test_values)
            else:
                test_values = np.linspace(low, high, 15)

            for val in test_values:
                candidate = current_best.copy()
                candidate[feature] = int(val) if feature in integer_features else float(val)
                score = prediction(_preprocessor_route, model, candidate)
                if score > best_score:
                    best_score = score
                    current_best = candidate.copy()

        return {
            "optimized_features": current_best,
            "predicted_popularity": round(best_score, 2),
        }
    except Exception as exc:
        logging.error(f"Wizard optimize failed: {exc}")
        return error_response("wizard-optimize", "Optimization failed for the provided feature set.", 500, str(exc))


@app.post("/generate-beat-base")
def generate_beat_base(request: BeatGenerationRequest):
    try:
        prompt_text = build_beatoven_prompt(request.features, request.duration_seconds, request.prompt_text)
        track_url = compose_beatoven_track(prompt_text, request.duration_seconds)

        if not track_url:
            return error_response("generate-beat-base", "Failed to generate AI beat base.", 500)

        return {
            "track_url": track_url,
            "prompt": prompt_text,
        }
    except requests.HTTPError as exc:
        print(exc)
        return error_response("generate-beat-base", "Beatoven API request failed.", 502, str(exc))
    except Exception as exc:
        print(exc)
        return error_response("generate-beat-base", "Beat generation failed.", 500, str(exc))
