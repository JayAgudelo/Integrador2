import os
import requests
import base64
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Obtener token
def get_access_token(client_id, client_secret):
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode("ascii")
    }
    data = {"grant_type": "client_credentials"}

    r = requests.post(url, headers=headers, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

# Obtener características de una canción
def get_track_audio_features(track_id, token):
    url = f"https://api.spotify.com/v1/audio-features/{track_id}"
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Ejemplo de uso
# -----------------------------
if __name__ == "__main__":
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    print("Token generado correctamente.")

    # Ejemplo con la canción '11dFghVXANMlKmJXsNCbNl' (Spotify API example)
    track_id = "11dFghVXANMlKmJXsNCbNl"
    features = get_track_audio_features(track_id, token)

    print("Características de audio:")
    for k, v in features.items():
        print(f"{k}: {v}")
