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

# Obtener datos de artista
def get_artist_data(artist_id, token):
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Ejemplo de uso
# -----------------------------
if __name__ == "__main__":
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    print("Token generado correctamente.")

    # Ejemplo con el artista Radiohead
    artist_id = "4Z8W4fKeB5YxbusRsdQVPb"
    artist_data = get_artist_data(artist_id, token)

    print("Nombre del artista:", artist_data["name"])
    print("Popularidad:", artist_data["popularity"])
    print("Géneros:", artist_data["genres"])
    print("Seguidores:", artist_data["followers"]["total"])
