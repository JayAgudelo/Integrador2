import requests

def get_reccobeats_track_features(track_id):
    url = f"https://api.reccobeats.com/v1/track/{track_id}/audio-features"
    # No usamos API key (según lo que encontramos)
    response = requests.get(url)
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None
    return response.json()

if __name__ == "__main__":
    # Aquí pones el ID que usarás en ReccoBeats para la canción
    track_id = "2670c328-c40f-45f4-80df-f48b29296deb"  # ejemplo: ID de Spotify de la canción
    features = get_reccobeats_track_features(track_id)
    if features:
        print("Features obtenidos:")
        for k, v in features.items():
            print(f"{k}: {v}")
    else:
        print("No se obtuvieron datos para el track.")
