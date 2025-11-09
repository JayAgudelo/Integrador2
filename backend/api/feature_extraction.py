import requests
import librosa
import numpy as np
import pandas as pd
def get_complete_features(audio_file_path):
    """
    Combine ReccoBeats API + librosa to get ALL features your model needs
    """
    features = {}
    
    # === Part 1: Get features from ReccoBeats ===
    print("Extracting features from ReccoBeats...")
    url = "https://api.reccobeats.com/v1/analysis/audio-features"
    
    with open(audio_file_path, 'rb') as f:
        files = {'audioFile': f}
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            recco_features = response.json()
            # Add ReccoBeats features
            features.update(recco_features)
        else:
            print(f"ReccoBeats error: {response.status_code}")
            return None
    
    # === Part 2: Get missing features from librosa ===
    print("Extracting missing features from librosa...")
    y, sr = librosa.load(audio_file_path)
    
    # Duration in milliseconds
    features['duration_ms'] = int(librosa.get_duration(y=y, sr=sr) * 1000)
    
    # Key detection (-1 to 11, where -1 = no key detected, 0=C, 1=C#, etc.)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = int(np.argmax(np.sum(chroma, axis=1)))
        features['key'] = key  # 0-11
    except:
        features['key'] = -1  # No key detected
    
    # Mode detection (0=minor, 1=major)
    # Using harmonic-percussive separation for better estimation
    try:
        harmonic, _ = librosa.effects.hpss(y)
        chroma_harmonic = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
        
        # Simplified major/minor detection based on chroma distribution
        # Major keys tend to have stronger 3rd and 5th intervals
        chroma_mean = np.mean(chroma_harmonic, axis=1)
        major_profile = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # Major scale pattern
        minor_profile = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # Minor scale pattern
        
        major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
        
        features['mode'] = 1 if major_corr > minor_corr else 0
    except:
        features['mode'] = 1  # Default to major
    
    # Time signature (usually 4/4)
    # Proper detection is very complex, most popular music is 4/4
    features['time_signature'] = 4
    
    # Genre - Can't be detected from audio alone
    features['genre'] = None  # or "unknown" or let user provide
    
    return features