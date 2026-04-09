import os
import kagglehub
import pandas as pd

def get_spotify_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    temp_folder = os.path.join(project_root, "temp")
    csv_file_path = os.path.join(temp_folder, "database_raw.csv")

    if os.path.exists(csv_file_path):
        print(f"Archivo encontrado en: {csv_file_path}")
        print("Cargando desde archivo local...")
        df = pd.read_csv(csv_file_path)
    else:
        print(f"Archivo no encontrado en: {csv_file_path}")
        print("Descargando desde Kaggle...")
        
        os.makedirs(temp_folder, exist_ok=True)
        
        # Descargar el dataset completo
        download_path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks")
        original_file = os.path.join(download_path, "spotify_data.csv")
        
        # Intentar diferentes encodings
        encodings = ['latin-1', 'ISO-8859-1', 'cp1252', 'utf-16']
        df = None
        
        for encoding in encodings:
            try:
                print(f"Intentando con encoding: {encoding}")
                df = pd.read_csv(original_file, encoding=encoding)
                print(f"¡Éxito con encoding: {encoding}!")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("No se pudo leer el archivo con ningún encoding")
        
        # Save the dataframe to the temp folder
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        print(f"Archivo guardado en: {csv_file_path}")

    return df