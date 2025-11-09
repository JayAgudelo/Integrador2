import React, { useState } from "react";
import confetti from "canvas-confetti";

interface Features {
  acousticness: number;
  danceability: number;
  energy: number;
  instrumentalness: number;
  liveness: number;
  loudness: number;
  mode: number;
  speechiness: number;
  tempo: number;
  time_signature: number;
  valence: number;
  duration_ms: number;
  key: number;
  genre: string;
}

// Configuración de sliders
const featureSliders: { key: keyof Features; label: string; min: number; max: number; step?: number; percentage?: boolean }[] = [
  { key: "acousticness", label: "Acousticness", min: 0, max: 1, step: 0.01, percentage: true },
  { key: "danceability", label: "Danceability", min: 0, max: 1, step: 0.01, percentage: true },
  { key: "energy", label: "Energy", min: 0, max: 1, step: 0.01, percentage: true },
  { key: "instrumentalness", label: "Instrumentalness", min: 0, max: 1, step: 0.01, percentage: true },
  { key: "liveness", label: "Liveness", min: 0, max: 1, step: 0.01, percentage: true },
  { key: "loudness", label: "Loudness", min: -60, max: 0, step: 0.1 },
  { key: "mode", label: "Mode", min: 0, max: 1, step: 1 },
  { key: "speechiness", label: "Speechiness", min: 0, max: 1, step: 0.01, percentage: true },
  { key: "tempo", label: "Tempo", min: 60, max: 200, step: 1 },
  { key: "time_signature", label: "Time Signature", min: 3, max: 5, step: 1 },
  { key: "valence", label: "Valence", min: 0, max: 1, step: 0.01, percentage: true },
  { key: "duration_ms", label: "Duration (ms)", min: 60000, max: 600000, step: 1000 },
  { key: "key", label: "Key", min: 0, max: 11, step: 1 },
];

// Lista de géneros completos
const genreOptions = [
  "genre_alt-rock", "genre_blues", "genre_chill", "genre_classical",
  "genre_country", "genre_dance", "genre_edm", "genre_electro",
  "genre_electronic", "genre_emo", "genre_folk", "genre_french",
  "genre_funk", "genre_german", "genre_hard-rock", "genre_hardcore",
  "genre_hip-hop", "genre_house", "genre_indie-pop", "genre_jazz",
  "genre_k-pop", "genre_metal", "genre_pop", "genre_punk",
  "genre_rock", "genre_sad", "genre_sertanejo", "genre_singer-songwriter",
  "genre_soul", "genre_spanish"
];

export default function SongOptimizerWizard(): JSX.Element {
  const [features, setFeatures] = useState<Features>({
    acousticness: 0.5,
    danceability: 0.5,
    energy: 0.5,
    instrumentalness: 0,
    liveness: 0,
    loudness: -20,
    mode: 1,
    speechiness: 0,
    tempo: 120,
    time_signature: 4,
    valence: 0.5,
    duration_ms: 180000,
    key: 0,
    genre: "alt-rock", // solo la parte real
  });

  const [locks, setLocks] = useState<Record<keyof Features, boolean>>(
    Object.fromEntries(Object.keys(features).map((k) => [k, k === "genre"])) as Record<keyof Features, boolean>
  );

  const [predictionResult, setPredictionResult] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Cambiar valor de slider
  const handleSliderChange = (key: keyof Features, value: number) => {
    setFeatures((prev) => ({ ...prev, [key]: value }));
  };

  // Toggle lock
  const toggleLock = (key: keyof Features) => {
    setLocks((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Cambio de género
  const handleGenreChange = (value: string) => {
    setFeatures((prev) => ({ ...prev, genre: value }));
  };

  // Optimizar
  const handleOptimize = async () => {
    setLoading(true);
    try {
      const payload = {
        features: features,
        locked_features: Object.entries(locks)
          .filter(([_, locked]) => locked)
          .map(([k]) => k)
      };

      const response = await fetch("http://localhost:8000/wizard-optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (data.predicted_popularity !== undefined) {
        setPredictionResult(data.predicted_popularity);
        if (data.predicted_popularity > 40) {
          confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 } });
        }
      }

      if (data.optimized_features) {
        const roundedFeatures = {
            ...data.optimized_features,
            duration_ms: Math.round(data.optimized_features.duration_ms),
        };
        setFeatures(roundedFeatures);
      }

    } catch (err) {
      console.error("Error optimizing:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex gap-6 p-6">
      {/* Formulario */}
      <div className="relative flex-1 bg-white p-6 rounded-xl shadow space-y-4">
        {loading && (
          <div className="absolute inset-0 bg-white bg-opacity-70 flex items-center justify-center z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-blue-600 border-b-4"></div>
          </div>
        )}

        <h2 className="text-lg font-bold mb-4">Optimizar canción</h2>

        {featureSliders.map((f) => (
          <div key={f.key} className="flex items-center gap-3">
            <label className="w-40">{f.label}</label>
            <input
              type="range"
              min={f.min}
              max={f.max}
              step={f.step || 0.01}
              value={features[f.key]}
              onChange={(e) => handleSliderChange(f.key, Number(e.target.value))}
              className="flex-1"
            />
            <span className="w-12">{f.percentage ? `${(Number(features[f.key]) * 100).toFixed(0)}%` : features[f.key]}</span>
            <button onClick={() => toggleLock(f.key)} className="ml-2">
              {locks[f.key] ? "🔒" : "🔓"}
            </button>
          </div>
        ))}

        {/* Dropdown de género */}
        <div className="flex items-center gap-3 mt-4">
          <label className="w-40 font-medium">Género</label>
          <select
            value={`genre_${features.genre}`}
            onChange={(e) => handleGenreChange(e.target.value.replace("genre_", ""))}
            className="flex-1 border rounded px-2 py-1 bg-white focus:outline-none focus:ring-2 focus:ring-blue-300"
          >
            {genreOptions.map((g) => (
              <option key={g} value={g}>
                {g.replace("genre_", "").replace("-", " ")}
              </option>
            ))}
          </select>
          <span className="ml-2">🔒</span>
        </div>

        <button
          onClick={handleOptimize}
          className="mt-4 w-full bg-blue-600 text-white py-2 rounded font-semibold"
          disabled={loading}
        >
          Optimizar
        </button>
      </div>

      {/* Resultados */}
      <div className="w-1/3 bg-white p-6 rounded-xl shadow flex flex-col items-center justify-center">
        <h2 className="text-lg font-bold mb-4">Predicción</h2>
        <div className="text-5xl font-bold mb-4">
          {predictionResult !== null ? predictionResult.toFixed(0) : "--"}
        </div>
        <details className="mt-4 w-full">
          <summary className="cursor-pointer text-blue-600 font-semibold">Parámetros optimizados</summary>
          <pre className="text-xs mt-2 overflow-auto max-h-64">{JSON.stringify(features, null, 2)}</pre>
        </details>
      </div>
    </div>
  );
}
