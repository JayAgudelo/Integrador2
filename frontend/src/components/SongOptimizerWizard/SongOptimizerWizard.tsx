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
  "alt-rock", "blues", "chill", "classical",
  "country", "dance", "edm", "electro",
  "electronic", "emo", "folk", "french",
  "funk", "german", "hard-rock", "hardcore",
  "hip-hop", "house", "indie-pop", "jazz",
  "k-pop", "metal", "pop", "punk",
  "rock", "sad", "sertanejo", "singer-songwriter",
  "soul", "spanish"
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
  <div className="flex justify-center p-6">
    <div className="w-full max-w-6xl flex gap-8">

      {/* FORMULARIO */}
      <div className="relative flex-1 bg-white p-8 rounded-2xl shadow-lg">

        {loading && (
          <div className="absolute inset-0 bg-white/70 flex items-center justify-center z-10 rounded-2xl">
            <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-[#1DB954] border-b-4"></div>
          </div>
        )}

        <h2 className="text-xl font-bold text-gray-900 mb-6">
          Optimizar canción
        </h2>

        <div className="space-y-5">
          {featureSliders.map((f) => (
            <div key={f.key} className="flex items-center gap-4">
              <label className="w-40 text-sm font-medium text-gray-700">
                {f.label}
              </label>

              <input
                type="range"
                min={f.min}
                max={f.max}
                step={f.step || 0.01}
                value={features[f.key]}
                onChange={(e) =>
                  handleSliderChange(f.key, Number(e.target.value))
                }
                className="flex-1 accent-[#1DB954]"
              />

              <span className="w-14 text-sm text-gray-600 text-right">
                {f.percentage
                  ? `${(Number(features[f.key]) * 100).toFixed(0)}%`
                  : features[f.key]}
              </span>

              <button
                onClick={() => toggleLock(f.key)}
                className="text-lg"
              >
                {locks[f.key] ? "🔒" : "🔓"}
              </button>
            </div>
          ))}
        </div>

        {/* Género */}
        <div className="flex items-center gap-4 mt-6">
          <label className="w-40 text-sm font-semibold text-gray-700">
            Género musical
          </label>

          <select
            value={`genre_${features.genre}`}
            onChange={(e) =>
              handleGenreChange(e.target.value.replace("genre_", ""))
            }
            className="flex-1 rounded-lg border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1DB954]"
          >
            {genreOptions.map((g) => (
              <option key={g} value={g}>
                {g.replace("genre_", "").replace("-", " ")}
              </option>
            ))}
          </select>

          <span className="text-lg">🔒</span>
        </div>

        {/* Botón optimizar */}
        <button
          onClick={handleOptimize}
          disabled={loading}
          className="mt-8 w-full bg-[#1DB954] hover:bg-[#169c46] text-white py-3 rounded-full font-semibold transition shadow disabled:opacity-60"
        >
          {loading ? "Optimizando..." : "Optimizar canción"}
        </button>
      </div>

      {/* RESULTADO */}
      <div className="w-1/3 bg-white p-8 rounded-2xl shadow-lg flex flex-col items-center justify-center">

        <h2 className="text-lg font-bold text-gray-800 mb-4">
          Popularidad estimada
        </h2>

        <div className="text-6xl font-extrabold text-[#1DB954] mb-3">
          {predictionResult !== null ? predictionResult.toFixed(0) : "--"}
        </div>

        <p className="text-sm text-gray-500 mb-6">
          escala 0 – 100
        </p>

        <details className="w-full">
          <summary className="cursor-pointer text-[#1DB954] font-medium">
            Ver parámetros optimizados
          </summary>
          <pre className="text-xs mt-3 overflow-auto max-h-64 bg-gray-50 p-3 rounded border">
            {JSON.stringify(features, null, 2)}
          </pre>
        </details>
      </div>

    </div>
  </div>
);

}
