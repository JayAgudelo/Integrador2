import confetti from "canvas-confetti";
import React, { useState } from "react";


interface PredictionResult {
  prediction?: number | string;
  features?: Record<string, any>;
  error?: string;
}

export default function TrackSearch(): JSX.Element {
  const [spotifyId, setSpotifyId] = useState<string>("");
  const [genre, setGenre] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!spotifyId.trim()) {
      setError("Ingresa un Spotify ID");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      console.log("Iniciando búsqueda del track en ReccoBeats...");

      // 1️⃣ Llamar a /search-track para obtener trackId de Recco
      const searchResp = await fetch(
        `http://localhost:8000/search-track?ids=${spotifyId.trim()}`
      );
      const searchData = await searchResp.json();
      console.log("Respuesta /search-track:", searchData);

      if (!searchResp.ok || searchData.error) {
        throw new Error(
          searchData.error || "No se pudo obtener el track de ReccoBeats"
        );
      }

      const trackId = searchData;
      console.log("TrackId obtenido:", trackId);

      if (!trackId) {
        throw new Error(
          "No se ha podido obtener las caracteristicas de esta cancion, intenta con el mp3"
        );
      }

      // 2️⃣ Llamar a /track-features con el trackId y género
      console.log("Obteniendo features del track...");
      const featuresResp = await fetch(
        `http://localhost:8000/track-features?track_id=${trackId}&genre=${encodeURIComponent(
          genre
        )}`
      );
      const featuresData = await featuresResp.json();
      console.log("Respuesta /track-features:", featuresData);

      if (!featuresResp.ok || featuresData.error) {
        throw new Error(
          featuresData.error || "No se pudo obtener las features del track"
        );
      }

      // 3️⃣ Llamar a /predict con las features obtenidas
      console.log("Enviando features a /predict...");
      const predictResp = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...featuresData }),
      });
      const predictData = await predictResp.json();
      console.log("Respuesta /predict:", predictData);

      if (!predictResp.ok || predictData.error) {
        throw new Error(predictData.error || "Error al predecir popularidad");
      }

      setResult(predictData);

      // Confeti si predicción > 30
      if (predictData.prediction && Number(predictData.prediction) > 40) {
        console.log("Prediccion alta, lanzando confeti!");
        confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 } });
      }
    } catch (err: any) {
      console.error("Error en handleSearch:", err);
      setError(err.message || "Error desconocido");
    } finally {
      setLoading(false);
      console.log("Búsqueda finalizada");
    }
  };

  const handleClear = () => {
    setSpotifyId("");
    setGenre("");
    setResult(null);
    setError(null);
    console.log("Formulario limpiado");
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-gray-100">
      <div className="w-full max-w-lg bg-white p-6 rounded-xl shadow">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Spotify Track ID
        </label>
        <input
          type="text"
          value={spotifyId}
          onChange={(e) => setSpotifyId(e.target.value)}
          placeholder="Ej: 2EKxmYmUdAVXlaHCnnW13o"
          className="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-300 mb-4"
        />

        <label className="block text-sm font-medium text-gray-700 mb-2">
          Género (opcional)
        </label>
        <input
          type="text"
          value={genre}
          onChange={(e) => setGenre(e.target.value)}
          placeholder="rock, pop, jazz..."
          className="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-300 mb-4"
        />

        <div className="flex gap-3 mb-4">
          <button
            onClick={handleSearch}
            disabled={loading}
            className="flex-1 bg-blue-600 text-white py-2 rounded font-semibold disabled:opacity-60"
          >
            {loading ? "Cargando..." : "Buscar y predecir"}
          </button>
          <button
            onClick={handleClear}
            className="px-4 py-2 border rounded font-semibold"
          >
            Limpiar
          </button>
        </div>

        {error && <p className="text-red-600 text-sm mb-3">{error}</p>}

        {result && (
          <div className="mt-4 p-3 bg-gray-50 rounded">
            <p className="font-semibold">
              Prediccion: {String(result.prediction)}
            </p>
            <details className="mt-2">
              <summary className="cursor-pointer text-sm text-blue-600">
                Ver features
              </summary>
              <pre className="text-xs mt-2 overflow-auto max-h-48">
                {JSON.stringify(result.features ?? {}, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}
