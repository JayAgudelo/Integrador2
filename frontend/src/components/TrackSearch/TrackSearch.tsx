import confetti from "canvas-confetti";
import React, { useState } from "react";


interface PredictionResult {
  prediction?: number | string;
  features?: Record<string, any>;
  error?: string;
}
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
  <div className="flex justify-center p-6">
    <div className="w-full max-w-xl">

      <div className="bg-white p-8 rounded-2xl shadow-lg">

        <h2 className="text-xl font-bold text-gray-900 mb-6">
          Buscar canción por Spotify ID
        </h2>

        {/* Spotify ID */}
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Spotify Track ID
        </label>
        <input
          type="text"
          value={spotifyId}
          onChange={(e) => setSpotifyId(e.target.value)}
          placeholder="Ej: 2EKxmYmUdAVXlaHCnnW13o"
          className="w-full rounded-lg border border-gray-300 px-4 py-2 mb-5 focus:outline-none focus:ring-2 focus:ring-[#1DB954]"
        />

        {/* Género */}
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Género musical
        </label>
        <select
          value={genre}
          onChange={(e) => setGenre(e.target.value)}
          className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#1DB954] mb-6"
        >
          <option value="">Seleccionar género</option>
          {genreOptions.map((g) => (
            <option key={g} value={g}>
              {g.replace("-", " ")}
            </option>
          ))}
        </select>

        {/* Botones */}
        <div className="flex gap-4">
          <button
            onClick={handleSearch}
            disabled={loading}
            className="flex-1 bg-[#1DB954] hover:bg-[#169c46] text-white py-2.5 rounded-full font-semibold transition disabled:opacity-60 shadow"
          >
            {loading ? "Analizando..." : "Buscar y predecir"}
          </button>

          <button
            onClick={handleClear}
            className="px-6 py-2.5 rounded-full border border-gray-300 font-semibold text-gray-700 hover:bg-gray-100 transition"
          >
            Limpiar
          </button>
        </div>

        {error && (
          <p className="mt-4 text-sm text-red-600 font-medium">
            {error}
          </p>
        )}

        {/* Resultado */}
        {result && (
  <div className="mt-8 p-5 rounded-2xl bg-[#121212] text-white shadow-lg">
    <p className="text-lg font-semibold">
      Predicción de popularidad
    </p>

    <p className="text-3xl font-bold text-[#1DB954] mt-1">
      {Number.isFinite(Number(result.prediction)) 
        ? Math.round(Number(result.prediction)) 
        : String(result.prediction)}
    </p>

    <details className="mt-4">
      <summary className="cursor-pointer text-sm text-gray-300">
        Ver features técnicos
      </summary>
      <pre className="text-xs mt-3 overflow-auto max-h-48 bg-black/40 p-3 rounded">
        {JSON.stringify(result.features ?? {}, null, 2)}
      </pre>
    </details>
  </div>
)}


      </div>
    </div>
  </div>
);


}
