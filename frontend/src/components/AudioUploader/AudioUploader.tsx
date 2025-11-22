import React, { useRef, useState } from "react";
import confetti from "canvas-confetti";

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
export default function AudioUploader(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [genre, setGenre] = useState<string>("");
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Manejar seleccion por input
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null;
    if (selected && selected.type.startsWith("audio/")) {
      setFile(selected);
      setError(null);
      setResult(null);
    } else {
      setError("Formato no soportado");
    }
  };

  // Manejar drag events
  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    if (e.type === "dragleave") setDragActive(false);
  };

  // Manejar drop
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0] ?? null;
    if (dropped && dropped.type.startsWith("audio/")) {
      setFile(dropped);
      setError(null);
      setResult(null);
    } else {
      setError("Formato no soportado");
    }
  };

  // Limpiar formulario
  const handleClear = () => {
    setFile(null);
    setGenre("");
    setResult(null);
    setError(null);
  };

  // Subir y procesar archivo
  const handleUpload = async () => {
    if (!file) {
      setError("Seleccionar un archivo primero");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("audio", file);
      if (genre.trim()) formData.append("genre", genre.trim());

      // 1️⃣ Llamar a /extract-features
      const featuresResp = await fetch("http://localhost:8000/extract-features", {
        method: "POST",
        body: formData,
      });

      if (!featuresResp.ok) {
        throw new Error(await featuresResp.text() || `HTTP ${featuresResp.status}`);
      }

      const featuresJson = await featuresResp.json();
      const features = featuresJson.features;
      if (!features) throw new Error("No se obtuvieron features");

      // 2️⃣ Llamar a /predict
      const predictResp = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(features),
      });

      if (!predictResp.ok) {
        throw new Error(await predictResp.text() || `HTTP ${predictResp.status}`);
      }

      const predictJson = await predictResp.json();
      setResult(predictJson);

      // 3️⃣ Mostrar confeti si prediction > 60
      if (predictJson.prediction && Number(predictJson.prediction) > 40) {
        confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 } });
      }

    } catch (err: any) {
      setError(err?.message ?? String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
  <div className="flex justify-center p-6">
    <div className="w-full max-w-3xl">

      {/* Zona de subida */}
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer p-10 rounded-2xl border-2 border-dashed transition-all duration-200 text-center
          ${dragActive 
            ? "border-[#1DB954] bg-green-50" 
            : "border-gray-300 bg-white hover:border-[#1DB954]"}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={handleFileChange}
        />

        <p className="text-lg font-semibold text-gray-800">
          Arrastra tu audio aquí
        </p>
        <p className="text-sm text-gray-500 mt-2">
          o haz clic para seleccionar archivo
        </p>
        <p className="text-xs text-gray-400 mt-1">
          mp3, wav, ogg
        </p>
      </div>

      {/* Panel de configuración */}
      <div className="mt-8 bg-white p-6 rounded-2xl shadow-lg">

        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Género musical
        </label>

        <select
          value={genre}
          onChange={(e) => setGenre(e.target.value)}
          className="w-full rounded-lg border px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#1DB954]"
        >
          <option value="">Seleccionar género</option>
          {genreOptions.map((g) => (
            <option key={g} value={g}>
              {g.replace("-", " ")}
            </option>
          ))}
        </select>

        {file && (
          <div className="mt-6 bg-gray-50 p-4 rounded-xl">
            <p className="font-medium text-gray-800 truncate">{file.name}</p>
            <p className="text-sm text-gray-500">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>

            <audio ref={audioRef} controls className="mt-3 w-full">
              <source src={URL.createObjectURL(file)} type={file.type} />
            </audio>
          </div>
        )}

        <div className="flex gap-4 mt-6">
          <button
            onClick={handleUpload}
            disabled={loading}
            className="flex-1 bg-[#1DB954] hover:bg-[#169c46] text-white py-3 rounded-full font-semibold transition disabled:opacity-60"
          >
            {loading ? "Analizando..." : "Analizar canción"}
          </button>

          <button
            onClick={handleClear}
            className="px-6 py-3 rounded-full border border-gray-300 hover:bg-gray-100 transition"
          >
            Limpiar
          </button>
        </div>

        {error && (
          <p className="mt-4 text-sm text-red-600 font-medium">
            {error}
          </p>
        )}

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