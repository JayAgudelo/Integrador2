import React, { useRef, useState } from "react";
import confetti from "canvas-confetti";

interface PredictionResult {
  prediction?: number | string;
  features?: Record<string, any>;
  error?: string;
}

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
    <div className="min-h-screen flex items-center justify-center bg-gray-100 p-6">
      <div className="w-full max-w-2xl">
        {/* Upload area */}
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`cursor-pointer p-8 rounded-2xl border-4 border-dashed transition-all duration-200
            ${dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-white"}`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <div className="text-center">
            <p className="mt-4 font-semibold text-gray-700">
              Arrastra tu audio aqui o haz clic para seleccionar
            </p>
            <p className="text-sm text-gray-500 mt-2">Formatos: mp3, wav, ogg</p>
          </div>
        </div>

        {/* Formulario */}
        <div className="mt-6 bg-white p-4 rounded-xl shadow">
          <label className="block text-sm font-medium text-gray-700">Genero</label>
          <input
            type="text"
            value={genre}
            onChange={(e) => setGenre(e.target.value)}
            placeholder="rock, pop, jazz..."
            className="mt-2 mb-3 w-full rounded border px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-300"
          />

          {file && (
            <div className="mb-4">
              <p className="font-medium text-gray-800 truncate">{file.name}</p>
              <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>

              {/* Player de audio */}
              <audio ref={audioRef} controls className="mt-2 w-full">
                <source src={URL.createObjectURL(file)} type={file.type} />
                Tu navegador no soporta audio
              </audio>
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={handleUpload}
              disabled={loading}
              className="flex-1 bg-blue-600 text-white py-2 rounded-lg font-semibold disabled:opacity-60"
            >
              {loading ? "Procesando..." : "Subir y analizar"}
            </button>

            <button onClick={handleClear} className="px-4 py-2 border rounded-lg">
              Limpiar
            </button>
          </div>

          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

          {result && (
            <div className="mt-4 p-3 bg-gray-50 rounded">
              <p className="font-semibold">Prediccion: {String(result.prediction)}</p>
              <details className="mt-2">
                <summary className="cursor-pointer text-sm text-blue-600">Ver features</summary>
                <pre className="text-xs mt-2 overflow-auto max-h-48">
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
