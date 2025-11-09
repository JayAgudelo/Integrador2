import React, { useState } from "react";
import AudioUploader from "./components/AudioUploader";
import TrackSearch from "./components/TrackSearch";

export default function App(): JSX.Element {
  const [activeTab, setActiveTab] = useState<"upload" | "search">("upload");

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-blue-600 text-white p-4">
        <h1 className="text-xl font-bold text-center">Music Popularity Predictor</h1>
      </header>

      {/* Tabs */}
      <div className="flex justify-center mt-4 space-x-4">
        <button
          onClick={() => setActiveTab("upload")}
          className={`px-4 py-2 rounded-lg font-semibold ${
            activeTab === "upload" ? "bg-blue-600 text-white" : "bg-white text-blue-600 border border-blue-600"
          }`}
        >
          Subir Audio
        </button>
        <button
          onClick={() => setActiveTab("search")}
          className={`px-4 py-2 rounded-lg font-semibold ${
            activeTab === "search" ? "bg-blue-600 text-white" : "bg-white text-blue-600 border border-blue-600"
          }`}
        >
          Buscar Track
        </button>
      </div>

      {/* Contenido */}
      <main className="p-6">
        {activeTab === "upload" && <AudioUploader />}
        {activeTab === "search" && <TrackSearch />}
      </main>
    </div>
  );
}
