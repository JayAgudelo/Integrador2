import React, { useState } from "react"
import AudioUploader from "./components/AudioUploader"
import TrackSearch from "./components/TrackSearch"
import SongOptimizerWizard from "./components/SongOptimizerWizard"
export default function App(): JSX.Element {
  const [activeTab, setActiveTab] = useState<"upload" | "search" | "wizard">("upload")

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">

      <header className="flex items-center justify-between px-6 py-4 bg-white shadow">

        <div className="flex items-center gap-3">
  <div className="w-10 h-10 bg-[#121212] rounded-full flex items-center justify-center shadow-md">
    <img src="/images/Spotify_logo_without_text.png" className="h-8" alt="Spotify" />
  </div>
</div>


        <h1 className="text-xl font-bold tracking-wide text-gray-900">
          Music Popularity Predictor
        </h1>

        <div></div>
      </header>

      {/* Tabs */}
      <div className="flex justify-center mt-6 space-x-4">
        {[
          { key: "upload", label: "Subir Audio" },
          { key: "search", label: "Buscar Track" },
          { key: "wizard", label: "Optimizar Canción" },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`px-5 py-2 rounded-full font-semibold transition-all duration-200 ${
              activeTab === tab.key
                ? "bg-[#1DB954] text-white shadow-md"
                : "border border-[#1DB954] text-[#1DB954] hover:bg-[#1DB954] hover:text-white"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Contenido */}
      <main className="p-6">
        <div className="bg-white rounded-2xl shadow-lg p-6">
          {activeTab === "upload" && <AudioUploader />}
          {activeTab === "search" && <TrackSearch />}
          {activeTab === "wizard" && <SongOptimizerWizard />}
        </div>
      </main>
    </div>
  )
}
