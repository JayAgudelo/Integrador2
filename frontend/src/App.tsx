import React, { useEffect, useMemo, useState } from "react";
import AnalysisResults from "./components/AnalysisResults";
import AudioUploader from "./components/AudioUploader";
import SongOptimizerWizard from "./components/SongOptimizerWizard";
import TrackSearch from "./components/TrackSearch";
import { Locale, useI18n } from "./i18n";
import { AnalysisSession, OptimizerSession } from "./types/analysis";
import {
  STORAGE_KEYS,
  loadAnalysisSession,
  loadJson,
  loadOptimizerSession,
  removeStoredValue,
  saveAnalysisSession,
  saveJson,
} from "./utils/storage";

type Screen = "home" | "analyze" | "results" | "optimize" | "how" | "help";
type AnalyzeMode = "upload" | "search";

const localeOptions: Locale[] = ["en", "es", "fr"];

export default function App(): JSX.Element {
  const { locale, setLocale, t } = useI18n();
  const [activeScreen, setActiveScreen] = useState<Screen>("home");
  const [analyzeMode, setAnalyzeMode] = useState<AnalyzeMode>(
    loadJson<AnalyzeMode>(STORAGE_KEYS.analyzeMode) ?? "upload"
  );
  const [analysisSession, setAnalysisSession] = useState<AnalysisSession | null>(loadAnalysisSession);
  const [optimizerSession, setOptimizerSession] = useState<OptimizerSession | null>(loadOptimizerSession);

  useEffect(() => {
    saveJson(STORAGE_KEYS.analyzeMode, analyzeMode);
  }, [analyzeMode]);

  const navigationItems: { key: Exclude<Screen, "results">; label: string }[] = useMemo(
    () => [
      { key: "home", label: t("nav.home") },
      { key: "analyze", label: t("nav.analyze") },
      { key: "optimize", label: t("nav.optimize") },
      { key: "how", label: t("nav.how") },
      { key: "help", label: t("nav.help") },
    ],
    [t]
  );

  const steps = useMemo(
    () => [
      { eyebrow: "Step 01", title: t("how.step1"), body: t("how.step1Body") },
      { eyebrow: "Step 02", title: t("how.step2"), body: t("how.step2Body") },
      { eyebrow: "Step 03", title: t("how.step3"), body: t("how.step3Body") },
    ],
    [t]
  );

  const limitations = useMemo(
    () => [t("help.limit1"), t("help.limit2"), t("help.limit3"), t("help.limit4")],
    [t]
  );

  const openAnalyze = (mode: AnalyzeMode) => {
    setAnalyzeMode(mode);
    setActiveScreen("analyze");
  };

  const handleAnalysisComplete = (session: AnalysisSession) => {
    setAnalysisSession(session);
    saveAnalysisSession(session);
    setOptimizerSession(null);
    removeStoredValue(STORAGE_KEYS.optimizer);
    setActiveScreen("results");
  };

  const openResults = () => {
    if (analysisSession) {
      setActiveScreen("results");
    }
  };

  const startOver = () => {
    setActiveScreen("analyze");
  };

  return (
    <div className="min-h-screen bg-app text-app-text">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="app-orb app-orb-cyan" />
        <div className="app-orb app-orb-violet" />
        <div className="app-grid" />
      </div>

      <header className="sticky top-0 z-50 border-b border-white/5 bg-app/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-4 sm:px-6 lg:px-8">
          <button type="button" onClick={() => setActiveScreen("home")} className="flex items-center gap-3 text-left">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/5 shadow-[0_0_30px_rgba(0,244,254,0.12)]">
              <div className="h-5 w-5 rounded-full bg-gradient-to-br from-app-primary to-app-secondary" />
            </div>
            <div>
              <p className="font-display text-lg font-bold uppercase tracking-[0.22em] text-app-primary">Sonic Architect</p>
              <p className="text-xs uppercase tracking-[0.24em] text-app-muted">Music Popularity Predictor</p>
            </div>
          </button>

          <nav className="hidden items-center gap-2 rounded-full bg-white/5 px-2 py-2 md:flex">
            {navigationItems.map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setActiveScreen(item.key)}
                className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                  activeScreen === item.key
                    ? "bg-gradient-to-r from-app-primary to-app-primary-strong text-app-ink shadow-[0_0_30px_rgba(0,244,254,0.18)]"
                    : "text-app-muted hover:text-app-text"
                }`}
              >
                {item.label}
              </button>
            ))}
            {analysisSession && (
              <button
                type="button"
                onClick={openResults}
                className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                  activeScreen === "results"
                    ? "bg-gradient-to-r from-app-primary to-app-primary-strong text-app-ink shadow-[0_0_30px_rgba(0,244,254,0.18)]"
                    : "text-app-muted hover:text-app-text"
                }`}
              >
                {t("nav.results")}
              </button>
            )}
          </nav>

          <div className="flex items-center gap-3">
            <label className="sr-only" htmlFor="locale-selector">
              {t("app.locale")}
            </label>
            <select
              id="locale-selector"
              value={locale}
              onChange={(event) => setLocale(event.target.value as Locale)}
              className="rounded-full bg-white/5 px-4 py-2 text-sm text-app-text outline-none"
            >
              {localeOptions.map((option) => (
                <option key={option} value={option}>
                  {option.toUpperCase()}
                </option>
              ))}
            </select>
            {analysisSession && (
              <button type="button" onClick={openResults} className="secondary-button hidden sm:inline-flex">
                {t("app.resumeResult")}
              </button>
            )}
            <button type="button" onClick={() => openAnalyze("upload")} className="rounded-full bg-white/5 px-4 py-2 text-sm font-semibold text-app-text transition hover:bg-white/10">
              {t("app.startAnalyzing")}
            </button>
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-7xl px-5 pb-16 pt-8 sm:px-6 lg:px-8">
        {activeScreen === "home" && (
          <div className="space-y-12">
            <section className="grid gap-8 lg:grid-cols-[1.25fr_0.75fr] lg:items-end">
              <div className="space-y-8">
                <div className="space-y-4">
                  <span className="eyebrow">{t("home.heroEyebrow")}</span>
                  <h1 className="max-w-4xl font-display text-5xl font-bold leading-[0.92] text-app-text sm:text-6xl lg:text-7xl">
                    {t("home.heroTitle")}
                  </h1>
                  <p className="max-w-2xl text-lg leading-8 text-app-muted">{t("home.heroBody")}</p>
                </div>

                <div className="flex flex-wrap gap-4">
                  <button type="button" onClick={() => openAnalyze("upload")} className="primary-button">
                    {t("home.ctaUpload")}
                  </button>
                  <button type="button" onClick={() => openAnalyze("search")} className="secondary-button">
                    {t("home.ctaSearch")}
                  </button>
                  <button type="button" onClick={() => setActiveScreen("optimize")} className="ghost-button">
                    {t("home.ctaOptimize")}
                  </button>
                </div>

                <div className="grid gap-4 sm:grid-cols-3">
                  <div className="metric-card">
                    <span className="metric-label">Routes</span>
                    <strong className="metric-value">{t("home.metricRoutes")}</strong>
                    <p className="metric-copy">{t("home.metricRoutesBody")}</p>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Prediction</span>
                    <strong className="metric-value">{t("home.metricPrediction")}</strong>
                    <p className="metric-copy">{t("home.metricPredictionBody")}</p>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Trust</span>
                    <strong className="metric-value">{t("home.metricTrust")}</strong>
                    <p className="metric-copy">{t("home.metricTrustBody")}</p>
                  </div>
                </div>
              </div>

              <div className="glass-panel p-6 sm:p-8">
                <div className="flex items-center justify-between">
                  <span className="eyebrow">{t("home.preview")}</span>
                  <span className="rounded-full bg-app-success/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-app-success">
                    Live Flow
                  </span>
                </div>
                <div className="mt-8 flex items-end gap-2">
                  {[32, 54, 43, 68, 81, 58, 74, 96, 61, 76, 49, 72].map((height, index) => (
                    <div
                      key={index}
                      className="flex-1 rounded-full bg-gradient-to-t from-app-secondary/60 to-app-primary"
                      style={{ height: `${height * 1.5}px` }}
                    />
                  ))}
                </div>
                <div className="mt-8 space-y-4">
                  <div className="rounded-3xl bg-black/30 p-5">
                    <p className="text-xs uppercase tracking-[0.2em] text-app-muted">{t("home.currentPosture")}</p>
                    <div className="mt-3 flex items-end justify-between gap-4">
                      <div>
                        <p className="font-display text-5xl font-bold text-app-primary">78</p>
                        <p className="text-sm text-app-muted">{t("home.estimatedScore")}</p>
                      </div>
                      <div className="max-w-[12rem] text-right text-sm leading-6 text-app-muted">
                        Strong rhythmic alignment with room to improve track length and brightness.
                      </div>
                    </div>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="soft-card">
                      <p className="soft-label">{t("home.routeUnreleased")}</p>
                      <p className="soft-value">{t("analyze.uploadTab")}</p>
                    </div>
                    <div className="soft-card">
                      <p className="soft-label">{t("home.routeReferences")}</p>
                      <p className="soft-value">{t("analyze.searchTab")}</p>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section className="grid gap-6 lg:grid-cols-12">
              <button type="button" onClick={() => openAnalyze("upload")} className="route-card lg:col-span-7">
                <h2 className="route-title">{t("home.uploadRouteTitle")}</h2>
                <p className="route-copy">{t("home.uploadRouteBody")}</p>
                <span className="route-action">{t("home.openAudio")}</span>
              </button>

              <button type="button" onClick={() => openAnalyze("search")} className="route-card route-card-compact lg:col-span-5">
                <h2 className="route-title">{t("home.searchRouteTitle")}</h2>
                <p className="route-copy">{t("home.searchRouteBody")}</p>
                <span className="route-action">{t("home.openLookup")}</span>
              </button>

              <button type="button" onClick={() => setActiveScreen("optimize")} className="feature-stage lg:col-span-12">
                <div className="space-y-4">
                  <h2 className="route-title">{t("home.optimizeTitle")}</h2>
                  <p className="route-copy max-w-2xl">{t("home.optimizeBody")}</p>
                </div>
                <div className="optimizer-preview">
                  {[
                    ["Danceability", "82%"],
                    ["Energy", "68%"],
                    ["Valence", "56%"],
                  ].map(([label, value]) => (
                    <div key={label}>
                      <p className="soft-label">{label}</p>
                      <div className="mini-track">
                        <div className="mini-fill" style={{ width: value }} />
                      </div>
                    </div>
                  ))}
                </div>
              </button>
            </section>

            <section className="grid gap-6 lg:grid-cols-3">
              {[
                ["UA-01", t("home.ua1"), t("home.ua1Body")],
                ["UA-02", t("home.ua2"), t("home.ua2Body")],
                ["UA-03", t("home.ua3"), t("home.ua3Body")],
              ].map(([tag, title, body]) => (
                <article key={tag} className="soft-panel">
                  <h3 className="mt-4 font-display text-2xl font-bold text-app-text">{title}</h3>
                  <p className="mt-4 text-sm leading-7 text-app-muted">{body}</p>
                </article>
              ))}
            </section>
          </div>
        )}

        {activeScreen === "analyze" && (
          <section className="space-y-8">
            <div className="section-heading">
              <div>
                <h1 className="font-display text-4xl font-bold text-app-text sm:text-5xl">{t("analyze.title")}</h1>
              </div>
              <p className="max-w-2xl text-base leading-7 text-app-muted">{t("analyze.body")}</p>
            </div>

            <div className="inline-flex rounded-full bg-white/5 p-2">
              <button
                type="button"
                onClick={() => openAnalyze("upload")}
                className={`tab-chip ${analyzeMode === "upload" ? "tab-chip-active" : ""}`}
              >
                {t("analyze.uploadTab")}
              </button>
              <button
                type="button"
                onClick={() => openAnalyze("search")}
                className={`tab-chip ${analyzeMode === "search" ? "tab-chip-active" : ""}`}
              >
                {t("analyze.searchTab")}
              </button>
            </div>

            {analyzeMode === "upload" ? (
              <AudioUploader onAnalysisComplete={handleAnalysisComplete} onTryAlternateRoute={() => openAnalyze("search")} />
            ) : (
              <TrackSearch onAnalysisComplete={handleAnalysisComplete} onTryAlternateRoute={() => openAnalyze("upload")} />
            )}
          </section>
        )}

        {activeScreen === "results" && analysisSession && (
          <AnalysisResults
            session={analysisSession}
            onAnalyzeAnother={startOver}
            onOpenHelp={() => setActiveScreen("help")}
            onOptimize={() => setActiveScreen("optimize")}
          />
        )}

        {activeScreen === "optimize" && (
          <section className="space-y-8">
            <div className="section-heading">
              <div>
                <h1 className="font-display text-4xl font-bold text-app-text sm:text-5xl">{t("optimize.title")}</h1>
              </div>
              <p className="max-w-2xl text-base leading-7 text-app-muted">{t("optimize.body")}</p>
            </div>

            <SongOptimizerWizard
              analysisSession={analysisSession}
              previousOptimization={optimizerSession}
              onGoToUpload={() => openAnalyze("upload")}
              onGoToSearch={() => openAnalyze("search")}
              onOptimizationComplete={setOptimizerSession}
            />
          </section>
        )}

        {activeScreen === "how" && (
          <section className="space-y-8">
            <div className="section-heading">
              <div>
                <h1 className="font-display text-4xl font-bold text-app-text sm:text-5xl">{t("how.title")}</h1>
              </div>
              <p className="max-w-2xl text-base leading-7 text-app-muted">{t("how.body")}</p>
            </div>

            <div className="grid gap-6 lg:grid-cols-3">
              {steps.map((step) => (
                <article key={step.eyebrow} className="glass-panel p-8">
                  <span className="eyebrow">{step.eyebrow}</span>
                  <h2 className="mt-5 font-display text-2xl font-bold text-app-text">{step.title}</h2>
                  <p className="mt-4 text-sm leading-7 text-app-muted">{step.body}</p>
                </article>
              ))}
            </div>
          </section>
        )}

        {activeScreen === "help" && (
          <section className="space-y-8">
            <div className="section-heading">
              <div>
                <h1 className="font-display text-4xl font-bold text-app-text sm:text-5xl">{t("help.title")}</h1>
              </div>
              <p className="max-w-2xl text-base leading-7 text-app-muted">{t("help.body")}</p>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1fr_0.85fr]">
              <div className="glass-panel p-8">
                <span className="eyebrow">Guidance</span>
                <h2 className="mt-4 font-display text-3xl font-bold text-app-text">{t("help.guidanceTitle")}</h2>
                <p className="mt-4 max-w-2xl text-sm leading-7 text-app-muted">{t("help.guidanceBody")}</p>
              </div>

              <div className="soft-panel">
                <span className="eyebrow">Limits</span>
                <ul className="mt-5 space-y-4 text-sm leading-7 text-app-muted">
                  {limitations.map((item) => (
                    <li key={item} className="flex gap-3">
                      <span className="mt-2 h-2.5 w-2.5 flex-none rounded-full bg-app-primary" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
