import React, { useEffect, useMemo, useState } from "react";
import confetti from "canvas-confetti";
import { useI18n } from "../../i18n";
import { AnalysisSession, FeatureSet, OptimizerSession } from "../../types/analysis";
import { saveOptimizerSession } from "../../utils/storage";
import { fetchWithTimeout } from "../../utils/fetchWithTimeout";
import { BACKEND_URL } from "../../utils/backend";

interface SongOptimizerWizardProps {
  analysisSession: AnalysisSession | null;
  previousOptimization: OptimizerSession | null;
  onGoToUpload: () => void;
  onGoToSearch: () => void;
  onOptimizationComplete: (session: OptimizerSession) => void;
}

interface OptimizationErrorState {
  title: string;
  message: string;
}

const featureSliders: {
  key: keyof FeatureSet;
  label: string;
  min: number;
  max: number;
  step?: number;
  percentage?: boolean;
}[] = [
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

const formatValue = (value: number, percentage?: boolean): string => {
  if (percentage) {
    return `${Math.round(value * 100)}%`;
  }

  if (Number.isInteger(value)) {
    return `${value}`;
  }

  return value.toFixed(2);
};

export default function SongOptimizerWizard({
  analysisSession,
  previousOptimization,
  onGoToUpload,
  onGoToSearch,
  onOptimizationComplete,
}: SongOptimizerWizardProps): JSX.Element {
  const { t } = useI18n();
  const baseFeatures = analysisSession?.features ?? null;
  const [features, setFeatures] = useState<FeatureSet | null>(baseFeatures);
  const [locks, setLocks] = useState<Record<string, boolean>>(
    Object.fromEntries(Object.keys(baseFeatures ?? {}).map((key) => [key, key === "genre"])) as Record<string, boolean>
  );
  const [result, setResult] = useState<OptimizerSession | null>(previousOptimization);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<OptimizationErrorState | null>(null);
  const [generatedBeatUrl, setGeneratedBeatUrl] = useState<string | null>(null);
  const [beatLoading, setBeatLoading] = useState(false);
  const [beatError, setBeatError] = useState<string | null>(null);

  useEffect(() => {
    console.debug("SongOptimizerWizard state", {
      analysisSession,
      previousOptimization,
      result,
    });
  }, [analysisSession, previousOptimization, result]);

  const changedParameters = useMemo(() => {
    if (!analysisSession || !result) {
      return [];
    }

    return Object.entries(result.optimizedFeatures)
      .filter(([key, value]) => analysisSession.features[key] !== value)
      .slice(0, 6);
  }, [analysisSession, result]);

  if (!analysisSession || !features) {
    return (
      <section className="space-y-8">
        <div className="section-heading">
          <div>
            <h1 className="font-display text-4xl font-bold text-app-text sm:text-5xl">{t("optimize.emptyTitle")}</h1>
          </div>
          <p className="max-w-2xl text-base leading-7 text-app-muted">{t("optimize.emptyBody")}</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <button type="button" onClick={onGoToUpload} className="route-card">
            <h2 className="route-title">{t("optimize.fromUpload")}</h2>
            <p className="route-copy">{t("upload.body")}</p>
          </button>
          <button type="button" onClick={onGoToSearch} className="route-card route-card-compact">
            <h2 className="route-title">{t("optimize.fromSearch")}</h2>
            <p className="route-copy">{t("search.body")}</p>
          </button>
        </div>
      </section>
    );
  }

  const handleSliderChange = (key: keyof FeatureSet, value: number) => {
    setFeatures((current) => (current ? { ...current, [key]: value } : current));
  };

  const toggleLock = (key: string) => {
    setLocks((current) => ({ ...current, [key]: !current[key] }));
  };

  const buildBeatPayload = (): FeatureSet => {
    return (result?.optimizedFeatures ?? features) as FeatureSet;
  };

  const handleGenerateBeat = async () => {
    setBeatLoading(true);
    setBeatError(null);
    setGeneratedBeatUrl(null);

    try {
      const backendUrl = BACKEND_URL;
      const payload = {
        features: buildBeatPayload(),
        duration_seconds: 15,
      };

      const response = await fetchWithTimeout(`${backendUrl}/generate-beat-base`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok || !data.track_url) {
        throw new Error(data.error || t("errors.beatGenerationBody"));
      }

      setGeneratedBeatUrl(data.track_url);
      console.info("Generated AI beat prompt:", data.prompt ?? "n/a");
    } catch (generationError) {
      setBeatError(
        generationError instanceof Error ? generationError.message : t("errors.beatGenerationBody")
      );
    } finally {
      setBeatLoading(false);
    }
  };

  const handleOptimize = async () => {
    setLoading(true);
    setError(null);

    try {
      const backendUrl = BACKEND_URL;
      const payload = {
        features,
        locked_features: Object.entries(locks)
          .filter(([, locked]) => locked)
          .map(([key]) => key),
      };

      const response = await fetchWithTimeout(`${backendUrl}/wizard-optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok || data.error || data.predicted_popularity === undefined || !data.optimized_features) {
        throw new Error(data.error || t("errors.optimizeBody"));
      }

      const nextResult: OptimizerSession = {
        originalPrediction: analysisSession.prediction,
        optimizedPrediction: Number(data.predicted_popularity),
        optimizedFeatures: {
          ...data.optimized_features,
          duration_ms: Math.round(data.optimized_features.duration_ms),
        },
        updatedAt: new Date().toISOString(),
      };

      setFeatures(nextResult.optimizedFeatures);
      setResult(nextResult);
      saveOptimizerSession(nextResult);
      onOptimizationComplete(nextResult);

      if (nextResult.optimizedPrediction > 40) {
        confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 } });
      }
    } catch (optimizationError) {
      setError({
        title: t("errors.optimizeTitle"),
        message: optimizationError instanceof Error ? optimizationError.message : t("errors.optimizeBody"),
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[0.7fr_1.3fr]">
      <aside className="space-y-6">
        <div className="glass-panel p-6">
          <h2 className="mt-4 font-display text-3xl font-bold text-app-text">{t("optimize.wizardSetup")}</h2>
          <p className="mt-4 text-sm leading-7 text-app-muted">{t("optimize.wizardBody")}</p>
        </div>

        <div className="result-panel">
          <p className="field-label">{t("optimize.estimatedScore")}</p>
          <p className="mt-4 font-display text-6xl font-bold text-app-primary">
            {result ? Math.round(result.optimizedPrediction) : "--"}
          </p>
          <p className="mt-2 text-sm leading-6 text-app-muted">
            {result
              ? `${Math.round(result.originalPrediction)} -> ${Math.round(result.optimizedPrediction)}`
              : analysisSession.sourceLabel}
          </p>
        </div>

        {result && (
          <div className="soft-panel">
            <h3 className="mt-4 font-display text-2xl font-bold text-app-text">{t("common.completed")}</h3>
            <ul className="mt-5 space-y-3 text-sm leading-7 text-app-muted">
              {changedParameters.length > 0 ? (
                changedParameters.map(([key, value]) => (
                  <li key={key}>
                    {key}: {String(analysisSession.features[key])} {"->"} {String(value)}
                  </li>
                ))
              ) : (
                <li>{t("optimize.body")}</li>
              )}
            </ul>
          </div>
        )}

        {error && (
          <div className="error-panel">
            <p className="error-title">{error.title}</p>
            <p className="error-copy">{error.message}</p>
          </div>
        )}
      </aside>

      <section className="app-shell-card relative">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center rounded-[32px] bg-app/70 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-4">
              <div className="h-14 w-14 animate-spin rounded-full border-4 border-app-primary/30 border-t-app-primary" />
              <p className="text-sm uppercase tracking-[0.2em] text-app-muted">{t("optimize.finding")}</p>
            </div>
          </div>
        )}

        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="mt-3 font-display text-3xl font-bold text-app-text">{t("optimize.controls")}</h2>
          </div>
          <div className="rounded-full bg-white/[0.05] px-4 py-2 text-xs uppercase tracking-[0.22em] text-app-muted">
            {t("optimize.lockPrompt")}
          </div>
        </div>

        <div className="mt-8 rounded-[24px] bg-white/[0.04] p-5">
          <label className="field-label" htmlFor="optimizer-genre">
            {t("upload.genre")}
          </label>
          <div className="mt-3 flex flex-col gap-3 sm:flex-row">
            <input id="optimizer-genre" value={String(features.genre)} readOnly className="field-input sm:flex-1" />
            <button
              type="button"
              onClick={() => toggleLock("genre")}
              className={`rounded-full px-4 py-3 text-sm font-semibold transition ${
                locks.genre ? "bg-app-primary text-app-ink" : "bg-white/5 text-app-text"
              }`}
            >
              {locks.genre ? t("common.locked") : t("common.free")}
            </button>
          </div>
        </div>

        <div className="mt-6 grid gap-5 md:grid-cols-2">
          {featureSliders.map((feature) => (
            <div key={feature.key} className="rounded-[24px] bg-black/25 p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="field-label">{feature.label}</p>
                  <p className="mt-2 text-lg font-semibold text-app-text">
                    {formatValue(Number(features[feature.key]), feature.percentage)}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => toggleLock(String(feature.key))}
                  className={`rounded-full px-3 py-2 text-xs font-semibold uppercase tracking-[0.18em] transition ${
                    locks[String(feature.key)] ? "bg-white/10 text-app-muted" : "bg-app-primary text-app-ink"
                  }`}
                >
                  {locks[String(feature.key)] ? t("common.locked") : t("common.free")}
                </button>
              </div>

              <input
                type="range"
                min={feature.min}
                max={feature.max}
                step={feature.step ?? 0.01}
                value={Number(features[feature.key])}
                onChange={(event) => handleSliderChange(feature.key, Number(event.target.value))}
                className="mt-5 h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-app-primary"
              />

              <div className="mt-3 flex items-center justify-between text-xs uppercase tracking-[0.18em] text-app-muted">
                <span>{feature.min}</span>
                <span>{feature.max}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-8 grid gap-4 sm:grid-cols-[1fr_auto]">
          <button type="button" onClick={handleOptimize} disabled={loading || beatLoading} className="primary-button disabled:opacity-60">
            {loading ? t("optimize.running") : t("optimize.run")}
          </button>
          <button
            type="button"
            onClick={handleGenerateBeat}
            disabled={loading || beatLoading}
            className="secondary-button disabled:opacity-60"
          >
            {beatLoading ? t("optimize.beatGenerating") : t("optimize.generateBeat")}
          </button>
        </div>

        {beatError && (
          <div className="error-panel mt-6">
            <p className="error-title">{t("errors.beatGenerationTitle")}</p>
            <p className="error-copy">{beatError}</p>
          </div>
        )}

        {generatedBeatUrl && (
          <div className="mt-6 rounded-[24px] bg-black/20 p-5">
            <p className="field-label">{t("optimize.generatedBeatTitle")}</p>
            <audio controls src={generatedBeatUrl} className="mt-4 w-full" />
            <p className="mt-3 text-sm leading-6 text-app-muted">{t("optimize.generatedBeatBody")}</p>
          </div>
        )}

        <details className="details-panel mt-8">
          <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-app-text">{t("optimize.viewParams")}</summary>
          <pre className="max-h-96 overflow-auto px-4 pb-4 text-xs leading-6 text-app-muted">
            {JSON.stringify(features, null, 2)}
          </pre>
        </details>
      </section>
    </div>
  );
}
