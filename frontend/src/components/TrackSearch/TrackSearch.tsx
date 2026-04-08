import confetti from "canvas-confetti";
import React, { useEffect, useState } from "react";
import useProgressLoop from "../../hooks/useProgressLoop";
import useDelayedMessage from "../../hooks/useDelayedMessage";
import { useI18n } from "../../i18n";
import { AnalysisSession, FeatureSet } from "../../types/analysis";
import { STORAGE_KEYS, loadJson, saveJson } from "../../utils/storage";
import { fetchWithTimeout } from "../../utils/fetchWithTimeout";
import { BACKEND_URL } from "../../utils/backend";

interface TrackSearchProps {
  onAnalysisComplete: (session: AnalysisSession) => void;
  onTryAlternateRoute: () => void;
}

interface SearchDraft {
  spotifyId: string;
  genre: string;
}

interface SearchErrorState {
  title: string;
  message: string;
}

interface SearchError extends Error, SearchErrorState {}

type ProgressStatus = "idle" | "loading" | "success" | "error";

const genreOptions = [
  "alt-rock",
  "blues",
  "chill",
  "classical",
  "country",
  "dance",
  "edm",
  "electro",
  "electronic",
  "emo",
  "folk",
  "french",
  "funk",
  "german",
  "hard-rock",
  "hardcore",
  "hip-hop",
  "house",
  "indie-pop",
  "jazz",
  "k-pop",
  "metal",
  "pop",
  "punk",
  "rock",
  "sad",
  "sertanejo",
  "singer-songwriter",
  "soul",
  "spanish",
];

const initialDraft = loadJson<SearchDraft>(STORAGE_KEYS.searchDraft);

const normalizeSpotifyId = (value: string): string => {
  const trimmed = value.trim();

  if (!trimmed) {
    return "";
  }

  if (trimmed.includes("spotify.com/track/")) {
    const trackSection = trimmed.split("/track/")[1] ?? "";
    return trackSection.split("?")[0] ?? "";
  }

  if (trimmed.startsWith("spotify:track:")) {
    return trimmed.replace("spotify:track:", "");
  }

  return trimmed;
};

export default function TrackSearch({
  onAnalysisComplete,
  onTryAlternateRoute,
}: TrackSearchProps): JSX.Element {
  const { t } = useI18n();
  const [spotifyId, setSpotifyId] = useState(initialDraft?.spotifyId ?? "");
  const [genre, setGenre] = useState(initialDraft?.genre ?? "");
  const [progressStatus, setProgressStatus] = useState<ProgressStatus>("idle");
  const [error, setError] = useState<SearchErrorState | null>(null);
  const progress = useProgressLoop(progressStatus);
  const longWait = useDelayedMessage(progressStatus === "loading");

  const buildError = (title: string, message: string): SearchError =>
    Object.assign(new Error(message), { title, message });

  useEffect(() => {
    saveJson(STORAGE_KEYS.searchDraft, { spotifyId, genre });
  }, [genre, spotifyId]);

  const canAnalyze = Boolean(spotifyId.trim() && genre && progressStatus !== "loading");

  const handleSearch = async () => {
    const normalizedId = normalizeSpotifyId(spotifyId);

    if (!genre) {
      setError({
        title: t("errors.genreRequiredTitle"),
        message: t("errors.genreRequiredBody"),
      });
      return;
    }

    if (!normalizedId) {
      setError({
        title: t("errors.spotifySourceTitle"),
        message: t("errors.spotifySourceBody"),
      });
      return;
    }

    setProgressStatus("loading");
    setError(null);

    try {
      const backendUrl = BACKEND_URL;
      const searchResponse = await fetchWithTimeout(`${backendUrl}/search-track?ids=${normalizedId}`);
      const searchData = await searchResponse.json();

      if (!searchResponse.ok || searchData.error) {
        throw buildError(t("errors.spotifyResolveTitle"), searchData.error || t("errors.spotifyResolveBody"));
      }

      const trackId = searchData;

      if (!trackId) {
        throw buildError(t("errors.spotifyResolveTitle"), t("errors.spotifyResolveBody"));
      }

      const featuresResponse = await fetchWithTimeout(
        `${backendUrl}/track-features?track_id=${trackId}&genre=${encodeURIComponent(genre)}`
      );
      const featuresData = await featuresResponse.json();

      if (!featuresResponse.ok || featuresData.error) {
        throw buildError(t("errors.spotifyFeaturesTitle"), featuresData.error || t("errors.spotifyFeaturesBody"));
      }

      const predictionResponse = await fetchWithTimeout(`${backendUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...featuresData }),
      });
      const predictionData = await predictionResponse.json();

      if (!predictionResponse.ok || predictionData.error) {
        throw buildError(t("errors.predictTitle"), predictionData.error || t("errors.predictBody"));
      }

      const prediction = Number(predictionData.prediction);

      if (!Number.isFinite(prediction)) {
        throw buildError(t("errors.predictTitle"), t("errors.predictBody"));
      }

      const session: AnalysisSession = {
        route: "search",
        genre,
        sourceLabel: normalizedId,
        sourceValue: normalizedId,
        prediction,
        features: predictionData.features as FeatureSet,
        createdAt: new Date().toISOString(),
      };

      setProgressStatus("success");

      if (prediction > 40) {
        confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 } });
      }

      window.setTimeout(() => {
        onAnalysisComplete(session);
      }, 220);
    } catch (searchError) {
      setProgressStatus("error");
      setError(
        typeof searchError === "object" && searchError !== null && "title" in searchError
          ? (searchError as SearchErrorState)
          : {
              title: t("errors.networkTitle"),
              message: searchError instanceof Error ? searchError.message : t("errors.networkBody"),
            }
      );
    }
  };

  const handleClear = () => {
    setSpotifyId("");
    setGenre("");
    setError(null);
    setProgressStatus("idle");
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
      <section className="app-shell-card">
        <div className="space-y-3">
          <h2 className="font-display text-3xl font-bold text-app-text sm:text-4xl">{t("search.title")}</h2>
          <p className="max-w-2xl text-sm leading-7 text-app-muted">{t("search.body")}</p>
        </div>

        <div className="mt-8 space-y-6">
          <div>
            <label className="field-label" htmlFor="spotify-track-id">
              {t("search.source")}
            </label>
            <input
              id="spotify-track-id"
              type="text"
              value={spotifyId}
              onChange={(event) => setSpotifyId(event.target.value)}
              placeholder={t("search.sourcePlaceholder")}
              className="field-input mt-3"
            />
            <p className="mt-3 text-sm leading-6 text-app-muted">{t("search.sourceBody")}</p>
          </div>

          <div>
            <label className="field-label" htmlFor="spotify-genre">
              {t("search.genre")}
            </label>
            <select
              id="spotify-genre"
              value={genre}
              onChange={(event) => setGenre(event.target.value)}
              className="field-select mt-3"
            >
              <option value="">{t("common.genreSelect")}</option>
              {genreOptions.map((item) => (
                <option key={item} value={item}>
                  {item.replace("-", " ")}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-8 rounded-[20px] bg-black/30 p-4">
          <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-app-muted">
            <span>{t("upload.progress")}</span>
            <span>
              {progressStatus === "loading"
                ? t("upload.status.running")
                : progressStatus === "success"
                ? t("upload.status.completed")
                : progressStatus === "error"
                ? t("upload.status.failed")
                : t("upload.status.waiting")}
            </span>
          </div>
          <div className="progress-track mt-4">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
            {progressStatus === "loading" && <div className="progress-flow" />}
          </div>
          <p className="mt-3 text-sm text-app-muted">
            {progressStatus === "loading"
              ? longWait
                ? t("search.progressBodyLongWait")
                : t("upload.progressBodyLoading")
              : t("upload.progressBodyIdle")}
          </p>
        </div>

        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          <div className="soft-card">
            <p className="soft-label">{t("search.resolve")}</p>
            <p className="soft-value">{t("search.resolveBody")}</p>
          </div>
          <div className="soft-card">
            <p className="soft-label">{t("search.retrieve")}</p>
            <p className="soft-value">{t("search.retrieveBody")}</p>
          </div>
          <div className="soft-card">
            <p className="soft-label">{t("search.predict")}</p>
            <p className="soft-value">{t("search.predictBody")}</p>
          </div>
        </div>

        <div className="action-row mt-8">
          <button
            type="button"
            onClick={handleSearch}
            disabled={!canAnalyze}
            className="primary-button disabled:cursor-not-allowed disabled:opacity-50"
          >
            {progressStatus === "loading" ? t("search.analyzing") : t("search.analyze")}
          </button>
          <button type="button" onClick={handleClear} className="secondary-button">
            {t("search.clear")}
          </button>
        </div>

        {!genre && <p className="mt-4 text-sm text-app-muted">{t("upload.genreRequiredHint")}</p>}

        {error && (
          <div className="error-panel mt-5">
            <p className="error-title">{error.title}</p>
            <p className="error-copy">{error.message}</p>
            <button type="button" onClick={onTryAlternateRoute} className="error-action">
              {t("search.tryAlternate")}
            </button>
          </div>
        )}
      </section>

      <aside className="space-y-6">
        <div className="glass-panel p-6">
          <span className="eyebrow">{t("search.tips")}</span>
          <ul className="mt-5 space-y-3 text-sm leading-7 text-app-muted">
            <li>{t("search.tip1")}</li>
            <li>{t("search.tip2")}</li>
            <li>{t("search.tip3")}</li>
          </ul>
        </div>

        <div className="result-panel">
          <p className="field-label">{t("upload.flowChecklist")}</p>
          <ul className="mt-4 space-y-3 text-sm leading-7 text-app-muted">
            <li>1. {t("search.resolveBody")}.</li>
            <li>2. {t("search.retrieveBody")}.</li>
            <li>3. {t("search.predictBody")}.</li>
            <li>4. {t("results.optimize")}.</li>
          </ul>
        </div>
      </aside>
    </div>
  );
}
