import React, { useEffect, useRef, useState } from "react";
import confetti from "canvas-confetti";
import useProgressLoop from "../../hooks/useProgressLoop";
import useDelayedMessage from "../../hooks/useDelayedMessage";
import { useI18n } from "../../i18n";
import { AnalysisSession, FeatureSet } from "../../types/analysis";
import { STORAGE_KEYS, loadJson, saveJson } from "../../utils/storage";
import { fetchWithTimeout } from "../../utils/fetchWithTimeout";
import { BACKEND_URL } from "../../utils/backend";

interface AudioUploaderProps {
  onAnalysisComplete: (session: AnalysisSession) => void;
  onTryAlternateRoute: () => void;
}

interface UploadDraft {
  genre: string;
}

interface UploadErrorState {
  title: string;
  message: string;
}

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

const initialDraft = loadJson<UploadDraft>(STORAGE_KEYS.uploadDraft);

function createUploadError(
  t: (key: string) => string,
  stage: "validation" | "extract" | "predict" | "network",
  detail?: string
): UploadErrorState {
  switch (stage) {
    case "validation":
      return {
        title: t("errors.genreRequiredTitle"),
        message: t("errors.genreRequiredBody"),
      };
    case "extract":
      return {
        title: t("errors.extractTitle"),
        message: detail || t("errors.extractBody"),
      };
    case "predict":
      return {
        title: t("errors.predictTitle"),
        message: detail || t("errors.predictBody"),
      };
    default:
      return {
        title: t("errors.networkTitle"),
        message: detail || t("errors.networkBody"),
      };
  }
}

export default function AudioUploader({
  onAnalysisComplete,
  onTryAlternateRoute,
}: AudioUploaderProps): JSX.Element {
  const { t } = useI18n();
  const [file, setFile] = useState<File | null>(null);
  const [genre, setGenre] = useState(initialDraft?.genre ?? "");
  const [dragActive, setDragActive] = useState(false);
  const [progressStatus, setProgressStatus] = useState<ProgressStatus>("idle");
  const [error, setError] = useState<UploadErrorState | null>(null);
  const [audioUrl, setAudioUrl] = useState("");

  const inputRef = useRef<HTMLInputElement | null>(null);
  const progress = useProgressLoop(progressStatus);
  const longWait = useDelayedMessage(progressStatus === "loading");

  useEffect(() => {
    saveJson(STORAGE_KEYS.uploadDraft, { genre });
  }, [genre]);

  useEffect(() => {
    if (!file) {
      setAudioUrl("");
      return;
    }

    const nextUrl = URL.createObjectURL(file);
    setAudioUrl(nextUrl);

    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const selectFile = (selected: File | null) => {
    if (selected && selected.type.startsWith("audio/")) {
      setFile(selected);
      setError(null);
      return;
    }

    setError({
      title: t("errors.invalidFileTitle"),
      message: t("errors.invalidFileBody"),
    });
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    selectFile(event.target.files?.[0] ?? null);
  };

  const handleDrag = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();

    if (event.type === "dragenter" || event.type === "dragover") {
      setDragActive(true);
    }

    if (event.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
    selectFile(event.dataTransfer.files?.[0] ?? null);
  };

  const handleClear = () => {
    setFile(null);
    setError(null);
    setProgressStatus("idle");
  };

  const canAnalyze = Boolean(file && genre && progressStatus !== "loading");

  const handleUpload = async () => {
    if (!genre) {
      setError(createUploadError(t, "validation"));
      return;
    }

    if (!file) {
      setError({
        title: t("errors.uploadFileTitle"),
        message: t("errors.uploadFileBody"),
      });
      return;
    }

    setProgressStatus("loading");
    setError(null);

    try {
      const backendUrl = BACKEND_URL;
      const formData = new FormData();
      formData.append("audio", file);
      formData.append("genre", genre);

      const featuresResponse = await fetchWithTimeout(`${backendUrl}/extract-features`, {
        method: "POST",
        body: formData,
      });

      const featuresJson = await featuresResponse.json();

      if (!featuresResponse.ok || featuresJson.error) {
        throw createUploadError(t, "extract", featuresJson.error);
      }

      const features = featuresJson.features as FeatureSet | undefined;

      if (!features) {
        throw createUploadError(t, "extract", "The backend did not return a valid feature set for the uploaded file.");
      }

      const predictionResponse = await fetchWithTimeout(`${backendUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
      });

      const predictionJson = await predictionResponse.json();

      if (!predictionResponse.ok || predictionJson.error) {
        throw createUploadError(t, "predict", predictionJson.error);
      }

      const prediction = Number(predictionJson.prediction);

      if (!Number.isFinite(prediction)) {
        throw createUploadError(t, "predict", "The backend returned an invalid popularity score.");
      }

      const session: AnalysisSession = {
        route: "upload",
        genre,
        sourceLabel: file.name,
        sourceValue: file.name,
        prediction,
        features: predictionJson.features as FeatureSet,
        createdAt: new Date().toISOString(),
      };

      setProgressStatus("success");

      if (prediction > 40) {
        confetti({ particleCount: 150, spread: 70, origin: { y: 0.6 } });
      }

      window.setTimeout(() => {
        onAnalysisComplete(session);
      }, 220);
    } catch (uploadError) {
      setProgressStatus("error");
      setError(
        typeof uploadError === "object" && uploadError !== null && "title" in uploadError
          ? (uploadError as UploadErrorState)
          : createUploadError(
              t,
              "network",
              uploadError instanceof Error ? uploadError.message : "Unexpected error while processing the uploaded track."
            )
      );
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
      <section className="app-shell-card">
        <div className="space-y-3">
          <h2 className="font-display text-3xl font-bold text-app-text sm:text-4xl">{t("upload.title")}</h2>
          <p className="max-w-2xl text-sm leading-7 text-app-muted">
            {t("upload.body")}
          </p>
        </div>

        <div
          onClick={() => inputRef.current?.click()}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`mt-8 cursor-pointer rounded-[28px] border border-dashed p-8 text-center transition sm:p-12 ${
            dragActive
              ? "border-app-primary bg-app-primary/10 shadow-[0_0_40px_rgba(0,244,254,0.12)]"
              : "border-white/10 bg-black/25 hover:border-app-primary/40"
          }`}
        >
          <input ref={inputRef} type="file" accept="audio/*" className="hidden" onChange={handleFileChange} />
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-white/5">
            <div className="h-6 w-6 rounded-full bg-gradient-to-br from-app-primary to-app-secondary" />
          </div>
          <h3 className="mt-6 font-display text-2xl font-bold text-app-text">{t("upload.dragTitle")}</h3>
          <p className="mt-3 text-sm text-app-muted">{t("upload.dragBody")}</p>
          <p className="mt-2 text-xs uppercase tracking-[0.22em] text-app-muted">{t("upload.supported")}</p>
        </div>

        <div className="mt-8 grid gap-6 md:grid-cols-2">
          <div>
            <label className="field-label" htmlFor="audio-genre">
              {t("upload.genre")}
            </label>
            <select
              id="audio-genre"
              value={genre}
              onChange={(event) => setGenre(event.target.value)}
              className="field-select mt-3"
            >
              <option value="">{t("upload.genrePlaceholder")}</option>
              {genreOptions.map((item) => (
                <option key={item} value={item}>
                  {item.replace("-", " ")}
                </option>
              ))}
            </select>
          </div>

          <div>
            <p className="field-label">{t("upload.progress")}</p>
            <div className="mt-3 rounded-[20px] bg-black/30 p-4">
              <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-app-muted">
                <span>Status</span>
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
                    ? t("upload.progressBodyLongWait")
                    : t("upload.progressBodyLoading")
                  : t("upload.progressBodyIdle")}
              </p>
            </div>
          </div>
        </div>

        {file && (
          <div className="mt-8 rounded-[24px] bg-white/[0.04] p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="field-label">{t("upload.selectedFile")}</p>
                <p className="mt-2 text-lg font-semibold text-app-text">{file.name}</p>
                <p className="text-sm text-app-muted">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
              <div className="min-w-[16rem] flex-1 sm:max-w-sm">
                {audioUrl && (
                  <audio controls className="w-full opacity-80">
                    <source src={audioUrl} type={file.type} />
                  </audio>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="action-row mt-8">
          <button
            type="button"
            onClick={handleUpload}
            disabled={!canAnalyze}
            className="primary-button disabled:cursor-not-allowed disabled:opacity-50"
          >
            {progressStatus === "loading" ? t("upload.analyzing") : t("upload.analyze")}
          </button>
          <button type="button" onClick={handleClear} className="secondary-button">
            {t("upload.clear")}
          </button>
        </div>

        {!genre && <p className="mt-4 text-sm text-app-muted">{t("upload.genreRequiredHint")}</p>}

        {error && (
          <div className="error-panel mt-5">
            <p className="error-title">{error.title}</p>
            <p className="error-copy">{error.message}</p>
            <button type="button" onClick={onTryAlternateRoute} className="error-action">
              {t("upload.tryAlternate")}
            </button>
          </div>
        )}
      </section>

      <aside className="space-y-6">
        <div className="glass-panel p-6">
          <span className="eyebrow">{t("upload.journeyFit")}</span>
          <div className="mt-5 space-y-4 text-sm leading-7 text-app-muted">
            <p>{t("upload.journeyBody1")}</p>
            <p>{t("upload.journeyBody2")}</p>
          </div>
        </div>

        <div className="result-panel">
          <p className="field-label">{t("upload.flowChecklist")}</p>
          <ul className="mt-4 space-y-3 text-sm leading-7 text-app-muted">
            <li>{t("upload.step1")}</li>
            <li>{t("upload.step2")}</li>
            <li>{t("upload.step3")}</li>
            <li>{t("upload.step4")}</li>
          </ul>
        </div>
      </aside>
    </div>
  );
}
