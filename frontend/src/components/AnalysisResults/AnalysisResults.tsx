import React from "react";
import { AnalysisSession } from "../../types/analysis";
import { useI18n } from "../../i18n";
import { buildFeatureInsights } from "../../utils/analysisInsights";

interface AnalysisResultsProps {
  session: AnalysisSession;
  onOptimize: () => void;
  onAnalyzeAnother: () => void;
  onOpenHelp: () => void;
}

const toneClasses = {
  positive: "text-app-success bg-app-success/10",
  watch: "text-[#ffd27a] bg-[#ffd27a]/10",
  neutral: "text-app-primary bg-app-primary/10",
};

export default function AnalysisResults({
  session,
  onOptimize,
  onAnalyzeAnother,
  onOpenHelp,
}: AnalysisResultsProps): JSX.Element {
  const { t } = useI18n();
  const insights = buildFeatureInsights(session.features);

  return (
    <section className="space-y-8">
      <div className="section-heading">
        <div>
          <h1 className="font-display text-4xl font-bold text-app-text sm:text-5xl">{t("results.title")}</h1>
        </div>
        <p className="max-w-2xl text-base leading-7 text-app-muted">
          {t("results.body")}
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <div className="app-shell-card">
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div>
              <span className="eyebrow">{session.route === "upload" ? t("results.uploadRoute") : t("results.searchRoute")}</span>
              <h2 className="mt-4 font-display text-3xl font-bold text-app-text sm:text-4xl">{session.sourceLabel}</h2>
              <p className="mt-3 text-sm leading-7 text-app-muted">
                {t("results.genre")}: <span className="text-app-text">{session.genre}</span>
              </p>
            </div>
            <div className="score-orbit">
              <div className="score-orbit-ring" />
              <div className="score-orbit-core">
                <span className="font-display text-6xl font-bold text-app-primary">{Math.round(session.prediction)}</span>
                <span className="text-xs uppercase tracking-[0.22em] text-app-muted">{t("results.score")}</span>
              </div>
            </div>
          </div>

          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            <div className="soft-card">
              <p className="soft-label">{t("results.tempo")}</p>
              <p className="soft-value">{session.features.tempo} BPM</p>
            </div>
            <div className="soft-card">
              <p className="soft-label">{t("results.energy")}</p>
              <p className="soft-value">{Math.round(Number(session.features.energy) * 100)}%</p>
            </div>
            <div className="soft-card">
              <p className="soft-label">{t("results.duration")}</p>
              <p className="soft-value">{(Number(session.features.duration_ms) / 60000).toFixed(2)} min</p>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {insights.map((insight) => (
            <article key={insight.title} className="glass-panel p-6">
              <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] ${toneClasses[insight.tone]}`}>
                {insight.tone === "positive" ? t("results.supportive") : insight.tone === "watch" ? t("results.watch") : t("results.context")}
              </span>
              <h3 className="mt-4 font-display text-2xl font-bold text-app-text">{insight.title}</h3>
              <p className="mt-4 text-sm leading-7 text-app-muted">{insight.body}</p>
            </article>
          ))}
        </div>
      </div>

      <div className="result-panel">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-3xl">
            <p className="field-label">{t("results.limitTitle")}</p>
            <p className="mt-3 text-sm leading-7 text-app-muted">
              {t("results.limitBody")}
            </p>
          </div>
          <div className="action-row">
            <button type="button" onClick={onAnalyzeAnother} className="secondary-button">
              {t("results.analyzeAnother")}
            </button>
            <button type="button" onClick={onOpenHelp} className="ghost-button">
              {t("results.whatDoesMean")}
            </button>
            <button type="button" onClick={onOptimize} className="primary-button">
              {t("results.optimize")}
            </button>
          </div>
        </div>
      </div>

      <details className="details-panel">
        <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-app-text">{t("results.viewFeatures")}</summary>
        <pre className="max-h-96 overflow-auto px-4 pb-4 text-xs leading-6 text-app-muted">
          {JSON.stringify(session.features, null, 2)}
        </pre>
      </details>
    </section>
  );
}
