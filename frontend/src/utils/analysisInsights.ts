import { FeatureSet } from "../types/analysis";

interface Insight {
  title: string;
  tone: "positive" | "watch" | "neutral";
  body: string;
}

const toMinutes = (durationMs: number): string => `${(durationMs / 60000).toFixed(2)} min`;

export function buildFeatureInsights(features: FeatureSet): Insight[] {
  const insights: Insight[] = [];

  if (features.danceability >= 0.7) {
    insights.push({
      title: "Danceability is supporting the track",
      tone: "positive",
      body: `The profile shows danceability at ${Math.round(features.danceability * 100)}%, which aligns with the docs' emphasis on rhythmic accessibility.`,
    });
  } else {
    insights.push({
      title: "Danceability may be limiting momentum",
      tone: "watch",
      body: `Danceability is currently ${Math.round(features.danceability * 100)}%, which suggests less immediate mainstream pull than a more rhythm-forward profile.`,
    });
  }

  if (features.duration_ms > 240000) {
    insights.push({
      title: "Track length is a constraint to review",
      tone: "watch",
      body: `At ${toMinutes(features.duration_ms)}, the song sits on the longer side, and duration is called out in the project materials as a meaningful popularity driver.`,
    });
  } else {
    insights.push({
      title: "Track length is commercially efficient",
      tone: "positive",
      body: `The current duration of ${toMinutes(features.duration_ms)} stays in a concise range that tends to support repeat listening.`,
    });
  }

  if (features.energy >= 0.65 || features.valence >= 0.65) {
    insights.push({
      title: "Energy profile is helping discoverability",
      tone: "positive",
      body: `The combination of energy (${Math.round(features.energy * 100)}%) and valence (${Math.round(features.valence * 100)}%) suggests an accessible listening posture.`,
    });
  } else {
    insights.push({
      title: "Energy and mood are more restrained",
      tone: "neutral",
      body: `Energy (${Math.round(features.energy * 100)}%) and valence (${Math.round(features.valence * 100)}%) indicate a more niche or subdued profile.`,
    });
  }

  return insights.slice(0, 3);
}
