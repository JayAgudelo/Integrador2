export type AnalysisRoute = "upload" | "search";

export interface FeatureSet {
  acousticness: number;
  danceability: number;
  duration_ms: number;
  energy: number;
  instrumentalness: number;
  key: number;
  liveness: number;
  loudness: number;
  mode: number;
  speechiness: number;
  tempo: number;
  time_signature: number;
  valence: number;
  genre: string;
  [key: string]: number | string | null | undefined;
}

export interface AnalysisSession {
  route: AnalysisRoute;
  genre: string;
  sourceLabel: string;
  sourceValue: string;
  prediction: number;
  features: FeatureSet;
  createdAt: string;
}

export interface OptimizerSession {
  originalPrediction: number;
  optimizedPrediction: number;
  optimizedFeatures: FeatureSet;
  updatedAt: string;
}
