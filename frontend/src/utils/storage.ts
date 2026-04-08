import { AnalysisSession, OptimizerSession } from "../types/analysis";

export const STORAGE_KEYS = {
  analysis: "mpp:last-analysis",
  optimizer: "mpp:last-optimizer",
  analyzeMode: "mpp:last-analyze-mode",
  uploadDraft: "mpp:upload-draft",
  searchDraft: "mpp:search-draft",
};

const isBrowser = (): boolean => typeof window !== "undefined" && !!window.localStorage;

export function loadJson<T>(key: string): T | null {
  if (!isBrowser()) {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export function saveJson<T>(key: string, value: T): void {
  if (!isBrowser()) {
    return;
  }

  window.localStorage.setItem(key, JSON.stringify(value));
}

export function removeStoredValue(key: string): void {
  if (!isBrowser()) {
    return;
  }

  window.localStorage.removeItem(key);
}

export function loadAnalysisSession(): AnalysisSession | null {
  return loadJson<AnalysisSession>(STORAGE_KEYS.analysis);
}

export function saveAnalysisSession(session: AnalysisSession): void {
  saveJson(STORAGE_KEYS.analysis, session);
}

export function loadOptimizerSession(): OptimizerSession | null {
  return loadJson<OptimizerSession>(STORAGE_KEYS.optimizer);
}

export function saveOptimizerSession(session: OptimizerSession): void {
  saveJson(STORAGE_KEYS.optimizer, session);
}
