export const DEFAULT_FETCH_TIMEOUT = 120000;

export async function fetchWithTimeout(
  input: RequestInfo,
  init?: RequestInit,
  timeout: number = DEFAULT_FETCH_TIMEOUT,
): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeout);

  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}
