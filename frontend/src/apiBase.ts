/** API origin for production (e.g. Railway/Render). Empty = same origin (local Vite proxy). */
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/** Render free instances sleep; the first request can hang or fail until the API wakes. */
export async function fetchApi(
  path: string,
  init: RequestInit = {},
  { attempts = 5, timeoutMs = 15000 }: { attempts?: number; timeoutMs?: number } = {},
): Promise<Response> {
  let lastError: unknown;
  for (let i = 0; i < attempts; i += 1) {
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(apiUrl(path), { ...init, signal: ctrl.signal });
      window.clearTimeout(timer);
      if (res.ok || res.status < 500) return res;
      lastError = new Error(`HTTP ${res.status}`);
    } catch (err) {
      window.clearTimeout(timer);
      lastError = err;
    }
    if (i < attempts - 1) {
      await new Promise((r) => window.setTimeout(r, 1500 * (i + 1)));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Request failed");
}
