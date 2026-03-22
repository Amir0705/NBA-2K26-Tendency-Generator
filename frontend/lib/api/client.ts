export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const API_TOKEN_KEY = "n2k_api_token";

function getStoredApiToken(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(API_TOKEN_KEY) ?? "";
}

export function getApiToken(): string {
  return getStoredApiToken().trim();
}

export function hasApiToken(): boolean {
  return Boolean(getApiToken());
}

function buildHeaders(extra?: Record<string, string>): HeadersInit {
  const token = getStoredApiToken();
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(extra ?? {}),
  };
}

export function setApiToken(token: string): void {
  if (typeof window === "undefined") return;
  const value = (token || "").trim();
  if (!value) return;
  window.localStorage.setItem(API_TOKEN_KEY, value);
}

export function clearApiToken(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(API_TOKEN_KEY);
}

export async function apiLogin(username: string, password: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: buildHeaders(),
    body: JSON.stringify({ username, password }),
    cache: "no-store",
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `API login error: ${response.status}`);
  }

  const payload = (await response.json()) as { token?: string };
  const token = String(payload?.token ?? "").trim();
  if (!token) {
    throw new Error("API login did not return a token.");
  }
  setApiToken(token);
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "GET",
    headers: buildHeaders(),
    cache: "no-store",
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `API error: ${response.status}`);
  }

  return (await response.json()) as T;
}
