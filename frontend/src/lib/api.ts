/**
 * SignBridge API client.
 * Proposed contracts for the Python FastAPI service — adapt paths/types to the real backend.
 * Nothing here simulates inference; every call hits the configured backend or fails.
 */

export const API_BASE_URL: string | undefined =
  (import.meta.env["VITE_API_BASE_URL"] as string | undefined) || undefined;

export type Language = "en" | "ta" | "hi" | "te";

export interface RecognizedWord {
  text: string;
  confidence?: number; // 0..1, only if backend supplies it
  uncertain?: boolean;
}

export interface PredictionResult {
  sentence: string;
  words?: RecognizedWord[];
  confidence?: number;
  latency_ms?: number;
}

export interface StartRecordingResponse {
  session_id: string;
  started_at?: string;
}

export interface TranslationResult {
  text: string;
  language: Language;
}

export interface SessionEntry {
  id: string;
  text: string;
  created_at?: string;
  language?: Language;
  status: "success" | "empty" | "error";
}

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_BASE_URL) throw new ApiError("Backend not connected");
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError("Backend not reachable");
  }
  if (!res.ok) throw new ApiError(`Request failed (${res.status})`, res.status);
  return res.json() as Promise<T>;
}

export const api = {
  isConfigured: () => Boolean(API_BASE_URL),
  /** MJPEG stream URL produced by the OpenCV backend. */
  streamUrl: (overlay: boolean) =>
    API_BASE_URL ? `${API_BASE_URL}/stream?landmarks=${overlay ? 1 : 0}` : null,
  health: () => request<{ status: string }>("/health"),
  startRecording: () => request<StartRecordingResponse>("/recording/start", { method: "POST" }),
  stopRecording: (sessionId: string) =>
    request<PredictionResult>("/recording/stop", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }),
  translate: (text: string, language: Language) =>
    request<TranslationResult>("/translate", {
      method: "POST",
      body: JSON.stringify({ text, language }),
    }),
  vocabulary: () => request<{ signs: string[]; sign_language?: string }>("/vocabulary"),
  sessions: () => request<SessionEntry[]>("/sessions"),
};
