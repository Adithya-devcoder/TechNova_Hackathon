import { useCallback, useEffect, useRef, useState } from "react";
import { api, type PredictionResult, type SessionEntry } from "@/lib/api";

export type BackendStatus = "checking" | "connected" | "disconnected" | "unconfigured";
export type RecState = "idle" | "starting" | "recording" | "processing" | "ready" | "empty" | "error";

export function useBackendHealth() {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const check = useCallback(async () => {
    if (!api.isConfigured()) return setStatus("unconfigured");
    try {
      await api.health();
      setStatus("connected");
    } catch {
      setStatus("disconnected");
    }
  }, []);
  useEffect(() => {
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, [check]);
  return { status, recheck: check };
}

export function useRecording(connected: boolean) {
  const [state, setState] = useState<RecState>("idle");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [history, setHistory] = useState<SessionEntry[]>([]);
  const session = useRef<string | null>(null);

  useEffect(() => {
    if (state !== "recording") return;
    const t0 = Date.now() - elapsed * 1000;
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  useEffect(() => {
    if (!connected) return;
    api.sessions().then(setHistory).catch(() => {});
  }, [connected]);

  const start = async () => {
    if (!connected || state === "starting" || state === "recording") return;
    setError(null);
    setState("starting");
    setElapsed(0);
    try {
      const r = await api.startRecording();
      session.current = r.session_id;
      setState("recording");
    } catch (e) {
      setError((e as Error).message);
      setState("error");
    }
  };

  const stop = async () => {
    if (state !== "recording" || !session.current) return;
    setState("processing");
    try {
      const r = await api.stopRecording(session.current);
      setResult(r);
      setState(r.sentence?.trim() ? "ready" : "empty");
      api.sessions().then(setHistory).catch(() => {});
    } catch (e) {
      setError((e as Error).message);
      setState("error");
    } finally {
      session.current = null;
    }
  };

  const clear = () => {
    setResult(null);
    setError(null);
    setElapsed(0);
    setState("idle");
  };

  return { state, result, error, elapsed, history, start, stop, clear };
}
