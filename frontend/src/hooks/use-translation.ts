
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { Language } from "@/lib/api";

export type TranslationState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; text: string; source: string; lang: Language; id: number }
  | { status: "error"; message: string; unavailable: boolean };

export function useTranslation(source: string, lang: Language) {
  const [state, setState] = useState<TranslationState>({ status: "idle" });
  const seq = useRef(0);

  const run = useCallback(async () => {
    const id = ++seq.current;

    if (!source.trim()) {
      setState({ status: "idle" });
      return;
    }

    setState({ status: "loading" });

    try {
      const result = await api.translate(source, lang);

      if (id !== seq.current) return;

      setState({
        status: "done",
        text: result.text,
        source,
        lang,
        id,
      });
    } catch (error) {
      if (id !== seq.current) return;

      const message =
        error instanceof Error ? error.message : "Translation failed.";

      setState({
        status: "error",
        message,
        unavailable:
          message.includes("Backend not connected") ||
          message.includes("Backend not reachable"),
      });
    }
  }, [source, lang]);

  useEffect(() => {
    void run();
    return () => {
      seq.current++;
    };
  }, [run]);

  return { state, retry: run };
}
