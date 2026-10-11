import { useState } from "react";
import { Copy, Check, X } from "lucide-react";
import type { PredictionResult } from "@/lib/api";
import type { RecState } from "@/hooks/use-backend";
import { cn } from "@/lib/utils";

const statusText: Record<RecState, string> = {
  idle: "Waiting for signs",
  starting: "Starting",
  recording: "Recording",
  processing: "Processing",
  ready: "Result ready",
  empty: "No signs recognized",
  error: "Something went wrong",
};

export function OutputPanel({ state, result, error, connected, onClear }: {
  state: RecState; result: PredictionResult | null; error: string | null; connected: boolean; onClear: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const text = result?.sentence?.trim() ?? "";

  const copy = async () => {
    if (!text) return;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const act = "inline-flex items-center gap-1.5 px-2 py-1 text-xs font-semibold text-paper-muted transition-colors duration-200 hover:text-paper-foreground disabled:opacity-40";

  return (
    <section aria-label="Recognition output" className="flex min-h-[28rem] flex-col bg-paper p-6 text-paper-foreground md:p-8">
      <div className="flex items-center gap-3 border-b border-paper-border pb-4">
        <h2 className="text-[0.6875rem] font-bold uppercase tracking-[0.16em]">Recognition output / 01</h2>
        <div className="ml-auto flex">
          <button className={act} onClick={copy} disabled={!text}>
            {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />} {copied ? "Copied" : "Copy"}
          </button>
          <button className={act} onClick={onClear} disabled={!text && state !== "error" && state !== "empty"}>
            <X className="size-3.5" /> Clear
          </button>
        </div>
      </div>
      <p className="mt-3 flex items-center gap-2 text-xs text-paper-muted" aria-live="polite">
        <span className={cn("size-1.5 rounded-full", state === "recording" ? "animate-pulse bg-destructive" : state === "error" ? "bg-destructive" : "bg-paper-muted")} />
        {statusText[state]}
      </p>

      <div className="flex flex-1 flex-col justify-center py-8" aria-live="polite">
        {state === "error" ? (
          <div>
            <p className="font-serif text-3xl text-destructive">{error}</p>
            <p className="mt-2 text-sm text-paper-muted">Check that the Python service is running, then try again.</p>
          </div>
        ) : text ? (
          <p key={text} className="font-serif text-4xl leading-[1.1] animate-rise md:text-5xl">{text}</p>
        ) : state === "empty" ? (
          <div>
            <p className="font-serif text-3xl">Nothing recognized this time.</p>
            <p className="mt-2 text-sm text-paper-muted">Try signing more slowly, with your hands fully in frame.</p>
          </div>
        ) : (
          <div>
            <svg width="120" height="24" viewBox="0 0 120 24" aria-hidden="true" className="mb-6 text-paper-muted">
              <path d="M2 18 C 14 2, 26 22, 38 8 S 54 16, 60 12" fill="none" stroke="currentColor" strokeWidth="1.3" />
              <path d="M66 12 H118" stroke="currentColor" strokeWidth="1.3" strokeDasharray="10 5" />
            </svg>
            <p className="font-serif text-4xl leading-tight md:text-5xl">Your words will<br />appear here.</p>
            <p className="mt-4 max-w-xs text-sm text-paper-muted">
              {connected ? "Start a recording to see recognized words and sentences." : "Backend not connected — recognition requires the Python service."}
            </p>
          </div>
        )}
      </div>

      {result?.words && result.words.length > 0 && (
        <div className="border-t border-paper-border py-4">
          <ul className="flex flex-wrap gap-x-4 gap-y-2">
            {result.words.map((w, i) => (
              <li key={i} className={cn("text-sm", w.uncertain && "border-b-2 border-warning")} title={w.uncertain ? "Uncertain prediction" : undefined}>
                {w.text}
                {w.confidence !== undefined && <sup className="ml-0.5 text-[0.65rem] text-paper-muted tabular-nums">{Math.round(w.confidence * 100)}%</sup>}
              </li>
            ))}
          </ul>
          {result.words.some((w) => w.uncertain) && (
            <p className="mt-3 text-xs text-paper-muted">Underlined in amber: uncertain prediction.</p>
          )}
        </div>
      )}

      {(result?.confidence !== undefined || result?.latency_ms !== undefined) && (
        <dl className="flex gap-10 border-t border-paper-border pt-4 text-xs">
          {result?.confidence !== undefined && (<div><dt className="text-paper-muted">Confidence</dt><dd className="font-serif text-2xl tabular-nums">{Math.round(result.confidence * 100)}%</dd></div>)}
          {result?.latency_ms !== undefined && (<div><dt className="text-paper-muted">Latency</dt><dd className="font-serif text-2xl tabular-nums">{Math.round(result.latency_ms)}ms</dd></div>)}
        </dl>
      )}
    </section>
  );
}
