import { useEffect, useRef, useState } from "react";
import { Copy, Check, Loader2, RotateCw, Volume2, Square } from "lucide-react";
import type { Language } from "@/lib/api";
import { useTranslation } from "@/hooks/use-translation";
import { useSpeech } from "@/hooks/use-speech";
import { cn } from "@/lib/utils";

const languages: { code: Language; native: string; label: string }[] = [
  { code: "en", native: "English", label: "English" },
  { code: "ta", native: "தமிழ்", label: "Tamil" },
  { code: "hi", native: "हिन्दी", label: "Hindi" },
  { code: "te", native: "తెలుగు", label: "Telugu" },
];

export function TranslationPanel({ source }: { source: string; connected?: boolean }) {
  const [lang, setLang] = useState<Language>("ta");
  const [copied, setCopied] = useState(false);
  const [voiceOn, setVoiceOn] = useState(false);
  const target = languages.find((l) => l.code === lang)!;
  const { state, retry } = useTranslation(source, lang);
  const speech = useSpeech(lang);
  const out = state.status === "done" ? state.text : "";

  // Speak each new successful translation exactly once.
  const spokenId = useRef<number | null>(null);
  const lastSeenId = useRef<number | null>(null);
  useEffect(() => {
    if (state.status !== "done") return;
    const isNew = lastSeenId.current !== state.id;
    lastSeenId.current = state.id;
    if (!voiceOn || !isNew || spokenId.current === state.id) return;
    spokenId.current = state.id;
    speech.speak(state.text);
  }, [state, voiceOn, speech]);

  // Cancel stale speech when a new translation starts or fails.
  useEffect(() => {
    if (state.status === "loading" || state.status === "error") speech.stop();
  }, [state.status, speech.stop]);

  const toggleVoice = () => {
    if (voiceOn) {
      speech.stop();
      setVoiceOn(false);
    } else {
      setVoiceOn(true);
      if (out) speech.speak(out);
    }
  };

  const copy = async () => {
    await navigator.clipboard.writeText(out);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const voiceNote =
    !voiceOn || speech.supported === null
      ? null
      : !speech.supported
        ? "This browser does not support voice output."
        : null;

  return (
    <section aria-labelledby="translate-h" className="grid gap-8 border-t py-14 lg:grid-cols-[1fr_2fr]">
      <div>
        <p className="eyebrow">02 / Translate</p>
        <h2 id="translate-h" className="mt-3 font-serif text-4xl">Carry it further.</h2>
        <p className="mt-3 max-w-xs text-sm text-muted-foreground">The recognized sentence is translated automatically into the language you choose.</p>
      </div>
      <div>
        <div role="radiogroup" aria-label="Target language" className="grid grid-cols-2 border sm:grid-cols-4">
          {languages.map((l) => (
            <button key={l.code} role="radio" aria-checked={lang === l.code}
              onClick={() => setLang(l.code)}
              className={cn("border-r px-4 py-3 text-left transition-colors duration-200 last:border-r-0",
                lang === l.code ? "bg-primary text-primary-foreground" : "hover:bg-card")}>
              <span className="block text-base font-semibold">{l.native}</span>
              <span className={cn("text-xs", lang === l.code ? "opacity-70" : "text-muted-foreground")}>{l.label}</span>
            </button>
          ))}
        </div>

        <div className="mt-6 grid gap-px bg-border md:grid-cols-2">
          <div className="bg-card p-5">
            <p className="eyebrow">Source · Recognized</p>
            <p className="mt-3 min-h-12 text-lg">{source || <span className="text-muted-foreground">No recognized text yet.</span>}</p>
          </div>
          <div className="bg-card p-5" aria-live="polite">
            <div className="flex items-center justify-between gap-2">
              <p className="eyebrow">Target · {target.label}</p>
              <button className="btn btn-ghost h-7 text-xs" onClick={copy} disabled={!out}>
                {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />} Copy
              </button>
            </div>
            <div className="mt-3 min-h-12 text-lg">
              {state.status === "done" && <p key={state.id} lang={lang} className="animate-rise">{state.text}</p>}
              {state.status === "loading" && <p className="flex items-center gap-2 text-muted-foreground"><Loader2 className="size-4 animate-spin" /> Translating…</p>}
              {state.status === "error" && (
                <p className={state.unavailable ? "text-warning" : "text-destructive"}>
                  {state.unavailable ? "Translation service not connected." : state.message}
                </p>
              )}
              {state.status === "idle" && <p className="text-muted-foreground">—</p>}
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-4">
          <button
            type="button"
            role="switch"
            aria-checked={voiceOn}
            onClick={toggleVoice}
            className="group inline-flex items-center gap-3 text-sm font-semibold"
          >
            <span className={cn("relative h-5 w-9 border transition-colors duration-200", voiceOn ? "border-primary bg-primary" : "bg-card")}>
              <span className={cn("absolute top-0.5 size-3.5 transition-all duration-200", voiceOn ? "left-[1.1rem] bg-primary-foreground" : "left-0.5 bg-muted-foreground")} />
            </span>
            Voice Output
            <span className="eyebrow">{voiceOn ? "On" : "Off"}</span>
          </button>

          {speech.speaking && (
            <>
              <span className="inline-flex items-center gap-2 text-xs text-muted-foreground" aria-live="polite">
                <Volume2 className="size-3.5 animate-pulse text-primary" /> Speaking
              </span>
              <button className="btn btn-ghost h-8 text-xs" onClick={speech.stop}>
                <Square className="size-3" /> Stop speaking
              </button>
            </>
          )}

          {state.status === "error" && source && (
            <button className="btn btn-outline ml-auto" onClick={() => void retry()}>
              <RotateCw className="size-4" /> Try again
            </button>
          )}
        </div>
        {voiceNote && <p className="mt-3 text-xs text-warning animate-drop">{voiceNote}</p>}
      </div>
    </section>
  );
}
