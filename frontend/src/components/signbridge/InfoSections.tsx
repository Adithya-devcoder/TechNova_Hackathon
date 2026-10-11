import { useEffect, useState } from "react";
import { api, type SessionEntry } from "@/lib/api";
import { LearnLibrary } from "./LearnLibrary";

export function History({ entries, connected }: { entries: SessionEntry[]; connected: boolean }) {
  return (
    <section aria-labelledby="hist-h" className="grid gap-8 border-t py-14 lg:grid-cols-[1fr_2fr]">
      <div>
        <p className="eyebrow">03 / Activity</p>
        <h2 id="hist-h" className="mt-3 font-serif text-4xl">Recent sessions</h2>
      </div>
      {entries.length === 0 ? (
        <p className="self-end border-l-2 border-primary pl-4 text-sm text-muted-foreground">
          {connected ? "No sessions yet. Completed recordings will be listed here." : "Session history appears once the Python service is connected."}
        </p>
      ) : (
        <ul className="divide-y border-y">
          {entries.map((e) => (
            <li key={e.id} className="grid grid-cols-[1fr_auto] gap-x-6 gap-y-1 py-4 text-sm sm:grid-cols-[1fr_auto_auto_auto]">
              <span className="truncate font-serif text-lg">{e.text || "—"}</span>
              <span className="text-xs text-muted-foreground tabular-nums">{e.created_at ? new Date(e.created_at).toLocaleString() : ""}</span>
              <span className="text-xs uppercase text-muted-foreground">{e.language ?? ""}</span>
              <span className={e.status === "success" ? "text-xs capitalize text-primary" : "text-xs capitalize text-muted-foreground"}>{e.status}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

const steps = [
  ["Capture", "The webcam captures signing movements."],
  ["Recognize", "Computer vision extracts hand and body landmarks for model inference."],
  ["Communicate", "Recognized text can be displayed and translated when translation support is connected."],
];

export function HowItWorks() {
  return (
    <section id="how" aria-labelledby="how-h" className="scroll-mt-8 border-t bg-card">
      <div className="mx-auto max-w-7xl px-5 py-16 md:px-10">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <h2 id="how-h" className="font-serif text-5xl leading-none md:text-6xl">From motion<br /><span className="text-muted-foreground">to meaning.</span></h2>
          <p className="eyebrow">How it works</p>
        </div>
        <ol className="mt-14 grid md:grid-cols-3">
          {steps.map(([t, d], i) => (
            <li key={t} className="border-t py-6 md:border-l md:border-t-0 md:px-8 md:first:border-l-0 md:first:pl-0">
              <span className="font-serif text-7xl leading-none text-primary">0{i + 1}</span>
              <h3 className="mt-6 text-xs font-bold uppercase tracking-[0.18em]">{t}</h3>
              <p className="mt-2 max-w-xs text-sm text-muted-foreground">{d}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

export function SupportedSigns({ connected }: { connected: boolean }) {
  const [signs, setSigns] = useState<string[] | null>(null);
  const [lang, setLang] = useState<string | undefined>();
  useEffect(() => {
    if (!connected) return;
    api.vocabulary().then((v) => { setSigns(v.signs); setLang(v.sign_language); }).catch(() => setSigns(null));
  }, [connected]);

  return (
    <section id="signs" aria-labelledby="signs-h" className="scroll-mt-8 grid gap-8 py-16 lg:grid-cols-[1fr_2fr]">
      <div>
        <p className="eyebrow">04 / Vocabulary</p>
        <h2 id="signs-h" className="mt-3 font-serif text-4xl">What the model knows.</h2>
      </div>
      <div className="text-sm">
        <p className="max-w-lg text-muted-foreground">
          Recognition is limited to the vocabulary{lang ? ` of ${lang}` : " and sign language"} supported by the active model.
          SignBridge does not claim full sign-language fluency or universal recognition.
        </p>
        {signs && signs.length > 0 ? (
          <ul className="mt-6 flex flex-wrap gap-px bg-border p-px">
            {signs.map((s) => <li key={s} className="bg-background px-3 py-2">{s}</li>)}
          </ul>
        ) : (
          <p className="mt-6 border border-dashed px-4 py-5 text-muted-foreground">
            Vocabulary not supplied yet — it loads from the backend once connected.
          </p>
        )}
      </div>
    </section>
  );
}

export function VocabularySection({ connected }: { connected: boolean }) {
  return <LearnLibrary connected={connected} />;
}
