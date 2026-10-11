import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { API_BASE_URL, ApiError } from "@/lib/api";
import { vocabularyQuery, type SignEntry } from "@/lib/vocabulary";
import { SignMediaView } from "./VocabularyStates";

type Phase = "disconnected" | "loading" | "error" | "unavailable" | "empty" | "ready";

export function LearnLibrary({ connected }: { connected: boolean }) {
  const q = useQuery({ ...vocabularyQuery, enabled: connected });
  const [search, setSearch] = useState("");
  const [cat, setCat] = useState<string | null>(null);
  const [open, setOpen] = useState<SignEntry | null>(null);

  const signs = q.data?.signs ?? [];
  const categories = useMemo(
    () => [...new Set(signs.map((s) => s.category).filter((c): c is string => !!c))].sort(),
    [signs],
  );
  const filtered = useMemo(() => {
    const t = search.trim().toLowerCase();
    return signs.filter(
      (s) =>
        (!cat || s.category === cat) &&
        (!t || s.label.toLowerCase().includes(t) || (s.meaning ?? "").toLowerCase().includes(t)),
    );
  }, [signs, search, cat]);

  const status = q.error instanceof ApiError ? q.error.status : undefined;
  const phase: Phase = !connected || !API_BASE_URL
    ? "disconnected"
    : q.isPending
      ? "loading"
      : q.isError
        ? status === 404 || status === 501 ? "unavailable" : "error"
        : signs.length === 0 ? "empty" : "ready";
  const ready = phase === "ready";

  return (
    <section id="learn" aria-labelledby="learn-h" className="scroll-mt-8 border-t py-20 md:py-28">
      <div className="grid gap-10 lg:grid-cols-[1.3fr_1fr] lg:items-end">
        <div>
          <p className="eyebrow">05 / Learn — Sign vocabulary</p>
          <h2 id="learn-h" className="mt-5 font-serif text-5xl leading-[1.02] md:text-7xl">
            Learn the language <br className="hidden md:block" />
            of <em className="text-primary">signs.</em>
          </h2>
        </div>
        <div>
          <p className="max-w-md text-base leading-relaxed text-muted-foreground">
            Explore every sign the recognition model understands — search by name or meaning, filter by category,
            and open a sign to see its reference and an example in use.
          </p>
          <StatusLine phase={phase} count={signs.length} language={q.data?.signLanguage} />
        </div>
      </div>

      <div className="mt-12 flex flex-col gap-5 border-y py-6 md:flex-row md:items-center">
        <label className="relative flex-1">
          <span className="sr-only">Search signs by name or meaning</span>
          <svg aria-hidden viewBox="0 0 24 24" className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            disabled={!ready}
            placeholder={ready ? "Search by sign name or meaning" : "Search becomes available once signs load"}
            className="h-14 w-full rounded-sm border bg-card pl-12 pr-4 text-base outline-none transition-[border-color,box-shadow] placeholder:text-muted-foreground focus:border-primary focus:shadow-[0_0_0_3px_color-mix(in_oklab,var(--primary)_20%,transparent)] disabled:cursor-not-allowed disabled:opacity-60"
          />
        </label>
        {ready && categories.length > 0 && (
          <div role="group" aria-label="Filter by category" className="flex flex-wrap gap-2">
            <Chip active={!cat} onClick={() => setCat(null)}>All</Chip>
            {categories.map((c) => (
              <Chip key={c} active={cat === c} onClick={() => setCat(cat === c ? null : c)}>{c}</Chip>
            ))}
          </div>
        )}
      </div>

      <div className="mt-10">
        {phase === "loading" && (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true" aria-label="Loading vocabulary">
            {Array.from({ length: 6 }).map((_, i) => <div key={i} className="h-96 animate-pulse rounded-sm bg-card" />)}
          </div>
        )}
        {(phase === "disconnected" || phase === "error" || phase === "unavailable" || phase === "empty") && (
          <EmptyLibrary phase={phase} onRetry={() => q.refetch()} />
        )}
        {ready && (filtered.length === 0 ? (
          <div role="status" className="animate-rise border border-dashed px-6 py-16 text-center">
            <p className="font-serif text-3xl">No signs match.</p>
            <p className="mt-2 text-sm text-muted-foreground">Try another word, or clear the filters.</p>
            <button type="button" className="btn mt-6" onClick={() => { setSearch(""); setCat(null); }}>Clear filters</button>
          </div>
        ) : (
          <>
            <p className="mb-5 text-xs uppercase tracking-[0.18em] text-muted-foreground" aria-live="polite">
              {filtered.length} of {signs.length} signs
            </p>
            <ul className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {filtered.map((s, i) => (
                <li key={s.label} className="animate-rise" style={{ ["--d" as string]: `${Math.min(i, 8) * 50}ms` }}>
                  <SignCard sign={s} onOpen={() => setOpen(s)} />
                </li>
              ))}
            </ul>
          </>
        ))}
      </div>

      <Dialog open={!!open} onOpenChange={(o) => !o && setOpen(null)}>
        <DialogContent className="max-w-2xl overflow-hidden border bg-background p-0">
          {open && (
            <>
              <SignMediaView sign={open} large />
              <div className="p-7">
                {open.category && <p className="eyebrow">{open.category}</p>}
                <DialogTitle className="mt-2 font-serif text-4xl font-normal">{open.label}</DialogTitle>
                <DialogDescription className="mt-3 text-base text-muted-foreground">
                  {open.meaning ?? "Meaning not provided by the model data."}
                </DialogDescription>
                {open.example && (
                  <p className="mt-6 border-l-2 border-primary pl-4 text-sm italic">“{open.example}”</p>
                )}
                {!open.media && (
                  <p className="mt-6 text-xs text-muted-foreground">No verified reference media supplied for this sign yet.</p>
                )}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </section>
  );
}

function SignCard({ sign, onOpen }: { sign: SignEntry; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="group flex h-full w-full flex-col overflow-hidden rounded-sm border bg-card text-left transition-[transform,border-color] duration-300 hover:-translate-y-1 hover:border-primary/60 focus-visible:border-primary focus-visible:outline-none"
    >
      <div className="overflow-hidden">
        <div className="transition-transform duration-500 group-hover:scale-[1.03]"><SignMediaView sign={sign} /></div>
      </div>
      <div className="flex flex-1 flex-col p-6">
        {sign.category && <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">{sign.category}</p>}
        <p className="mt-1 font-serif text-3xl">{sign.label}</p>
        {sign.meaning && <p className="mt-2 text-sm text-muted-foreground">{sign.meaning}</p>}
        {sign.example && <p className="mt-4 line-clamp-2 border-l-2 border-primary pl-3 text-xs text-muted-foreground">{sign.example}</p>}
        <span className="mt-auto pt-6 text-xs font-semibold uppercase tracking-[0.18em] text-primary">Learn sign →</span>
      </div>
    </button>
  );
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={`press rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] transition-colors ${active ? "border-primary bg-primary text-primary-foreground" : "text-muted-foreground hover:border-primary/60 hover:text-foreground"}`}
    >
      {children}
    </button>
  );
}

function StatusLine({ phase, count, language }: { phase: Phase; count: number; language?: string | undefined }) {
  const map: Record<Phase, [string, string]> = {
    disconnected: ["bg-muted-foreground", "Recognition service not connected"],
    loading: ["bg-primary animate-pulse", "Loading vocabulary…"],
    error: ["bg-destructive", "Service unreachable"],
    unavailable: ["bg-amber-400", "Service connected · vocabulary endpoint missing"],
    empty: ["bg-muted-foreground", "Service connected · no signs reported"],
    ready: ["bg-primary", `${count} verified signs${language ? ` · ${language}` : ""}`],
  };
  const [dot, text] = map[phase];
  return (
    <p className="mt-6 inline-flex items-center gap-3 border px-4 py-2 text-xs uppercase tracking-[0.16em] text-muted-foreground" role="status">
      <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden />
      {text}
    </p>
  );
}

function EmptyLibrary({ phase, onRetry }: { phase: Exclude<Phase, "loading" | "ready">; onRetry: () => void }) {
  const body: Record<typeof phase, string> = {
    disconnected: "Once the Python recognition service is connected, this library loads the verified signs straight from the model — every card here will be a sign it can actually recognise.",
    unavailable: "The recognition service is running, but it doesn't provide the vocabulary list (GET /api/vocabulary) yet. Signs appear here as soon as it does.",
    error: "We couldn't reach the recognition service to load its vocabulary. Check that it's running, then try again.",
    empty: "The recognition service responded, but reported no supported signs.",
  };
  return (
    <div className="relative animate-rise overflow-hidden rounded-sm border bg-card">
      {/* Reserved card layout — outlines only, no content */}
      <div aria-hidden className="grid gap-6 p-6 opacity-40 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="flex h-80 flex-col rounded-sm border border-dashed">
            <div className="aspect-[4/3] border-b border-dashed" />
            <div className="m-5 h-3 w-1/2 rounded-full bg-border" />
            <div className="mx-5 h-2 w-3/4 rounded-full bg-border/70" />
          </div>
        ))}
      </div>
      <div className="absolute inset-0 flex items-center justify-center bg-[radial-gradient(ellipse_at_center,var(--card)_35%,color-mix(in_oklab,var(--card)_70%,transparent)_80%)] p-6">
        <div role="status" className="max-w-lg text-center">
          <Mark />
          <p className="mt-8 font-serif text-4xl leading-tight md:text-5xl">Your sign library is getting ready.</p>
          <p className="mt-4 text-sm leading-relaxed text-muted-foreground">{body[phase]}</p>
          {(phase === "error" || phase === "unavailable") && (
            <button type="button" className="btn mt-6" onClick={onRetry}>Try again</button>
          )}
        </div>
      </div>
    </div>
  );
}

/** Abstract mark — interlocking arcs; deliberately not any specific sign. */
function Mark() {
  return (
    <svg viewBox="0 0 120 120" className="mx-auto h-28 w-28" fill="none" aria-hidden>
      <circle cx="60" cy="60" r="56" stroke="var(--border)" />
      <circle cx="60" cy="60" r="40" stroke="var(--border)" strokeDasharray="2 6" className="origin-center animate-[spin_40s_linear_infinite] motion-reduce:animate-none" />
      <path d="M34 72c0-16 12-30 26-30s26 14 26 30" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round" />
      <path d="M44 72c0-9 7-18 16-18s16 9 16 18" stroke="var(--foreground)" strokeOpacity=".7" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="60" cy="42" r="3" fill="var(--primary)" />
    </svg>
  );
}
