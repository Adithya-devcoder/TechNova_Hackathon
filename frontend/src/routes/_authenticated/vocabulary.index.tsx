import { useMemo, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Search, ArrowUpRight, X } from "lucide-react";
import { Nav } from "@/components/signbridge/Nav";
import { useBackendHealth } from "@/hooks/use-backend";
import { vocabularyQuery } from "@/lib/vocabulary";
import { SignMediaView, VocabularyStatus } from "@/components/signbridge/VocabularyStates";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/_authenticated/vocabulary/")({
  head: () => ({
    meta: [
      { title: "Sign Vocabulary — SignBridge" },
      { name: "description", content: "Explore the signs supported by SignBridge's recognition model." },
      { property: "og:title", content: "Sign Vocabulary — SignBridge" },
      { property: "og:description", content: "Explore the signs supported by SignBridge's recognition model." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: VocabularyPage,
});

function VocabularyPage() {
  const { status, recheck } = useBackendHealth();
  const { data, error, isLoading } = useQuery(vocabularyQuery);
  const [q, setQ] = useState("");
  const [cat, setCat] = useState<string | null>(null);

  const categories = useMemo(
    () => [...new Set((data?.signs ?? []).map((s) => s.category).filter((c): c is string => !!c))].sort(),
    [data],
  );
  const results = useMemo(() => {
    const term = q.trim().toLowerCase();
    return (data?.signs ?? []).filter(
      (s) =>
        (!cat || s.category === cat) &&
        (!term || s.label.toLowerCase().includes(term) || (s.meaning ?? "").toLowerCase().includes(term)),
    );
  }, [data, q, cat]);

  return (
    <div className="min-h-screen overflow-x-hidden">
      <Nav status={status} onRecheck={recheck} />
      <main className="mx-auto max-w-7xl px-5 md:px-10">
        <section className="grid items-end gap-10 border-b py-12 md:grid-cols-[1.2fr_1fr] md:py-16">
          <div>
            <p className="eyebrow animate-rise">Sign Vocabulary / Learn</p>
            <h1 className="mt-6 font-serif text-6xl leading-[0.92] tracking-tight animate-rise [--d:80ms] sm:text-7xl lg:text-8xl">
              Every sign<br />has a <span className="italic text-primary">story.</span>
            </h1>
          </div>
          <p className="max-w-sm text-lg leading-relaxed text-muted-foreground animate-rise [--d:160ms]">
            Explore the signs the current recognition model understands — each one drawn directly from the model's own labels.
            {data?.signLanguage && <> Sign language: <span className="text-foreground">{data.signLanguage}</span>.</>}
          </p>
        </section>

        <section aria-label="Search vocabulary" className="py-10">
          <label htmlFor="vocab-search" className="eyebrow">Search by sign or meaning</label>
          <div className="relative mt-3">
            <Search className="pointer-events-none absolute left-4 top-1/2 size-5 -translate-y-1/2 text-muted-foreground" />
            <input id="vocab-search" type="search" value={q} onChange={(e) => setQ(e.target.value)} disabled={!data}
              placeholder={data ? "Try “hello” or “thank you”" : "Available once the vocabulary loads"}
              className="h-14 w-full border bg-card pl-12 pr-12 text-lg outline-none transition-[border-color,box-shadow] duration-300 placeholder:text-muted-foreground focus:border-primary focus:shadow-[0_0_0_3px_color-mix(in_oklab,var(--color-primary)_18%,transparent)] disabled:opacity-60" />
            {q && (
              <button onClick={() => setQ("")} aria-label="Clear search" className="btn btn-ghost absolute right-2 top-1/2 -translate-y-1/2"><X className="size-4" /></button>
            )}
          </div>
          {categories.length > 0 && (
            <div role="group" aria-label="Filter by category" className="mt-5 flex flex-wrap gap-2">
              {[null, ...categories].map((c) => (
                <button key={c ?? "all"} onClick={() => setCat(c)} aria-pressed={cat === c}
                  className={cn("press border px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em]",
                    cat === c ? "border-primary bg-primary text-primary-foreground" : "text-muted-foreground hover:border-foreground hover:text-foreground")}>
                  {c ?? "All"}
                </button>
              ))}
            </div>
          )}
        </section>

        <section aria-label="Signs" className="pb-20">
          {!data ? (
            <VocabularyStatus loading={isLoading} error={error} />
          ) : data.signs.length === 0 ? (
            <div className="border border-dashed px-6 py-12 text-center">
              <p className="font-serif text-3xl">No signs reported.</p>
              <p className="mt-3 text-sm text-muted-foreground">The model returned an empty vocabulary.</p>
            </div>
          ) : (
            <>
              <p className="mb-5 text-sm text-muted-foreground" aria-live="polite">
                {results.length} of {data.signs.length} signs{q && <> matching “{q}”</>}{cat && <> in {cat}</>}
              </p>
              {results.length === 0 ? (
                <div className="animate-rise border border-dashed px-6 py-12 text-center">
                  <p className="font-serif text-3xl">No signs match.</p>
                  <p className="mt-3 text-sm text-muted-foreground">Try a different word, or clear the filters.</p>
                  <button className="btn btn-outline mt-6" onClick={() => { setQ(""); setCat(null); }}>Clear filters</button>
                </div>
              ) : (
                <ul className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                  {results.map((s, i) => (
                    <li key={s.label} className="animate-rise" style={{ ["--d" as string]: `${Math.min(i, 9) * 40}ms` }}>
                      <Link to="/vocabulary/$label" params={{ label: s.label }}
                        className="group block overflow-hidden rounded-sm border bg-card transition-[transform,border-color] duration-300 hover:-translate-y-1 hover:border-primary/60 focus-visible:-translate-y-1">
                        <SignMediaView sign={s} />
                        <div className="p-5">
                          {s.category && <p className="eyebrow">{s.category}</p>}
                          <p className="mt-1 font-serif text-3xl">{s.label}</p>
                          {s.meaning && <p className="mt-2 line-clamp-2 text-sm text-muted-foreground">{s.meaning}</p>}
                          <span className="mt-4 inline-flex items-center gap-1 text-xs font-semibold uppercase tracking-[0.14em] text-primary">
                            View details <ArrowUpRight className="size-3.5 transition-transform duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
                          </span>
                        </div>
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
