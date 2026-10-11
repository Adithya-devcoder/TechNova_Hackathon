import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { Nav } from "@/components/signbridge/Nav";
import { useBackendHealth } from "@/hooks/use-backend";
import { vocabularyQuery } from "@/lib/vocabulary";
import { SignMediaView, VocabularyStatus } from "@/components/signbridge/VocabularyStates";

export const Route = createFileRoute("/_authenticated/vocabulary/$label")({
  head: ({ params }) => ({
    meta: [
      { title: `${params.label} — Sign Vocabulary — SignBridge` },
      { name: "description", content: `Meaning, example and reference for the sign “${params.label}”.` },
      { property: "og:title", content: `${params.label} — SignBridge` },
      { property: "og:description", content: `Meaning, example and reference for the sign “${params.label}”.` },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: SignDetail,
});

function SignDetail() {
  const { label } = Route.useParams();
  const { status, recheck } = useBackendHealth();
  const { data, error, isLoading } = useQuery(vocabularyQuery);
  const sign = data?.signs.find((s) => s.label === label);

  return (
    <div className="min-h-screen overflow-x-hidden">
      <Nav status={status} onRecheck={recheck} />
      <main className="mx-auto max-w-7xl px-5 py-10 md:px-10">
        <Link to="/vocabulary" className="btn btn-ghost -ml-2 text-xs uppercase tracking-[0.14em]">
          <ArrowLeft className="size-4" /> All signs
        </Link>
        <div className="mt-8">
          {!data ? (
            <VocabularyStatus loading={isLoading} error={error} />
          ) : !sign ? (
            <div className="border border-dashed px-6 py-12 text-center">
              <p className="font-serif text-3xl">“{label}” isn't in the model's vocabulary.</p>
              <Link to="/vocabulary" className="btn btn-outline mt-6">Browse supported signs</Link>
            </div>
          ) : (
            <article className="grid gap-10 lg:grid-cols-[1.2fr_1fr]">
              <div className="animate-rise overflow-hidden rounded-sm border">
                <SignMediaView sign={sign} large />
              </div>
              <div className="animate-rise [--d:100ms]">
                <p className="eyebrow">{sign.category ?? "Sign"}</p>
                <h1 className="mt-4 font-serif text-6xl leading-none sm:text-7xl">{sign.label}</h1>
                <dl className="mt-10 divide-y border-y">
                  <Row term="Meaning" value={sign.meaning} />
                  <Row term="Example" value={sign.example && <span className="font-serif text-xl italic">“{sign.example}”</span>} />
                  <Row term="Category" value={sign.category} />
                  <Row term="Reference" value={sign.media ? (sign.media.type === "video" ? "Video" : "Image") : undefined} />
                </dl>
              </div>
            </article>
          )}
        </div>
      </main>
    </div>
  );
}

function Row({ term, value }: { term: string; value: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[7rem_1fr] gap-4 py-4 text-sm">
      <dt className="eyebrow pt-0.5">{term}</dt>
      <dd>{value || <span className="text-muted-foreground">Not provided by the model data.</span>}</dd>
    </div>
  );
}
