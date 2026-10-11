import { API_BASE_URL, ApiError } from "@/lib/api";
import type { SignEntry } from "@/lib/vocabulary";

export function SignMediaView({ sign, large }: { sign: SignEntry; large?: boolean }) {
  const cls = "aspect-[4/3] w-full bg-stage object-cover";
  if (sign.media?.type === "image") return <img src={sign.media.url} alt={`Reference for the sign “${sign.label}”`} loading="lazy" className={cls} />;
  if (sign.media?.type === "video")
    return <video src={sign.media.url} className={cls} muted loop playsInline controls={large} autoPlay={large} aria-label={`Reference video for “${sign.label}”`} />;
  return (
    <div className={cls + " flex items-center justify-center text-stage-muted"}>
      <span className="font-serif text-4xl text-stage-foreground/80">{sign.label}</span>
    </div>
  );
}

export function VocabularyStatus({ error, loading }: { error: unknown; loading: boolean }) {
  if (loading)
    return (
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true" aria-label="Loading vocabulary">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-72 animate-pulse rounded-sm bg-card" />
        ))}
      </div>
    );
  const status = error instanceof ApiError ? error.status : undefined;
  const title = !API_BASE_URL
    ? "Backend not connected."
    : status === 404 || status === 501
      ? "Vocabulary isn't available yet."
      : status === undefined
        ? "Backend not reachable."
        : "Couldn't load the vocabulary.";
  const body = !API_BASE_URL
    ? "Supported signs come from the recognition model. Connect the Python service to see them here."
    : status === 404 || status === 501
      ? "The Python service is running but doesn't provide GET /api/vocabulary yet."
      : "Check that the Python service is running, then try again.";
  return (
    <div role="status" className="animate-rise border border-dashed px-6 py-12 text-center">
      <p className="font-serif text-3xl">{title}</p>
      <p className="mx-auto mt-3 max-w-md text-sm text-muted-foreground">{body}</p>
    </div>
  );
}
