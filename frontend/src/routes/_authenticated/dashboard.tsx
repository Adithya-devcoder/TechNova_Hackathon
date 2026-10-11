import { createFileRoute } from "@tanstack/react-router";
import { useBackendHealth, useRecording } from "@/hooks/use-backend";
import { Nav } from "@/components/signbridge/Nav";
import { Hero } from "@/components/signbridge/Hero";
import { CameraPanel } from "@/components/signbridge/CameraPanel";
import { OutputPanel } from "@/components/signbridge/OutputPanel";
import { TranslationPanel } from "@/components/signbridge/TranslationPanel";
import { History, HowItWorks, SupportedSigns, VocabularySection } from "@/components/signbridge/InfoSections";

export const Route = createFileRoute("/_authenticated/dashboard")({
  head: () => ({
    meta: [
      { title: "Dashboard — SignBridge" },
      { name: "description", content: "Turn continuous signs into clear, readable text with SignBridge." },
      { property: "og:title", content: "SignBridge — Live sign language to text" },
      { property: "og:description", content: "Turn continuous signs into clear, readable text." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  const { status, recheck } = useBackendHealth();
  const connected = status === "connected";
  const rec = useRecording(connected);
  const clear = () => {
    if (rec.result?.sentence && !window.confirm("Clear the recognized text?")) return;
    rec.clear();
  };

  return (
    <div className="min-h-screen overflow-x-hidden">
      <Nav status={status} onRecheck={recheck} />
      <main className="mx-auto max-w-7xl px-5 md:px-10">
        <Hero />

        <div id="translate" className="scroll-mt-4 pt-12">
          <div className="mb-6 flex flex-wrap items-baseline justify-between gap-4">
            <p className="eyebrow">01 / Live translation workspace</p>
            {status !== "connected" && status !== "checking" && (
              <p role="status" className="text-xs text-warning">
                Backend not connected — recognition requires the Python service
                {status === "unconfigured" ? " (set VITE_API_BASE_URL)." : "."}
              </p>
            )}
          </div>
          <div className="grid gap-8 pb-14 lg:grid-cols-[1.45fr_1fr] lg:gap-0">
            <div className="lg:pr-10">
              <CameraPanel connected={connected} state={rec.state} elapsed={rec.elapsed} onStart={rec.start} onStop={rec.stop} onClear={clear} />
            </div>
            <div className="lg:mt-7">
              <OutputPanel state={rec.state} result={rec.result} error={rec.error} connected={connected} onClear={clear} />
            </div>
          </div>
        </div>

        <TranslationPanel source={rec.result?.sentence?.trim() ?? ""} connected={connected} />
        <History entries={rec.history} connected={connected} />
      </main>
      <HowItWorks />
      <div className="mx-auto max-w-7xl px-5 md:px-10">
        <SupportedSigns connected={connected} />
        <VocabularySection connected={connected} />
      </div>
      <footer className="border-t">
        <div className="mx-auto flex max-w-7xl flex-wrap justify-between gap-2 px-5 py-8 text-xs text-muted-foreground md:px-10">
          <span className="font-bold tracking-[0.32em] text-foreground">SIGNBRIDGE</span>
          <span>Human connection, reimagined.</span>
        </div>
      </footer>
    </div>
  );
}
