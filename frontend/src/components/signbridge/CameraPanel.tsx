import { useState } from "react";
import { Circle, Square, RotateCcw, Loader2, Camera, CameraOff } from "lucide-react";
import { api } from "@/lib/api";
import type { RecState } from "@/hooks/use-backend";
import { StatusDot } from "./StatusDot";

const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

interface Props {
  connected: boolean;
  state: RecState;
  elapsed: number;
  onStart: () => void;
  onStop: () => void;
  onClear: () => void;
}

export function CameraPanel({ connected, state, elapsed, onStart, onStop, onClear }: Props) {
  const [overlay, setOverlay] = useState(false);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [streamError, setStreamError] = useState(false);
  const src = connected && cameraEnabled ? api.streamUrl(overlay) : null;
  const recording = state === "recording";
  const busy = state === "starting" || state === "processing";
  const online = connected && !streamError;

  return (
    <section aria-label="Camera input" className="flex flex-col">
      <div className="mb-3 flex items-center justify-between text-xs">
        <span className="eyebrow">Camera input</span>
        <span className="flex items-center gap-2 text-muted-foreground">
          <StatusDot tone={online ? "ok" : "off"} /> {online ? "Stream online" : "Stream offline"}
        </span>
      </div>
      <div className="frame-marks relative aspect-[4/3] overflow-hidden border bg-stage text-stage-foreground">
        {src && !streamError ? (
          <img src={src} alt="Live webcam feed from the recognition service" className="size-full object-cover" onError={() => setStreamError(true)} />
        ) : (
          <div className="flex size-full flex-col items-center justify-center gap-5 px-8 text-center">
            <svg width="96" height="64" viewBox="0 0 96 64" aria-hidden="true" className="text-stage-muted">
              <path d="M4 50 C 24 50, 28 14, 48 32 S 72 14, 92 14" fill="none" stroke="currentColor" strokeWidth="1.2" strokeDasharray="3 4" />
              <circle cx="4" cy="50" r="3" fill="currentColor" />
              <circle cx="92" cy="14" r="3" fill="none" stroke="currentColor" strokeWidth="1.2" />
            </svg>
            <div>
              <p className="font-serif text-2xl md:text-3xl">{!connected ? "Camera not connected" : !cameraEnabled ? "Camera is off" : "Waiting for frames"}</p>
              <p className="mx-auto mt-2 max-w-sm text-sm text-stage-muted">
                {!connected ? "The webcam service is not connected." : !cameraEnabled ? "Switch the camera on when you are ready." : "The service is online but isn't streaming video yet."}
              </p>
            </div>
          </div>
        )}
        {recording && (
          <span className="absolute right-4 top-4 z-10 flex items-center gap-2 bg-stage/80 px-2.5 py-1 text-xs font-bold tracking-widest text-primary">
            <StatusDot tone="live" /> LIVE <span className="font-mono font-normal tabular-nums text-stage-foreground">{fmt(elapsed)}</span>
          </span>
        )}
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="btn btn-outline h-12 px-4"
          onClick={() => {
            setCameraEnabled((enabled) => !enabled);
            setStreamError(false);
          }}
          disabled={!connected || busy}
          aria-pressed={cameraEnabled}
        >
          {cameraEnabled ? <CameraOff className="size-4" /> : <Camera className="size-4" />}
          {cameraEnabled ? "Turn camera off" : "Turn camera on"}
        </button>
        {recording ? (
          <button className="btn btn-primary h-12 px-6" onClick={onStop}><Square className="size-3.5 fill-current" /> Stop recording</button>
        ) : (
          <button className="btn btn-primary h-12 px-6" onClick={onStart} disabled={!connected || !cameraEnabled || busy}>
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Circle className="size-3.5 fill-current" />}
            {state === "processing" ? "Processing…" : state === "starting" ? "Starting…" : "Start recording"}
          </button>
        )}
        <button className="btn btn-outline h-12" onClick={onClear} disabled={recording || busy}>
          <RotateCcw className="size-3.5" /> Reset
        </button>
        <label className="ml-auto flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
          <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} className="size-4 accent-primary" disabled={!connected || !cameraEnabled} />
          Landmark overlay · visualisation
        </label>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">Keep your hands and upper body within the frame.</p>
    </section>
  );
}