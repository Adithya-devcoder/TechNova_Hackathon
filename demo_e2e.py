"""
demo_e2e.py — SignBridge end-to-end demo
=========================================
Captures live webcam video, extracts hand and body landmarks using the
existing CV pipeline, buffers a complete signed sentence using SentenceBuffer,
converts the landmarks to the (T, 543, 3) format the model expects, saves the
clip as a .npy file, and calls Member 1's trained model via a subprocess.

Cross-environment design
------------------------
The CV pipeline (mediapipe, opencv, numpy 1.26) lives in this project's .venv.
The ML model (torch, numpy 2.x) lives in a separate worktree at:

    C:\\Users\\ASUS\\Downloads\\TechNova\\signbridge-model

Rather than attempting cross-venv imports (which would fail due to incompatible
numpy versions), the demo:

1. Saves the (T, 543, 3) clip as a temporary .npy file.
2. Calls the model worktree's Python interpreter directly as a subprocess,
   passing the .npy path as a command-line argument.
3. The model subprocess runs `python -m models.predict_clip <clip.npy>` which
   is an existing, unmodified script in the model worktree.
4. JSON output is read from the subprocess stdout and displayed here.

No model files are copied. No model code is modified. No branches are changed.

Controls
--------
  SPACE  — finish recording the current sentence and run the model.
           Recording restarts automatically after the result is displayed.
  R      — restart recording (discard current sentence without running model).
  Q      — quit and release all resources.

Output
------
  • Live video window with HUD showing recording state and detection status.
  • Console: predicted text, per-word confidences, and model latency.
  • Clip saved to: clips/clip_<timestamp>.npy  (relative to this script).

Usage
-----
    .venv\\Scripts\\python.exe demo_e2e.py

Optional arguments (edit the CONFIG block below to change defaults):
    MODEL_WORKTREE   path to the signbridge-model checkout
    MODEL_PYTHON     path to the model worktree's python interpreter
    CLIPS_DIR        directory where .npy clips are saved
    MIN_FRAMES       minimum frames required to attempt recognition
    MAX_FRAMES       hard cap on sentence length (frames)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Reuse existing SignBridge CV modules ─────────────────────────────────────
from signbridge.preprocessing.capture import WebcamCapture, WebcamCaptureError
from signbridge.preprocessing.landmark_extractor import (
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    LandmarkExtractor,
)
from signbridge.preprocessing.sentence_buffer import SentenceBuffer, SentenceResult

# ── Configuration — edit these if paths change ───────────────────────────────
CONFIG = {
    "MODEL_WORKTREE": r"C:\Users\ASUS\Downloads\TechNova\signbridge-model",
    "MODEL_PYTHON":   r"C:\Users\ASUS\Downloads\TechNova\signbridge-model\.venv\Scripts\python.exe",
    "CLIPS_DIR":      "clips",
    "CAMERA_INDEX":   0,
    "CAM_WIDTH":      640,
    "CAM_HEIGHT":     480,
    "CAM_FPS":        30,
    "MIN_FRAMES":     10,    # ~0.3 s at 30 fps — short clips are discarded
    "MAX_FRAMES":     600,   # 20 s safety cap
}

# ── Colours and font ─────────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_SIMPLEX
C_GREEN     = (0, 220, 0)
C_RED       = (30, 30, 220)
C_YELLOW    = (0, 220, 220)
C_WHITE     = (255, 255, 255)
C_GREY_BG   = (45, 45, 45)
C_ORANGE    = (0, 165, 255)
C_BLUE      = (220, 100, 0)

# ── MediaPipe drawing ────────────────────────────────────────────────────────
import mediapipe as mp
mp_drawing  = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

HAND_LM_SPEC = mp_drawing.DrawingSpec(color=C_GREEN,  thickness=2, circle_radius=3)
HAND_CN_SPEC = mp_drawing.DrawingSpec(color=(0,140,0), thickness=1)
POSE_LM_SPEC = mp_drawing.DrawingSpec(color=C_BLUE,   thickness=2, circle_radius=3)
POSE_CN_SPEC = mp_drawing.DrawingSpec(color=(140,50,0),thickness=1)


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def run_model(clip_path: Path) -> dict:
    """Call the model worktree's predict_clip script and return parsed JSON.

    The model is invoked as a subprocess using its own Python interpreter
    so that torch and the newer numpy version are available without touching
    the CV pipeline's environment.

    Parameters
    ----------
    clip_path:
        Path to the saved .npy file of shape (T, 543, 3) float32.

    Returns
    -------
    dict with keys 'text', 'words', 'latency_ms'.
    Returns {'text': '', 'words': [], 'latency_ms': 0, 'error': <msg>}
    on failure.
    """
    model_python = CONFIG["MODEL_PYTHON"]
    worktree     = CONFIG["MODEL_WORKTREE"]

    cmd = [model_python, "-m", "models.predict_clip", str(clip_path)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"text": "", "words": [], "latency_ms": 0,
                "error": "model subprocess timed out after 30 s"}
    except FileNotFoundError as exc:
        return {"text": "", "words": [], "latency_ms": 0,
                "error": f"model Python not found: {exc}"}

    # Warnings from check_clip() go to stdout too — find the JSON line.
    stdout_lines = proc.stdout.strip().splitlines()
    warnings_seen = [l for l in stdout_lines if l.startswith("WARNING:")]
    for w in warnings_seen:
        print(f"  [model] {w}")

    # The JSON block starts with '{' — take everything from that line onward.
    json_start = next((i for i, l in enumerate(stdout_lines) if l.startswith("{")), None)
    if json_start is None or proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        return {"text": "", "words": [], "latency_ms": 0,
                "error": f"model failed (rc={proc.returncode}): {err[:200]}"}

    json_text = "\n".join(stdout_lines[json_start:])
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {"text": "", "words": [], "latency_ms": 0,
                "error": f"JSON parse error: {exc}"}


# ---------------------------------------------------------------------------
# .npy saving
# ---------------------------------------------------------------------------

def save_clip(holistic: np.ndarray, clips_dir: Path) -> Path:
    """Save (T, 543, 3) float32 array to a timestamped .npy file.

    Returns the **absolute** path to the saved file so the model subprocess
    can locate it regardless of its working directory.
    """
    clips_dir = clips_dir.resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]
    path = clips_dir / f"clip_{ts}.npy"
    np.save(path, holistic)
    return path


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, state: str, frame_count: int, lm, last_result: dict | None) -> None:
    """Render recording state and detection status onto *frame* in-place."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), C_GREY_BG, -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # Recording indicator.
    if state == "RECORDING":
        dot_colour = C_RED
        label = f"● REC  {frame_count} frames"
    else:
        dot_colour = C_YELLOW
        label = "○ IDLE  — press SPACE to start"

    cv2.putText(frame, label, (10, 22), FONT, 0.60, dot_colour, 2, cv2.LINE_AA)

    # Detection status.
    def status(text, detected, y, colour_on=C_GREEN):
        c = colour_on if detected else (70, 70, 70)
        m = "●" if detected else "○"
        cv2.putText(frame, f"{m} {text}", (10, y), FONT, 0.48, c, 1, cv2.LINE_AA)

    status(f"Left hand  ({HAND_LANDMARK_COUNT} pts)", lm.left_hand_detected,  44)
    status(f"Right hand ({HAND_LANDMARK_COUNT} pts)", lm.right_hand_detected, 62)
    status(f"Pose       ({POSE_LANDMARK_COUNT} pts)", lm.pose_detected,       80, colour_on=C_BLUE)

    # Last prediction (if any).
    if last_result and last_result.get("text"):
        text_label = f'"{last_result["text"]}"'
        cv2.putText(frame, text_label, (10, 104), FONT, 0.52, C_YELLOW, 1, cv2.LINE_AA)

    # Bottom key hint.
    hint = "SPACE=flush  R=restart  Q=quit"
    cv2.putText(frame, hint, (10, h - 10), FONT, 0.40, C_WHITE, 1, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, raw) -> None:
    """Draw MediaPipe landmarks on *frame* using the raw Holistic results."""
    if raw is None:
        return
    if raw.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, raw.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=POSE_LM_SPEC,
            connection_drawing_spec=POSE_CN_SPEC,
        )
    if raw.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, raw.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_LM_SPEC,
            connection_drawing_spec=HAND_CN_SPEC,
        )
    if raw.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, raw.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_LM_SPEC,
            connection_drawing_spec=HAND_CN_SPEC,
        )


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate config before opening camera.
    if not Path(CONFIG["MODEL_PYTHON"]).is_file():
        print(f"ERROR: model Python not found at {CONFIG['MODEL_PYTHON']}")
        print("       Update CONFIG['MODEL_PYTHON'] in demo_e2e.py")
        sys.exit(1)
    if not Path(CONFIG["MODEL_WORKTREE"]).is_dir():
        print(f"ERROR: model worktree not found at {CONFIG['MODEL_WORKTREE']}")
        sys.exit(1)

    clips_dir = Path(CONFIG["CLIPS_DIR"])

    print("=" * 62)
    print("SignBridge — End-to-End Demo")
    print(f"  Model worktree : {CONFIG['MODEL_WORKTREE']}")
    print(f"  Clips saved to : {clips_dir.resolve()}")
    print(f"  Min frames     : {CONFIG['MIN_FRAMES']}")
    print("  Controls  SPACE=flush sentence  R=restart  Q=quit")
    print("=" * 62)

    extractor = LandmarkExtractor(static_image_mode=False)
    buf       = SentenceBuffer(
        max_frames=CONFIG["MAX_FRAMES"],
        min_frames=CONFIG["MIN_FRAMES"],
    )
    cam = WebcamCapture(
        camera_index=CONFIG["CAMERA_INDEX"],
        width=CONFIG["CAM_WIDTH"],
        height=CONFIG["CAM_HEIGHT"],
        fps=CONFIG["CAM_FPS"],
    )

    last_result: dict | None = None
    frame_count = 0      # frames buffered in current sentence

    try:
        cam.open()
        print(f"\n  Camera: {cam.actual_width}×{cam.actual_height} @ {cam.actual_fps:.0f} fps")
        print("  Press SPACE when you have finished signing a sentence.\n")

        # Start in IDLE — user presses SPACE to begin.
        buf_state = "IDLE"

        for frame_bgr in cam.frames():

            # ── Landmark extraction ──────────────────────────────────────
            lm  = extractor.extract(frame_bgr)
            raw = extractor.last_results       # for drawing only

            # ── Append to buffer if recording ────────────────────────────
            if buf.is_recording:
                buf.append(lm)
                frame_count = buf.frame_count

            # ── Draw landmarks + HUD ─────────────────────────────────────
            display = frame_bgr.copy()
            draw_landmarks(display, raw)
            draw_hud(display, buf.state, frame_count, lm, last_result)
            cv2.imshow("SignBridge — E2E Demo (SPACE=flush  R=restart  Q=quit)", display)

            # ── Key handling ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQ pressed — exiting.")
                break

            elif key == ord("r"):
                buf.reset()
                buf.start()
                frame_count = 0
                last_result = None
                print("  [demo] Recording restarted.")

            elif key == ord(" "):
                # SPACE: if IDLE, start recording; if RECORDING, flush.
                if not buf.is_recording:
                    buf.start()
                    frame_count = 0
                    print("  [demo] Recording started — sign a sentence, then press SPACE again.")
                else:
                    print(f"\n  [demo] Flushing {buf.frame_count} frames …")
                    result = buf.flush()
                    frame_count = 0

                    if result is None:
                        print(f"  [demo] Too few frames (< {CONFIG['MIN_FRAMES']}) — discarded.")
                        buf.start()   # auto-restart
                    else:
                        # ── Save clip ────────────────────────────────────
                        clip_path = save_clip(result.holistic, clips_dir)
                        print(f"  [demo] Clip saved: {clip_path}")
                        print(f"         Shape: {result.holistic.shape}, "
                              f"dtype: {result.holistic.dtype}")
                        lh_frames = sum(f.left_hand_detected  for f in result.frames)
                        rh_frames = sum(f.right_hand_detected for f in result.frames)
                        po_frames = sum(f.pose_detected       for f in result.frames)
                        print(f"         LH detected: {lh_frames}/{len(result.frames)} frames  "
                              f"RH: {rh_frames}/{len(result.frames)}  "
                              f"Pose: {po_frames}/{len(result.frames)}")

                        # ── Run model ────────────────────────────────────
                        print("  [demo] Running model …")
                        t0 = time.perf_counter()
                        prediction = run_model(clip_path)
                        wall_ms = (time.perf_counter() - t0) * 1000

                        print(f"\n  ┌─ Prediction ─────────────────────────────")
                        print(f"  │  text       : {prediction.get('text', '') or '(nothing recognised)'}")
                        print(f"  │  latency    : {prediction.get('latency_ms', 0):.1f} ms (model)")
                        print(f"  │  wall time  : {wall_ms:.0f} ms (subprocess)")
                        if "error" in prediction:
                            print(f"  │  ERROR      : {prediction['error']}")
                        words = prediction.get("words", [])
                        if words:
                            print(f"  │  words ({len(words)}):")
                            for w in words:
                                conf = w.get("confidence", 0)
                                uncertain = "?" if w.get("uncertain") else " "
                                print(f"  │    {uncertain} {w['word']:<15} conf={conf:.3f}  "
                                      f"frames [{w.get('start','?')}–{w.get('end','?')}]")
                        print(f"  └──────────────────────────────────────────\n")

                        last_result = prediction

                        # Auto-restart for the next sentence.
                        buf.start()

    except WebcamCaptureError as exc:
        print(f"\nCamera error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.release()
        extractor.close()
        cv2.destroyAllWindows()
        print("Resources released. Bye!")


if __name__ == "__main__":
    main()
