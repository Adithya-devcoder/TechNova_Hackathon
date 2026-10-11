"""
test_landmarks.py — SignBridge live landmark visualisation script.

Reuses the existing signbridge.preprocessing modules for detection.
Draws MediaPipe landmarks on the live webcam feed using MediaPipe's own
drawing utilities (no detection logic is duplicated).

Controls
--------
  Q  — quit and release the webcam.

Output (printed to console each frame)
---------------------------------------
  • Left hand detected: True/False  (21 landmarks)
  • Right hand detected: True/False (21 landmarks)
  • Pose detected: True/False       (33 landmarks)
  • Feature vector shape            (always (258,))

No frames or recordings are saved to disk.
"""

import sys

import cv2
import mediapipe as mp

# ── Reuse existing SignBridge modules ────────────────────────────────────────
from signbridge.preprocessing.capture import WebcamCapture, WebcamCaptureError
from signbridge.preprocessing.landmark_extractor import (
    FEATURE_DIM,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    LandmarkExtractor,
)

# ── MediaPipe drawing utilities (visualisation only, not used in production) ─
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# ── Colour palette ───────────────────────────────────────────────────────────
COLOUR_GREEN  = (0, 200, 0)
COLOUR_BLUE   = (255, 100, 0)
COLOUR_YELLOW = (0, 220, 220)
COLOUR_WHITE  = (255, 255, 255)
COLOUR_BLACK  = (0, 0, 0)
COLOUR_GREY   = (60, 60, 60)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Drawing spec overrides ───────────────────────────────────────────────────
HAND_LANDMARK_SPEC  = mp_drawing.DrawingSpec(color=COLOUR_GREEN,  thickness=2, circle_radius=3)
HAND_CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(0, 140, 0),  thickness=1)
POSE_LANDMARK_SPEC  = mp_drawing.DrawingSpec(color=COLOUR_BLUE,   thickness=2, circle_radius=3)
POSE_CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(180, 60, 0), thickness=1)


def draw_overlay(frame, lm, vector):
    """Render landmarks and HUD text onto *frame* in-place.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR frame — modified in-place.
    lm : FrameLandmarks
        Detection flags from ``LandmarkExtractor.extract()``.
    vector : numpy.ndarray
        Feature vector of shape ``(FEATURE_DIM,)``; only used to display shape.
    raw : mediapipe results object
        Stored in ``extractor.last_results``; used only for drawing.
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark strip at the top for text readability.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), COLOUR_GREY, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── Status indicators ────────────────────────────────────────────────────
    def status_line(label, detected, y, colour_on=COLOUR_GREEN, colour_off=(80, 80, 80)):
        colour = colour_on if detected else colour_off
        marker = "●" if detected else "○"
        cv2.putText(frame, f"{marker} {label}", (10, y),
                    FONT, 0.55, colour, 1, cv2.LINE_AA)

    status_line(
        f"Left hand  ({HAND_LANDMARK_COUNT} pts)" if lm.left_hand_detected else "Left hand",
        lm.left_hand_detected, y=22,
    )
    status_line(
        f"Right hand ({HAND_LANDMARK_COUNT} pts)" if lm.right_hand_detected else "Right hand",
        lm.right_hand_detected, y=44,
    )
    status_line(
        f"Pose       ({POSE_LANDMARK_COUNT} pts)" if lm.pose_detected else "Pose",
        lm.pose_detected, y=66,
        colour_on=COLOUR_BLUE,
    )

    # Feature vector shape (always fixed regardless of detection).
    cv2.putText(
        frame,
        f"Feature vector: {vector.shape}  dtype={vector.dtype}",
        (10, 90),
        FONT, 0.45, COLOUR_YELLOW, 1, cv2.LINE_AA,
    )

    # Bottom hint.
    cv2.putText(
        frame, "Press Q to quit",
        (10, h - 12), FONT, 0.45, COLOUR_WHITE, 1, cv2.LINE_AA,
    )


def draw_landmarks(frame, raw):
    """Draw MediaPipe landmarks using MediaPipe's own drawing utils.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR frame — modified in-place.
    raw :
        ``extractor.last_results`` — the raw MediaPipe Holistic result.
    """
    if raw is None:
        return

    # Pose skeleton.
    if raw.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            raw.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=POSE_LANDMARK_SPEC,
            connection_drawing_spec=POSE_CONNECTION_SPEC,
        )

    # Left hand.
    if raw.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            raw.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_LANDMARK_SPEC,
            connection_drawing_spec=HAND_CONNECTION_SPEC,
        )

    # Right hand.
    if raw.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            raw.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_LANDMARK_SPEC,
            connection_drawing_spec=HAND_CONNECTION_SPEC,
        )


def main():
    print("=" * 60)
    print("SignBridge — Live Landmark Test")
    print(f"  FEATURE_DIM = {FEATURE_DIM}  (left_hand=63, right_hand=63, pose=132)")
    print("  Press Q in the video window to quit.")
    print("=" * 60)

    extractor = LandmarkExtractor(
        static_image_mode=False,        # tracking mode for live video
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cam = WebcamCapture(camera_index=0, width=640, height=480, fps=30)

    try:
        cam.open()
        print(f"  Camera: {cam.actual_width}×{cam.actual_height} @ {cam.actual_fps:.0f} fps\n")

        frame_count = 0

        for frame_bgr in cam.frames():
            frame_count += 1

            # ── Core detection via existing SignBridge module ────────────────
            lm = extractor.extract(frame_bgr)          # FrameLandmarks
            raw = extractor.last_results               # raw results for drawing only

            # ── Draw skeleton on a copy so the original is not mutated ───────
            display = frame_bgr.copy()
            draw_landmarks(display, raw)
            draw_overlay(display, lm, lm.vector)

            # ── Console output (throttled to every 15 frames) ────────────────
            if frame_count % 15 == 1:
                lh = f"YES ({HAND_LANDMARK_COUNT} pts)" if lm.left_hand_detected  else "no"
                rh = f"YES ({HAND_LANDMARK_COUNT} pts)" if lm.right_hand_detected else "no"
                po = f"YES ({POSE_LANDMARK_COUNT} pts)" if lm.pose_detected       else "no"
                print(
                    f"[frame {frame_count:5d}]  "
                    f"left_hand={lh:<14}  right_hand={rh:<14}  "
                    f"pose={po:<14}  vector={lm.vector.shape}"
                )

            # ── Display ──────────────────────────────────────────────────────
            cv2.imshow("SignBridge — Landmark Test (Q to quit)", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nQ pressed — exiting.")
                break

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
