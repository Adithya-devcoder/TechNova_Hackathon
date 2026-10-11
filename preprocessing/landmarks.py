"""Webcam or video frames -> MediaPipe Holistic landmarks in the layout the model was trained on.

Each frame becomes a float32 array [543, 3] of (x, y, z) in the Kaggle ASL Signs row order:
face 0-467, left hand 468-488, pose 489-521, right hand 522-542, with NaN for parts not detected.
Left/right are the signer's own hands, as in the training data.

Uses the MediaPipe Tasks HolisticLandmarker, because recent mediapipe releases no longer ship the
legacy `mp.solutions.holistic` API. The model file is downloaded on first use.
"""
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from models import native

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'holistic_landmarker.task'
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/holistic_landmarker/'
             'holistic_landmarker/float16/latest/holistic_landmarker.task')

N_ROWS = 543
LEFT_HAND = slice(468, 489)
RIGHT_HAND = slice(522, 543)
# (result field, first row, number of points). The Tasks face mesh has 478 points; training data has 468.
PARTS = (('face_landmarks', 0, 468), ('left_hand_landmarks', 468, 21),
         ('pose_landmarks', 489, 33), ('right_hand_landmarks', 522, 21))


def ensure_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def result_to_array(result):
    """HolisticLandmarkerResult -> [543, 3] float32 with NaN for missing parts."""
    out = np.full((N_ROWS, 3), np.nan, np.float32)
    for field, start, n in PARTS:
        points = getattr(result, field, None) or []
        if points:
            out[start:start + n] = [[p.x, p.y, p.z] for p in points[:n]]
    return out


def has_hands(landmarks):
    """True if either hand was detected in this frame."""
    return bool(np.isfinite(landmarks[LEFT_HAND, 0]).any() or np.isfinite(landmarks[RIGHT_HAND, 0]).any())


class HolisticExtractor:
    """Stateful per-stream extractor (video mode tracks the person across frames).

    All MediaPipe calls run on the native worker thread (models/native.py). Timestamps may restart at 0
    for a new camera session or video: they are shifted so MediaPipe always sees increasing times.
    """

    def __init__(self, model_path=MODEL_PATH, min_confidence=0.5):
        self._landmarker = native.run(self._create, str(ensure_model(model_path)), min_confidence)
        self._last_ts = -1
        self._offset = 0

    @staticmethod
    def _create(model_path, min_confidence):
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_confidence,
            min_hand_landmarks_confidence=min_confidence,
        )
        return vision.HolisticLandmarker.create_from_options(options)

    def process(self, frame_bgr, timestamp_ms):
        """One BGR frame -> [543, 3] landmarks."""
        ts = int(timestamp_ms) + self._offset
        if ts <= self._last_ts:   # a new stream started from 0: continue after the previous one
            self._offset = self._last_ts + 33 - int(timestamp_ms)
            ts = int(timestamp_ms) + self._offset
        self._last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = native.run(self._landmarker.detect_for_video, mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        return result_to_array(result)

    def close(self):
        if self._landmarker is not None:
            native.run(self._landmarker.close)
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_shared = None


def shared_extractor():
    """One extractor for the whole app process, reused by every camera session and video (never closed)."""
    global _shared
    if _shared is None:
        _shared = HolisticExtractor()
    return _shared


def video_to_landmarks(path, progress=None):
    """Video file -> (landmarks [T, 543, 3], timestamps_ms [T])."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f'cannot open video {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames, times = [], []
    extractor = shared_extractor()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = len(frames) * 1000.0 / fps
            frames.append(extractor.process(frame, t))
            times.append(t)
            if progress and total:
                progress(min(len(frames) / total, 1.0))
    finally:
        cap.release()
    if not frames:
        return np.zeros((0, N_ROWS, 3), np.float32), np.zeros(0)
    return np.stack(frames), np.array(times)


# Upper-body skeleton (pose indices) and hand bones (hand indices) for the on-screen overlay
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def draw_landmarks(frame_bgr, landmarks):
    """Return a copy of the frame with pose and hand landmarks drawn on it."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    def draw(points, edges, color):
        xy = points[:, :2] * (w, h)
        for a, b in edges:
            if np.isfinite(xy[[a, b]]).all():
                cv2.line(img, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), color, 2)
        for p in xy:
            if np.isfinite(p).all():
                cv2.circle(img, (int(p[0]), int(p[1])), 3, color, -1)

    draw(landmarks[489:522], POSE_EDGES, (200, 200, 200))
    draw(landmarks[LEFT_HAND], HAND_EDGES, (80, 200, 80))
    draw(landmarks[RIGHT_HAND], HAND_EDGES, (230, 140, 40))
    return img
