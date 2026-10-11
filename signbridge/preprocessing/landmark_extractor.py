"""
signbridge.preprocessing.landmark_extractor
============================================
Per-frame MediaPipe Holistic landmark extraction.

Feature layout
--------------
For each frame the extractor produces a flat feature vector whose layout is
**fixed and deterministic regardless of detection outcome**.

Coordinate convention
~~~~~~~~~~~~~~~~~~~~~
All landmark coordinates are taken directly from MediaPipe's normalised output:

* ``x`` — horizontal position, 0.0 = left edge, 1.0 = right edge.
* ``y`` — vertical position,   0.0 = top  edge, 1.0 = bottom edge.
* ``z`` — depth relative to the wrist (hand) or hip (pose).  Scale matches ``x``.
* ``visibility`` — pose landmarks only; confidence that the landmark is visible (0–1).

Three values (x, y, z) are stored per hand landmark; four values (x, y, z,
visibility) are stored per pose landmark.

Segment ordering
~~~~~~~~~~~~~~~~
+------------------+------------------------+-------+-------+--------+
| Segment          | Source                 | Count | Vals  | Offset |
+==================+========================+=======+=======+========+
| Left hand        | MediaPipe left hand    |    21 |     3 |      0 |
+------------------+------------------------+-------+-------+--------+
| Right hand       | MediaPipe right hand   |    21 |     3 |     63 |
+------------------+------------------------+-------+-------+--------+
| Pose (body)      | MediaPipe pose         |    33 |     4 |    126 |
+------------------+------------------------+-------+-------+--------+
| **Total**        |                        |    75 |       | **258**|
+------------------+------------------------+-------+-------+--------+

``FEATURE_DIM = 258``

Face landmarks are **excluded** from this version but can be added as an
additional segment at the end without breaking existing offsets.

Missing-landmark handling
~~~~~~~~~~~~~~~~~~~~~~~~~
If a hand or the pose is not detected in a frame, the corresponding segment is
filled with **zeros** so the feature vector always has shape ``(258,)``.  The ML
model must be trained with zero-filled vectors in mind.

Usage
-----
::

    extractor = LandmarkExtractor()
    frame_bgr = ...            # numpy array, uint8, BGR
    lm = extractor.extract(frame_bgr)
    vector: np.ndarray = lm.to_vector()   # shape (258,), dtype float32

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-dimension constants — used by SequenceBuilder and the ML contract
# ---------------------------------------------------------------------------

#: Landmarks per hand (MediaPipe hand model).
HAND_LANDMARK_COUNT: int = 21

#: Values stored per hand landmark (x, y, z).
HAND_VALUES_PER_LANDMARK: int = 3

#: Dimension contributed by one hand.
HAND_DIM: int = HAND_LANDMARK_COUNT * HAND_VALUES_PER_LANDMARK  # 63

#: Landmarks in the MediaPipe pose model.
POSE_LANDMARK_COUNT: int = 33

#: Values stored per pose landmark (x, y, z, visibility).
POSE_VALUES_PER_LANDMARK: int = 4

#: Dimension contributed by the pose.
POSE_DIM: int = POSE_LANDMARK_COUNT * POSE_VALUES_PER_LANDMARK  # 132

#: Total feature dimension for one frame.
#:
#: Layout (byte/element offsets):
#:   [0:63]   left hand  (21 × 3)
#:   [63:126] right hand (21 × 3)
#:   [126:258] pose      (33 × 4)
FEATURE_DIM: int = HAND_DIM + HAND_DIM + POSE_DIM  # 258

# Segment slice objects for convenient indexing.
LEFT_HAND_SLICE: slice = slice(0, HAND_DIM)
RIGHT_HAND_SLICE: slice = slice(HAND_DIM, 2 * HAND_DIM)
POSE_SLICE: slice = slice(2 * HAND_DIM, FEATURE_DIM)


@dataclass
class FrameLandmarks:
    """Landmarks extracted from a single frame.

    Attributes
    ----------
    left_hand_detected:
        True if MediaPipe detected a left hand in this frame.
    right_hand_detected:
        True if MediaPipe detected a right hand in this frame.
    pose_detected:
        True if MediaPipe detected a pose (body) in this frame.
    vector:
        Flat float32 array of shape ``(FEATURE_DIM,)`` = ``(258,)``.
        Segments for undetected landmarks are zero-filled.

    Notes
    -----
    ``left`` / ``right`` refer to the **signer's** left and right hand as
    identified by MediaPipe Holistic, not the camera's left and right.
    """

    left_hand_detected: bool = False
    right_hand_detected: bool = False
    pose_detected: bool = False
    vector: np.ndarray = field(
        default_factory=lambda: np.zeros(FEATURE_DIM, dtype=np.float32)
    )

    def __post_init__(self) -> None:
        if self.vector.shape != (FEATURE_DIM,):
            raise ValueError(
                f"FrameLandmarks.vector must have shape ({FEATURE_DIM},), "
                f"got {self.vector.shape}"
            )
        if self.vector.dtype != np.float32:
            raise ValueError(
                f"FrameLandmarks.vector must be float32, got {self.vector.dtype}"
            )


class LandmarkExtractor:
    """Extracts hand and body landmarks from a single BGR frame.

    Uses MediaPipe Holistic in static-image mode for per-frame extraction.
    For live video (streaming), set ``static_image_mode=False`` to enable
    tracking and reduce latency (the default).

    Parameters
    ----------
    static_image_mode:
        Pass ``True`` when processing offline frames (e.g. dataset extraction).
        Pass ``False`` (default) for real-time webcam streaming.
    min_detection_confidence:
        Minimum confidence for initial detection (0.0–1.0).
    min_tracking_confidence:
        Minimum confidence for landmark tracking (only relevant when
        ``static_image_mode=False``).

    Notes
    -----
    Call ``close()`` (or use as a context manager) when done to release
    MediaPipe resources.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Caches the raw MediaPipe result from the most recent extract() call.
        # Intended for debug/visualisation scripts only — not part of the ML contract.
        self._last_results: Optional[object] = None
        logger.info(
            "LandmarkExtractor: MediaPipe Holistic initialised "
            "(static_image_mode=%s, min_detection=%.2f, min_tracking=%.2f)",
            static_image_mode,
            min_detection_confidence,
            min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        """Extract landmarks from one BGR frame.

        Parameters
        ----------
        frame_bgr:
            NumPy array of shape ``(H, W, 3)``, dtype ``uint8``, **BGR** colour
            order (as returned by ``cv2.VideoCapture.read``).

        Returns
        -------
        FrameLandmarks
            Detected status flags and a flat float32 feature vector of shape
            ``(FEATURE_DIM,)`` = ``(258,)``.  Undetected segments are zeros.
        """
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected a BGR frame of shape (H, W, 3), got {frame_bgr.shape}"
            )

        # MediaPipe requires RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Mark as non-writeable to pass by reference and avoid a copy.
        frame_rgb.flags.writeable = False
        results = self._holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        self._last_results = results  # cached for visualisation; not for production use

        vector = np.zeros(FEATURE_DIM, dtype=np.float32)

        # --- Left hand ---
        left_detected = results.left_hand_landmarks is not None
        if left_detected:
            vector[LEFT_HAND_SLICE] = _hand_landmarks_to_array(
                results.left_hand_landmarks
            )

        # --- Right hand ---
        right_detected = results.right_hand_landmarks is not None
        if right_detected:
            vector[RIGHT_HAND_SLICE] = _hand_landmarks_to_array(
                results.right_hand_landmarks
            )

        # --- Pose ---
        pose_detected = results.pose_landmarks is not None
        if pose_detected:
            vector[POSE_SLICE] = _pose_landmarks_to_array(results.pose_landmarks)

        return FrameLandmarks(
            left_hand_detected=left_detected,
            right_hand_detected=right_detected,
            pose_detected=pose_detected,
            vector=vector,
        )

    @property
    def last_results(self) -> Optional[object]:
        """The raw MediaPipe Holistic result from the most recent ``extract()`` call.

        **Debug / visualisation use only.**  Do not use this in production code
        or pass it to the ML model.  The type is intentionally ``object`` to
        signal that callers should not depend on its internal structure.

        Returns ``None`` before the first call to ``extract()``.
        """
        return self._last_results

    def close(self) -> None:
        """Release MediaPipe Holistic resources."""
        self._holistic.close()
        logger.info("LandmarkExtractor: closed.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _hand_landmarks_to_array(hand_landmarks) -> np.ndarray:
    """Flatten 21 hand landmarks → float32 array of shape (63,).

    Values are (x, y, z) for each landmark, in MediaPipe index order
    (0 = wrist, 1 = thumb_cmc, …, 20 = pinky_tip).
    """
    data = np.empty(HAND_DIM, dtype=np.float32)
    for i, lm in enumerate(hand_landmarks.landmark):
        base = i * HAND_VALUES_PER_LANDMARK
        data[base]     = lm.x
        data[base + 1] = lm.y
        data[base + 2] = lm.z
    return data


def _pose_landmarks_to_array(pose_landmarks) -> np.ndarray:
    """Flatten 33 pose landmarks → float32 array of shape (132,).

    Values are (x, y, z, visibility) for each landmark, in MediaPipe index
    order (0 = nose, …, 32 = right_foot_index).
    """
    data = np.empty(POSE_DIM, dtype=np.float32)
    for i, lm in enumerate(pose_landmarks.landmark):
        base = i * POSE_VALUES_PER_LANDMARK
        data[base]     = lm.x
        data[base + 1] = lm.y
        data[base + 2] = lm.z
        data[base + 3] = lm.visibility
    return data
