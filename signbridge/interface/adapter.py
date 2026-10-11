"""
signbridge.interface.adapter
============================
Converts one frame's ``FrameLandmarks`` (CV pipeline output) into the
``(543, 3)`` float32 array expected by Member 1's ``predict()`` function.

Background
----------
The CV pipeline stores 258 features per frame in a flat vector:

    [0:63]    left hand  — 21 landmarks × (x, y, z)
    [63:126]  right hand — 21 landmarks × (x, y, z)
    [126:258] pose       — 33 landmarks × (x, y, z, visibility)

Member 1's model expects a ``(T, 543, 3)`` array whose second axis follows
the full MediaPipe Holistic canonical slot layout:

    Slots 0–467    face landmarks (468)  — not captured by CV pipeline
    Slots 468–488  left hand       (21)
    Slots 489–521  pose            (33)
    Slots 522–542  right hand      (21)

Missing landmarks must be ``np.nan`` (not 0.0), because the model uses
``np.isfinite()`` and ``np.nanmedian()`` to detect absent hands and perform
signer normalisation.  Feeding zeros silently corrupts those steps.

Missing-landmark handling
-------------------------
``FrameLandmarks`` carries **boolean detection flags** (``left_hand_detected``,
``right_hand_detected``, ``pose_detected``) set by the extractor at the moment
MediaPipe runs.  These flags are the **authoritative source** of absence:

* If the flag is ``False`` → the corresponding 543-slot segment is NaN,
  regardless of the numeric values stored in the vector (which are 0.0 by
  construction but could theoretically be valid coordinates).
* If the flag is ``True``  → coordinates are read from the vector and placed
  into the correct 543 slots.

This design avoids the ambiguity of trying to infer absence from zeros, which
would fail for any landmark genuinely located at (0, 0, 0).

Limitation (documented)
-----------------------
The adapter accepts ``FrameLandmarks`` objects — it **cannot** convert a bare
``(258,)`` numpy array without detection flags, because zeros are ambiguous.
Callers that only have the flat vector and not the ``FrameLandmarks`` object
must treat all three segments as present and should document this assumption.
``frame_vector_to_holistic()`` is provided for this case and raises a warning.

Public API
----------
``frame_landmarks_to_holistic(frame_lm)``
    Convert one ``FrameLandmarks`` → ``(543, 3)`` float32, NaN for missing.

``sequence_to_holistic(frame_lm_list)``
    Convert a list of ``FrameLandmarks`` → ``(T, 543, 3)`` float32 array.

``frame_vector_to_holistic(vector, left_detected, right_detected, pose_detected)``
    Low-level conversion when detection flags are supplied separately.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Sequence

import numpy as np

from signbridge.preprocessing.landmark_extractor import (
    FEATURE_DIM,
    HAND_DIM,
    HAND_LANDMARK_COUNT,
    HAND_VALUES_PER_LANDMARK,
    LEFT_HAND_SLICE,
    POSE_DIM,
    POSE_LANDMARK_COUNT,
    POSE_VALUES_PER_LANDMARK,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
    FrameLandmarks,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model-side 543-slot constants (MediaPipe Holistic canonical order)
# ---------------------------------------------------------------------------

#: Total landmark slots in the model's input array.
HOLISTIC_SLOTS: int = 543

#: Number of spatial values per slot (x, y, z).
HOLISTIC_VALUES: int = 3

#: Face landmark slot range [0, 468).  Not captured by the CV pipeline.
FACE_SLOT_START: int = 0
FACE_SLOT_END: int = 468          # exclusive

#: Left-hand landmark slot range [468, 489).
LH_SLOT_START: int = 468
LH_SLOT_END: int = 489            # exclusive  (21 slots)

#: Pose landmark slot range [489, 522).
POSE_SLOT_START: int = 489
POSE_SLOT_END: int = 522          # exclusive  (33 slots)

#: Right-hand landmark slot range [522, 543).
RH_SLOT_START: int = 522
RH_SLOT_END: int = 543            # exclusive  (21 slots)

# Sanity checks (caught at import time, never silently wrong).
assert LH_SLOT_END - LH_SLOT_START == HAND_LANDMARK_COUNT,   "LH slot count mismatch"
assert RH_SLOT_END - RH_SLOT_START == HAND_LANDMARK_COUNT,   "RH slot count mismatch"
assert POSE_SLOT_END - POSE_SLOT_START == POSE_LANDMARK_COUNT, "Pose slot count mismatch"
assert HOLISTIC_SLOTS == FACE_SLOT_END + (LH_SLOT_END - LH_SLOT_START) + \
       (POSE_SLOT_END - POSE_SLOT_START) + (RH_SLOT_END - RH_SLOT_START)

# ---------------------------------------------------------------------------
# CV-side extraction helpers
# ---------------------------------------------------------------------------

def _extract_hand_xyz(vector: np.ndarray, hand_slice: slice) -> np.ndarray:
    """Return shape ``(21, 3)`` float32 from a 63-element hand segment."""
    return vector[hand_slice].reshape(HAND_LANDMARK_COUNT, HAND_VALUES_PER_LANDMARK)


def _extract_pose_xyz(vector: np.ndarray) -> np.ndarray:
    """Return shape ``(33, 3)`` float32 — x, y, z only, dropping visibility.

    The CV pipeline stores 4 values per pose landmark (x, y, z, visibility).
    The model's ``(543, 3)`` format has only 3 values per slot, so visibility
    is intentionally dropped here.
    """
    pose_raw = vector[POSE_SLICE].reshape(POSE_LANDMARK_COUNT, POSE_VALUES_PER_LANDMARK)
    return pose_raw[:, :3]   # drop column index 3 (visibility)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def frame_vector_to_holistic(
    vector: np.ndarray,
    left_detected: bool,
    right_detected: bool,
    pose_detected: bool,
) -> np.ndarray:
    """Convert a single 258-element feature vector to ``(543, 3)`` Holistic format.

    Parameters
    ----------
    vector:
        Float32 array of shape ``(258,)`` as produced by ``LandmarkExtractor``.
    left_detected:
        Whether the left hand was detected in this frame.
    right_detected:
        Whether the right hand was detected in this frame.
    pose_detected:
        Whether the body pose was detected in this frame.

    Returns
    -------
    numpy.ndarray
        Shape ``(543, 3)``, dtype ``float32``.
        Slots for undetected or unavailable landmarks are ``np.nan``.

    Notes
    -----
    Face slots (0–467) are always ``np.nan`` because the CV pipeline does not
    capture face landmarks.  The model's ``select_points()`` ignores face slots,
    so this has no effect on model predictions.
    """
    if vector.shape != (FEATURE_DIM,):
        raise ValueError(
            f"Expected vector of shape ({FEATURE_DIM},), got {vector.shape}"
        )
    if vector.dtype != np.float32:
        raise ValueError(
            f"Expected float32 vector, got {vector.dtype}"
        )

    out = np.full((HOLISTIC_SLOTS, HOLISTIC_VALUES), np.nan, dtype=np.float32)

    # Face slots (0–467): always NaN — not captured by CV pipeline.
    # (no action needed; array is pre-filled with NaN)

    # Left hand slots (468–488).
    if left_detected:
        out[LH_SLOT_START:LH_SLOT_END] = _extract_hand_xyz(vector, LEFT_HAND_SLICE)

    # Pose slots (489–521): x, y, z; visibility dropped.
    if pose_detected:
        out[POSE_SLOT_START:POSE_SLOT_END] = _extract_pose_xyz(vector)

    # Right hand slots (522–542).
    if right_detected:
        out[RH_SLOT_START:RH_SLOT_END] = _extract_hand_xyz(vector, RIGHT_HAND_SLICE)

    return out


def frame_landmarks_to_holistic(frame_lm: FrameLandmarks) -> np.ndarray:
    """Convert one ``FrameLandmarks`` to a ``(543, 3)`` float32 Holistic array.

    This is the primary conversion function.  It uses the detection flags
    carried by ``FrameLandmarks`` to determine which segments are absent,
    avoiding the ambiguity of inferring absence from zero-valued coordinates.

    Parameters
    ----------
    frame_lm:
        ``FrameLandmarks`` object returned by ``LandmarkExtractor.extract()``.

    Returns
    -------
    numpy.ndarray
        Shape ``(543, 3)``, dtype ``float32``.

        Slot layout (MediaPipe Holistic canonical order):
          - ``[0:468,   :]``  face       — always NaN (not captured)
          - ``[468:489, :]``  left hand  — NaN if ``left_hand_detected=False``
          - ``[489:522, :]``  pose       — NaN if ``pose_detected=False``
          - ``[522:543, :]``  right hand — NaN if ``right_hand_detected=False``
    """
    if not isinstance(frame_lm, FrameLandmarks):
        raise TypeError(
            f"Expected FrameLandmarks, got {type(frame_lm).__name__}"
        )
    return frame_vector_to_holistic(
        frame_lm.vector,
        left_detected=frame_lm.left_hand_detected,
        right_detected=frame_lm.right_hand_detected,
        pose_detected=frame_lm.pose_detected,
    )


def sequence_to_holistic(frame_lm_list: Sequence[FrameLandmarks]) -> np.ndarray:
    """Convert a sequence of ``FrameLandmarks`` to a ``(T, 543, 3)`` array.

    This is the array shape expected by Member 1's ``predict()`` function.

    Parameters
    ----------
    frame_lm_list:
        Ordered sequence of ``FrameLandmarks`` objects (oldest first).
        Must be non-empty.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, 543, 3)``, dtype ``float32``, where T = len(frame_lm_list).

    Raises
    ------
    ValueError
        If ``frame_lm_list`` is empty.

    Notes
    -----
    - This function does **not** perform sentence segmentation, resampling,
      or any temporal alignment.  Those are the responsibility of the
      integration layer (Member 3 / Member 1).
    - Calling ``predict()`` on this array is valid only when ``frame_lm_list``
      represents a complete signed sentence (per Member 1's contract).
    """
    if len(frame_lm_list) == 0:
        raise ValueError(
            "frame_lm_list must be non-empty. "
            "Collect at least one frame before calling sequence_to_holistic()."
        )

    frames = [frame_landmarks_to_holistic(fl) for fl in frame_lm_list]
    result = np.stack(frames, axis=0)   # (T, 543, 3)

    assert result.shape == (len(frame_lm_list), HOLISTIC_SLOTS, HOLISTIC_VALUES), (
        f"BUG: unexpected shape {result.shape}"
    )
    assert result.dtype == np.float32, f"BUG: unexpected dtype {result.dtype}"

    logger.debug(
        "sequence_to_holistic: T=%d, shape=%s, "
        "left_detected_frames=%d, right_detected_frames=%d, pose_detected_frames=%d",
        len(frame_lm_list),
        result.shape,
        sum(fl.left_hand_detected for fl in frame_lm_list),
        sum(fl.right_hand_detected for fl in frame_lm_list),
        sum(fl.pose_detected for fl in frame_lm_list),
    )
    return result
