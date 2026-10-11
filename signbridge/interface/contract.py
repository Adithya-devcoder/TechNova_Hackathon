"""
signbridge.interface.contract
==============================
**PROPOSED ML Input/Output Contract — awaiting Member 1 approval.**

STATUS = "PROPOSED"   # Change to "AGREED" once Member 1 signs off.

This file is the single authoritative reference for the data format that flows
between Member 2 (Computer Vision) and Member 1 (AI/ML Model).  It contains:

  1. Constants that define the exact tensor shape and dtype.
  2. A ``PredictionResult`` dataclass describing the model's expected output.
  3. A ``ModelInterface`` abstract base class that Member 1's model must implement.
  4. A ``MockModel`` that returns dummy predictions — **for development only**.
     Never report MockModel outputs as real results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION A — Input tensor specification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Shape : (T, F)   where T = WINDOW_SIZE = 30 (configurable)
                         F = FEATURE_DIM  = 258 (fixed for this version)
  Dtype : numpy.float32
  Name  : "landmark_sequence"

Column layout (F = 258):
  [0   : 63 ]   Left  hand — 21 landmarks × (x, y, z)
  [63  : 126]   Right hand — 21 landmarks × (x, y, z)
  [126 : 258]   Pose       — 33 landmarks × (x, y, z, visibility)

Coordinate convention:
  x   : normalised horizontal,  0.0 = left  edge, 1.0 = right edge
  y   : normalised vertical,    0.0 = top   edge, 1.0 = bottom edge
  z   : depth relative to wrist (hand) or mid-hip (pose); same scale as x
  vis : MediaPipe visibility score, 0.0–1.0 (pose only)

Missing landmarks:
  When a hand or the body is not detected in a frame, the corresponding
  columns for that frame are **zero-filled**.  The ML model must be trained
  on data that includes zero-filled frames.

Row order:
  Rows are in chronological order: row 0 is the oldest frame in the window,
  row T-1 is the most recent.

Face landmarks:
  Excluded in this version.  They can be appended as columns [258:] in a
  future version without breaking existing column indices.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION B — Output specification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The model returns a ``PredictionResult`` for each window it processes.
  See the ``PredictionResult`` dataclass below for field definitions.

  Confidence score:
    A single float in [0.0, 1.0] representing how confident the model is.
    Values below UNCERTAINTY_THRESHOLD should be flagged in the UI.
    Threshold is advisory; Member 1 sets the actual value based on calibration.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION C — Open questions for Member 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [ ] Is WINDOW_SIZE=30 (≈1 s at 30 fps) appropriate for the chosen model?
  [ ] Should the model receive a batch of windows or one at a time?
  [ ] Should zero-filled frames be masked (attention mask / padding mask)?
  [ ] What is the vocabulary list and index-to-label mapping?
  [ ] What confidence calibration method will be used?
  [ ] Will the model output per-window words, per-window CTC tokens, or a
      full sentence per clip?  (Affects how Member 3 assembles text.)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from signbridge.preprocessing.landmark_extractor import FEATURE_DIM

# ---------------------------------------------------------------------------
# Contract status — update once Member 1 approves
# ---------------------------------------------------------------------------

STATUS: str = "PROPOSED"  # → "AGREED" once sign-off is received

# ---------------------------------------------------------------------------
# Input-side constants
# ---------------------------------------------------------------------------

#: Default number of frames per window.  Member 1 may request a different value.
WINDOW_SIZE: int = 30

#: Feature dimension (columns).  Must match landmark_extractor.FEATURE_DIM.
assert FEATURE_DIM == 258, f"FEATURE_DIM mismatch: {FEATURE_DIM}"

#: Expected dtype of the input tensor.
INPUT_DTYPE: np.dtype = np.dtype("float32")

# ---------------------------------------------------------------------------
# Output-side constants
# ---------------------------------------------------------------------------

#: Windows with confidence below this threshold should be flagged in the UI.
#: Advisory only — Member 1 sets the real threshold after calibration.
UNCERTAINTY_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Output of the ML model for one landmark-sequence window.

    Attributes
    ----------
    predicted_sign:
        The sign label predicted for this window, e.g. ``"hello"``.
        ``None`` if the model is not confident enough to commit to a label
        (e.g. transition frame, blank in CTC).
    confidence:
        Confidence score in ``[0.0, 1.0]``.  Values below
        ``UNCERTAINTY_THRESHOLD`` should be highlighted in the UI.
    is_uncertain:
        ``True`` when ``confidence < UNCERTAINTY_THRESHOLD``.
        Set automatically in ``__post_init__``.
    raw_logits:
        Optional raw model output (before softmax / argmax).  Useful for
        debugging and calibration.  Not required for production.
    """

    predicted_sign: Optional[str]
    confidence: float
    is_uncertain: bool = field(init=False)
    raw_logits: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        self.is_uncertain = self.confidence < UNCERTAINTY_THRESHOLD


# ---------------------------------------------------------------------------
# Abstract interface — Member 1 must implement this
# ---------------------------------------------------------------------------


class ModelInterface(abc.ABC):
    """Abstract base class that Member 1's sign-recognition model must implement.

    Member 3 (Frontend) will call ``predict()`` on an instance of a subclass.

    Contract
    --------
    ``predict(landmark_sequence)`` takes a float32 array of shape
    ``(WINDOW_SIZE, FEATURE_DIM)`` and returns a ``PredictionResult``.

    The function must be **pure** with respect to the tensor: calling it twice
    with the same input must return equivalent results (determinism, modulo
    dropout during training).
    """

    @abc.abstractmethod
    def predict(self, landmark_sequence: np.ndarray) -> PredictionResult:
        """Predict the sign for one landmark-sequence window.

        Parameters
        ----------
        landmark_sequence:
            NumPy array, shape ``(T, F)`` where T = WINDOW_SIZE, F = FEATURE_DIM.
            dtype must be ``float32``.

        Returns
        -------
        PredictionResult
        """
        ...

    # ------------------------------------------------------------------
    # Input validation helper — Member 1 may call this inside predict()
    # ------------------------------------------------------------------

    @staticmethod
    def validate_input(landmark_sequence: np.ndarray) -> None:
        """Raise ``ValueError`` if the tensor does not match the contract.

        Parameters
        ----------
        landmark_sequence:
            Tensor to validate.
        """
        if landmark_sequence.ndim != 2:
            raise ValueError(
                f"Expected 2-D array (T, F), got {landmark_sequence.ndim}-D"
            )
        _, F = landmark_sequence.shape
        if F != FEATURE_DIM:
            raise ValueError(
                f"Expected F={FEATURE_DIM} features, got F={F}"
            )
        if landmark_sequence.dtype != INPUT_DTYPE:
            raise ValueError(
                f"Expected dtype float32, got {landmark_sequence.dtype}"
            )


# ---------------------------------------------------------------------------
# Mock model — DEVELOPMENT ONLY, never report as real results
# ---------------------------------------------------------------------------


class MockModel(ModelInterface):
    """Stub implementation for development and integration testing.

    Returns a fixed dummy label with a fixed confidence score.
    Label cycles through a short vocabulary so the UI can show variety.

    ⚠ MockModel outputs are **not** real sign-recognition results.
    Label them clearly in the UI and remove before the final demo.
    """

    _DUMMY_VOCAB: List[str] = [
        "hello", "thank_you", "please", "help", "yes", "no",
        "my_name_is", "how_are_you", "good", "water",
    ]

    def __init__(self) -> None:
        self._call_count: int = 0

    def predict(self, landmark_sequence: np.ndarray) -> PredictionResult:
        self.validate_input(landmark_sequence)
        label = self._DUMMY_VOCAB[self._call_count % len(self._DUMMY_VOCAB)]
        self._call_count += 1
        return PredictionResult(
            predicted_sign=label,
            confidence=0.42,   # intentionally below UNCERTAINTY_THRESHOLD
            raw_logits=None,
        )
