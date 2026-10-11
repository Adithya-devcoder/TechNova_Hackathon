"""
signbridge.preprocessing.sentence_buffer
==========================================
Buffers a complete signed sentence as a list of ``FrameLandmarks`` objects
and converts it to the ``(T, 543, 3)`` float32 array expected by Member 1's
``predict()`` function.

Why this module exists
----------------------
``SequenceBuilder`` stores only ``frame_lm.vector`` (a ``(258,)`` float32
array) and discards the boolean detection flags (``left_hand_detected``,
``right_hand_detected``, ``pose_detected``).  Once those flags are lost, the
adapter cannot distinguish a genuinely absent landmark (which must map to
``NaN``) from a landmark that happened to be located at the origin ``(0,0,0)``
(which must map to the real coordinates).

``SentenceBuffer`` solves this by accumulating **complete ``FrameLandmarks``
objects** — flags and vector together — and never converting to a raw NumPy
array until the caller explicitly calls ``flush()``.

Sentence boundary detection
----------------------------
No automatic boundary detection is implemented in this version.  The caller
is responsible for deciding when a sentence ends and calling ``flush()``.
For the first demo, a key press (e.g. ``SPACE``) is the recommended trigger.
A smarter heuristic (silence / hand stillness) can be added later without
changing this module's API.

Relationship to other modules
------------------------------
+--------------------+------------------------------------------+
| Module             | Role                                     |
+====================+==========================================+
| LandmarkExtractor  | Produces FrameLandmarks (flags + vector) |
+--------------------+------------------------------------------+
| SequenceBuilder    | Sliding-window (T, 258) for exploration  |
+--------------------+------------------------------------------+
| **SentenceBuffer** | Full-sentence (T, 543, 3) for predict()  |
+--------------------+------------------------------------------+
| adapter            | Converts FrameLandmarks → (543, 3) NaN   |
+--------------------+------------------------------------------+

Usage
-----
::

    buffer = SentenceBuffer()
    buffer.start()

    for frame_bgr in cam.frames():
        lm = extractor.extract(frame_bgr)
        buffer.append(lm)

        if key == ord(' '):          # sentence boundary trigger
            result = buffer.flush()
            if result is not None:
                holistic_array = result.holistic   # (T, 543, 3), NaN for missing
                frames_list    = result.frames     # list[FrameLandmarks], flags intact
                # pass holistic_array to predict()

    buffer.reset()    # safe to call at any time

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from signbridge.interface.adapter import (
    HOLISTIC_SLOTS,
    HOLISTIC_VALUES,
    sequence_to_holistic,
)
from signbridge.preprocessing.landmark_extractor import FrameLandmarks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type returned by flush()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SentenceResult:
    """Immutable result of flushing a complete sentence.

    Attributes
    ----------
    frames:
        Ordered list of ``FrameLandmarks`` objects (oldest first).
        Flags (``left_hand_detected``, etc.) are fully preserved.
    holistic:
        The same data converted to shape ``(T, 543, 3)`` float32 in
        MediaPipe Holistic canonical slot order, with ``np.nan`` for every
        landmark slot that was not detected in that frame.
        This is the array that should be passed to ``predict()``.

    Notes
    -----
    ``len(frames) == holistic.shape[0]``  is always true.
    """

    frames: List[FrameLandmarks]
    holistic: np.ndarray

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("SentenceResult.frames must not be empty.")
        if self.holistic.shape != (len(self.frames), HOLISTIC_SLOTS, HOLISTIC_VALUES):
            raise ValueError(
                f"SentenceResult.holistic shape {self.holistic.shape} does not match "
                f"({len(self.frames)}, {HOLISTIC_SLOTS}, {HOLISTIC_VALUES})"
            )
        if self.holistic.dtype != np.float32:
            raise ValueError(
                f"SentenceResult.holistic must be float32, got {self.holistic.dtype}"
            )


# ---------------------------------------------------------------------------
# SentenceBuffer
# ---------------------------------------------------------------------------

class SentenceBuffer:
    """Accumulates ``FrameLandmarks`` for one complete signed sentence.

    Parameters
    ----------
    max_frames:
        Hard upper limit on how many frames a single sentence may contain.
        If ``append()`` is called when the buffer is already full, the oldest
        frame is dropped (sliding behaviour) and a warning is logged.
        Default: 600 (20 seconds at 30 fps — a safety cap, not a target).
        Set to ``None`` to disable the cap (unbounded, not recommended for
        live use).
    min_frames:
        Minimum number of frames required for ``flush()`` to return a result.
        If the buffer holds fewer frames than this when ``flush()`` is called,
        ``flush()`` returns ``None`` and logs a warning.  Default: 1.

    State machine
    -------------
    ``IDLE``
        Initial state and state after ``reset()``.  ``append()`` is a no-op;
        ``flush()`` returns ``None``.

    ``RECORDING``
        Entered via ``start()``.  ``append()`` accumulates frames.
        ``flush()`` converts, resets, and returns a ``SentenceResult``.

    State transitions::

        IDLE  →  start()      →  RECORDING
        RECORDING  →  flush()  →  IDLE
        RECORDING  →  reset()  →  IDLE
        IDLE       →  reset()  →  IDLE   (safe no-op)
    """

    # States
    _IDLE = "IDLE"
    _RECORDING = "RECORDING"

    def __init__(
        self,
        max_frames: Optional[int] = 600,
        min_frames: int = 1,
    ) -> None:
        if max_frames is not None and max_frames < 1:
            raise ValueError(f"max_frames must be >= 1 or None, got {max_frames}")
        if min_frames < 1:
            raise ValueError(f"min_frames must be >= 1, got {min_frames}")
        if max_frames is not None and min_frames > max_frames:
            raise ValueError(
                f"min_frames ({min_frames}) must be <= max_frames ({max_frames})"
            )

        self._max_frames: Optional[int] = max_frames
        self._min_frames: int = min_frames
        self._frames: List[FrameLandmarks] = []
        self._state: str = self._IDLE

        logger.debug(
            "SentenceBuffer: created (max_frames=%s, min_frames=%d)",
            max_frames,
            min_frames,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state: ``'IDLE'`` or ``'RECORDING'``."""
        return self._state

    @property
    def is_recording(self) -> bool:
        """``True`` if the buffer is currently recording a sentence."""
        return self._state == self._RECORDING

    @property
    def frame_count(self) -> int:
        """Number of frames currently held in the buffer."""
        return len(self._frames)

    @property
    def max_frames(self) -> Optional[int]:
        """Hard cap on sentence length (frames), or ``None`` if unbounded."""
        return self._max_frames

    @property
    def min_frames(self) -> int:
        """Minimum frames required for ``flush()`` to return a result."""
        return self._min_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enter RECORDING state and clear any leftover frames.

        It is safe to call ``start()`` when already RECORDING; this resets
        the buffer and starts a fresh sentence (equivalent to ``reset()``
        followed by ``start()``).
        """
        self._frames = []
        self._state = self._RECORDING
        logger.debug("SentenceBuffer: started recording.")

    def append(self, frame_lm: FrameLandmarks) -> None:
        """Append one frame's landmarks to the current sentence.

        Parameters
        ----------
        frame_lm:
            ``FrameLandmarks`` from ``LandmarkExtractor.extract()``.
            The full object is stored — detection flags are preserved.

        Raises
        ------
        TypeError
            If ``frame_lm`` is not a ``FrameLandmarks`` instance.

        Notes
        -----
        If the buffer is ``IDLE``, the frame is silently ignored and a
        debug message is logged.  Call ``start()`` first.

        If the buffer is at ``max_frames`` capacity, the oldest frame is
        evicted (sliding-window behaviour) and a warning is logged.
        """
        if not isinstance(frame_lm, FrameLandmarks):
            raise TypeError(
                f"SentenceBuffer.append() expects FrameLandmarks, "
                f"got {type(frame_lm).__name__}"
            )

        if self._state == self._IDLE:
            logger.debug(
                "SentenceBuffer.append() called while IDLE — frame ignored. "
                "Call start() first."
            )
            return

        # Hard cap: drop oldest frame if at capacity.
        if self._max_frames is not None and len(self._frames) >= self._max_frames:
            self._frames.pop(0)
            logger.warning(
                "SentenceBuffer: max_frames (%d) reached; oldest frame evicted. "
                "Consider calling flush() sooner.",
                self._max_frames,
            )

        self._frames.append(frame_lm)

    def flush(self) -> Optional[SentenceResult]:
        """End the current sentence and convert it to a ``SentenceResult``.

        Converts all buffered ``FrameLandmarks`` objects to a
        ``(T, 543, 3)`` float32 array via ``sequence_to_holistic()`` and
        returns a ``SentenceResult``.  Resets the buffer to ``IDLE``
        regardless of outcome.

        Returns
        -------
        SentenceResult
            If the buffer held >= ``min_frames`` frames.
        None
            If the buffer was ``IDLE``, or held fewer than ``min_frames``
            frames.  A warning is logged in the latter case.

        Notes
        -----
        After ``flush()`` returns, the buffer is in ``IDLE`` state.
        Call ``start()`` to begin the next sentence.
        """
        if self._state == self._IDLE:
            logger.debug("SentenceBuffer.flush() called while IDLE — returning None.")
            self._frames = []
            return None

        n = len(self._frames)

        if n < self._min_frames:
            logger.warning(
                "SentenceBuffer.flush(): only %d frame(s) buffered "
                "(min_frames=%d) — returning None and discarding.",
                n,
                self._min_frames,
            )
            self._frames = []
            self._state = self._IDLE
            return None

        # Snapshot the frames before clearing state.
        frames_snapshot: List[FrameLandmarks] = list(self._frames)

        # Reset before conversion so a crash in sequence_to_holistic()
        # doesn't leave the buffer in an inconsistent state.
        self._frames = []
        self._state = self._IDLE

        holistic = sequence_to_holistic(frames_snapshot)   # (T, 543, 3)

        result = SentenceResult(frames=frames_snapshot, holistic=holistic)

        logger.info(
            "SentenceBuffer.flush(): sentence complete — "
            "T=%d frames, holistic shape=%s, "
            "left_hand_frames=%d, right_hand_frames=%d, pose_frames=%d",
            len(frames_snapshot),
            holistic.shape,
            sum(fl.left_hand_detected for fl in frames_snapshot),
            sum(fl.right_hand_detected for fl in frames_snapshot),
            sum(fl.pose_detected for fl in frames_snapshot),
        )
        return result

    def reset(self) -> None:
        """Discard all buffered frames and return to ``IDLE``.

        Safe to call at any time and from any state.  Does not raise.
        """
        n = len(self._frames)
        self._frames = []
        self._state = self._IDLE
        if n > 0:
            logger.debug("SentenceBuffer.reset(): discarded %d frame(s).", n)
        else:
            logger.debug("SentenceBuffer.reset(): buffer was already empty.")
