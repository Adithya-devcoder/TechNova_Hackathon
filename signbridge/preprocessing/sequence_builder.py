"""
signbridge.preprocessing.sequence_builder
==========================================
Converts per-frame ``FrameLandmarks`` into fixed-shape NumPy windows.

Overview
--------
``SequenceBuilder`` accumulates ``FrameLandmarks`` objects one at a time.
Each time enough frames have been buffered it emits a *window* — a 2-D
NumPy array of shape ``(window_size, FEATURE_DIM)`` — which is the direct
input to Member 1's temporal model (LSTM / GRU / Transformer).

Window strategy — **sliding window with configurable stride**
-------------------------------------------------------------
* A window is emitted every ``stride`` frames once the buffer holds at least
  ``window_size`` frames.
* ``stride=1`` gives a maximally overlapping stream (one window per frame
  after the first full window).
* ``stride=window_size`` gives non-overlapping, adjacent windows.
* Default: ``window_size=30``, ``stride=1``.

One window does **not** necessarily correspond to a complete sign or sentence.
Segmentation of signs from a window stream is the responsibility of the ML
model (Member 1).

Output array contract
---------------------
* **Shape**  : ``(window_size, FEATURE_DIM)``  →  ``(30, 258)`` by default.
* **Dtype**  : ``float32``
* **Layout** : rows are frames in chronological order (oldest first).
               columns follow ``FEATURE_DIM`` layout from ``landmark_extractor``.
* **Missing data**: zero-filled columns wherever landmarks were not detected
                    in the original frame.

Usage
-----
::

    builder = SequenceBuilder(window_size=30, stride=1)
    for frame_lm in stream_of_frame_landmarks:
        windows = builder.push(frame_lm)
        for window in windows:          # usually 0 or 1 windows per push
            model.predict(window)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List

import numpy as np

from signbridge.preprocessing.landmark_extractor import FEATURE_DIM, FrameLandmarks

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """Accumulates ``FrameLandmarks`` and emits sliding-window NumPy arrays.

    Parameters
    ----------
    window_size:
        Number of frames per window (T dimension of the output tensor).
        Default: 30 (≈ 1 second at 30 fps).
    stride:
        How many new frames must arrive before the next window is emitted.
        ``stride=1``            → maximum overlap (densest stream).
        ``stride=window_size``  → non-overlapping windows.
        Default: 1.

    Raises
    ------
    ValueError
        If ``window_size < 1`` or ``stride < 1`` or ``stride > window_size``.
    """

    def __init__(self, window_size: int = 30, stride: int = 1) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if stride > window_size:
            raise ValueError(
                f"stride ({stride}) must be <= window_size ({window_size})"
            )

        self._window_size = window_size
        self._stride = stride

        # Deque capped at window_size so old frames are automatically discarded.
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        # Counts frames since the last window was emitted.
        self._frames_since_emit: int = 0

        logger.debug(
            "SequenceBuilder: window_size=%d, stride=%d, FEATURE_DIM=%d",
            window_size,
            stride,
            FEATURE_DIM,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Number of frames per output window."""
        return self._window_size

    @property
    def stride(self) -> int:
        """Frames between successive window emissions."""
        return self._stride

    @property
    def buffered_frames(self) -> int:
        """How many frames are currently held in the buffer."""
        return len(self._buffer)

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of each emitted window: ``(window_size, FEATURE_DIM)``."""
        return (self._window_size, FEATURE_DIM)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, frame_lm: FrameLandmarks) -> List[np.ndarray]:
        """Add one frame's landmarks to the buffer.

        Parameters
        ----------
        frame_lm:
            ``FrameLandmarks`` from ``LandmarkExtractor.extract()``.

        Returns
        -------
        list[numpy.ndarray]
            A list of windows (usually empty or containing one array).
            Each array has shape ``(window_size, FEATURE_DIM)`` and dtype
            ``float32``.  The list is empty when the buffer is not yet full
            or the stride counter has not been reached.
        """
        if not isinstance(frame_lm, FrameLandmarks):
            raise TypeError(
                f"Expected FrameLandmarks, got {type(frame_lm).__name__}"
            )

        self._buffer.append(frame_lm.vector.copy())
        self._frames_since_emit += 1

        emitted: List[np.ndarray] = []

        if (
            len(self._buffer) == self._window_size
            and self._frames_since_emit >= self._stride
        ):
            window = np.stack(list(self._buffer), axis=0)  # (T, F)
            assert window.shape == (self._window_size, FEATURE_DIM), (
                f"BUG: window shape {window.shape} != expected "
                f"({self._window_size}, {FEATURE_DIM})"
            )
            assert window.dtype == np.float32, (
                f"BUG: window dtype {window.dtype} != float32"
            )
            emitted.append(window)
            self._frames_since_emit = 0
            logger.debug(
                "SequenceBuilder: emitted window shape=%s, "
                "buffer_size=%d, frames_since_emit reset",
                window.shape,
                len(self._buffer),
            )

        return emitted

    def reset(self) -> None:
        """Clear the internal buffer and reset the stride counter.

        Call this between independent signing clips to avoid contaminating
        a new clip with leftover frames from the previous one.
        """
        self._buffer.clear()
        self._frames_since_emit = 0
        logger.debug("SequenceBuilder: buffer reset.")

    def push_batch(self, frame_landmarks: List[FrameLandmarks]) -> List[np.ndarray]:
        """Convenience method: push multiple frames at once.

        Parameters
        ----------
        frame_landmarks:
            Ordered list of ``FrameLandmarks`` objects.

        Returns
        -------
        list[numpy.ndarray]
            All windows emitted during the batch push, in emission order.
        """
        all_windows: List[np.ndarray] = []
        for flm in frame_landmarks:
            all_windows.extend(self.push(flm))
        return all_windows
