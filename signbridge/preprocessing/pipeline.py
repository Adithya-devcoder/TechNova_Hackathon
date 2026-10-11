"""
signbridge.preprocessing.pipeline
===================================
Orchestrates webcam capture → landmark extraction → sequence building.

``CVPipeline`` ties the three modules together into a single generator that
yields ready-to-use NumPy windows for the ML model.

Usage
-----
::

    pipeline = CVPipeline(camera_index=0, window_size=30, stride=1)
    with pipeline:
        for window in pipeline.stream():
            # window: np.ndarray, shape (30, 258), dtype float32
            # Pass to model.predict(window) — do NOT save raw frames.
            pass

Design notes
------------
- No webcam footage is written to disk.
- ``CVPipeline`` does not perform sign recognition.  It only prepares the
  feature tensor that the ML model (Member 1) will consume.
- The pipeline can also be driven frame-by-frame via ``process_frame()`` for
  use in a Streamlit loop (Member 3).
"""

from __future__ import annotations

import logging
from typing import Generator, List, Optional

import numpy as np

from signbridge.preprocessing.capture import WebcamCapture, WebcamCaptureError
from signbridge.preprocessing.landmark_extractor import (
    FrameLandmarks,
    LandmarkExtractor,
)
from signbridge.preprocessing.sequence_builder import SequenceBuilder

logger = logging.getLogger(__name__)


class CVPipeline:
    """End-to-end computer-vision pipeline: webcam → landmarks → windows.

    Parameters
    ----------
    camera_index:
        Which webcam to open (0 = system default).
    window_size:
        Frames per output window.  Default: 30.
    stride:
        New frames required between window emissions.  Default: 1.
    static_image_mode:
        Set ``True`` when processing pre-recorded frames; ``False`` (default)
        for live webcam (enables MediaPipe tracking to reduce latency).
    min_detection_confidence:
        MediaPipe detection threshold (0.0–1.0).
    min_tracking_confidence:
        MediaPipe tracking threshold (0.0–1.0).
    cam_width:
        Requested webcam capture width in pixels.
    cam_height:
        Requested webcam capture height in pixels.
    cam_fps:
        Requested webcam frame-rate.
    """

    def __init__(
        self,
        camera_index: int = 0,
        window_size: int = 30,
        stride: int = 1,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        cam_width: int = 640,
        cam_height: int = 480,
        cam_fps: int = 30,
    ) -> None:
        self._cam = WebcamCapture(
            camera_index=camera_index,
            width=cam_width,
            height=cam_height,
            fps=cam_fps,
        )
        self._extractor = LandmarkExtractor(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._builder = SequenceBuilder(window_size=window_size, stride=stride)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CVPipeline":
        self._cam.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release webcam and MediaPipe resources."""
        self._cam.release()
        self._extractor.close()
        logger.info("CVPipeline: closed.")

    def process_frame(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """Process a single pre-supplied BGR frame through the full pipeline.

        Use this method when you already have a frame (e.g. from a Streamlit
        ``st.camera_input`` widget) instead of the internal webcam.

        Parameters
        ----------
        frame_bgr:
            BGR uint8 array of shape ``(H, W, 3)``.

        Returns
        -------
        list[numpy.ndarray]
            Zero or more windows, each of shape ``(window_size, FEATURE_DIM)``,
            dtype ``float32``.
        """
        frame_lm: FrameLandmarks = self._extractor.extract(frame_bgr)
        return self._builder.push(frame_lm)

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield landmark-sequence windows from the live webcam.

        The webcam must be open (use as a context manager, or call
        ``self._cam.open()`` first).

        Yields
        ------
        numpy.ndarray
            Shape ``(window_size, FEATURE_DIM)``, dtype ``float32``.
            One window per ``stride`` frames once the buffer is full.

        Notes
        -----
        - Frames are **never saved** to disk.
        - Break out of the ``for`` loop to stop capture.
        """
        for frame_bgr in self._cam.frames():
            windows = self.process_frame(frame_bgr)
            yield from windows

    def reset(self) -> None:
        """Reset the sequence builder buffer (call between signing clips)."""
        self._builder.reset()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of each window tensor: ``(window_size, FEATURE_DIM)``."""
        return self._builder.output_shape
