"""
signbridge.preprocessing.capture
=================================
Webcam frame capture using OpenCV.

This module intentionally does **not** write frames to disk.  Saving footage is
opt-in and must be done explicitly by calling code — never automatically.

Usage
-----
::

    with WebcamCapture(camera_index=0) as cam:
        for frame in cam:
            # frame is a numpy array, shape (H, W, 3), dtype uint8, BGR colour order.
            process(frame)
            if done:
                break

Thread safety
-------------
``WebcamCapture`` is **not** thread-safe.  Use one instance per thread.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamCaptureError(RuntimeError):
    """Raised when the webcam cannot be opened or a frame cannot be read."""


class WebcamCapture:
    """Context-managed OpenCV webcam reader.

    Parameters
    ----------
    camera_index:
        Integer index of the webcam device (0 = system default).
    width:
        Requested frame width in pixels.  The camera may silently ignore this
        and use its nearest supported resolution.
    height:
        Requested frame height in pixels.
    fps:
        Requested capture frame-rate.  Informational only on most drivers.

    Notes
    -----
    - Frames are BGR uint8 arrays, shape ``(height, width, 3)``.
    - The actual resolution may differ from the requested one; call
      ``cap.actual_width`` / ``cap.actual_height`` after opening.
    - Webcam footage is **never saved** by this class.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._camera_index = camera_index
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Properties (available after __enter__)
    # ------------------------------------------------------------------

    @property
    def actual_width(self) -> int:
        """Pixel width reported by the driver after opening."""
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore[union-attr]

    @property
    def actual_height(self) -> int:
        """Pixel height reported by the driver after opening."""
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore[union-attr]

    @property
    def actual_fps(self) -> float:
        """Frame-rate reported by the driver after opening."""
        self._require_open()
        return float(self._cap.get(cv2.CAP_PROP_FPS))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WebcamCapture":
        self._open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the webcam explicitly (alternative to using ``with`` statement)."""
        self._open()

    def release(self) -> None:
        """Release the webcam and free OS resources."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.info("WebcamCapture: released camera %d", self._camera_index)
        self._cap = None

    def read_frame(self) -> np.ndarray:
        """Read a single frame from the webcam.

        Returns
        -------
        numpy.ndarray
            BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

        Raises
        ------
        WebcamCaptureError
            If the webcam is not open or the frame could not be read.
        """
        self._require_open()
        ok, frame = self._cap.read()  # type: ignore[union-attr]
        if not ok or frame is None:
            raise WebcamCaptureError(
                f"Failed to read a frame from camera {self._camera_index}. "
                "Check that the webcam is connected and not in use by another application."
            )
        return frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield frames continuously until the webcam is released.

        Yields
        ------
        numpy.ndarray
            BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

        Notes
        -----
        The caller is responsible for breaking out of the loop.
        Calling ``release()`` from another thread will cause the next iteration
        to raise ``WebcamCaptureError``.
        """
        self._require_open()
        while True:
            try:
                yield self.read_frame()
            except WebcamCaptureError:
                logger.warning("WebcamCapture: frame read failed — stopping generator.")
                return

    # ------------------------------------------------------------------
    # Iterator protocol (delegates to frames())
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        return self.frames()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            logger.debug("WebcamCapture: already open on camera %d", self._camera_index)
            return

        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            raise WebcamCaptureError(
                f"Cannot open webcam at index {self._camera_index}. "
                "Ensure the device is connected and not used by another application."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        cap.set(cv2.CAP_PROP_FPS, self._requested_fps)

        self._cap = cap
        logger.info(
            "WebcamCapture: opened camera %d at %dx%d @ %.1f fps",
            self._camera_index,
            self.actual_width,
            self.actual_height,
            self.actual_fps,
        )

    def _require_open(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            raise WebcamCaptureError(
                "WebcamCapture is not open. "
                "Use it as a context manager or call open() first."
            )
