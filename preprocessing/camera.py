"""One webcam for the whole app, handed from one Streamlit run to the next.

Streamlit starts a new run whenever a widget changes, while the previous run may still be reading the
camera. Opening the same webcam a second time made OpenCV's DirectShow backend throw
("Unknown C++ exception from OpenCV code"). Instead, the camera stays open across runs: each run
takes ownership with acquire(), the previous run's reads then return None and its loop exits, and the
device is only released when the camera is switched off.
"""
import os
import threading
import time

import cv2

SETTINGS = ((cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
            (cv2.CAP_PROP_BUFFERSIZE, 1))   # small frames and a 1-frame buffer: fresh frames, less lag


def open_device(index):
    """Open a webcam, or return None. Optional settings the driver refuses are skipped, never fatal."""
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(index)
    except cv2.error:
        return None
    if not cap.isOpened():
        cap.release()
        return None
    for prop, value in SETTINGS:
        try:
            cap.set(prop, value)
        except cv2.error:
            pass
    return cap


class SharedCamera:
    def __init__(self, opener=open_device, attempts=10, retry_s=0.2):
        self._opener, self._attempts, self._retry_s = opener, attempts, retry_s
        self._lock = threading.Lock()
        self._cap = self._index = self._owner = None

    def acquire(self, index):
        """Make the caller the camera's owner, opening the device if needed. Returns a token, or None."""
        token = object()
        with self._lock:
            self._owner = token
            if self._cap is not None and self._index == index:
                return token
            self._release_device()
            for _ in range(self._attempts):
                cap = self._opener(index)
                if cap is not None:
                    self._cap, self._index = cap, index
                    return token
                time.sleep(self._retry_s)
            self._owner = None
            return None

    def read(self, token):
        """(ok, frame) for the owner; (None, None) once another run has taken the camera over."""
        with self._lock:
            if self._owner is not token:
                return None, None
            if self._cap is None:
                return False, None
            try:
                return self._cap.read()
            except cv2.error:
                return False, None

    def release(self):
        """Close the device (camera switched off, or it stopped working)."""
        with self._lock:
            self._owner = None
            self._release_device()

    def _release_device(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except cv2.error:
                pass
        self._cap = self._index = None
