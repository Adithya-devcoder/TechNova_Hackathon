"""The webcam is opened once and handed between Streamlit runs instead of being opened twice."""
import cv2
import numpy as np

from preprocessing.camera import SharedCamera, open_device


class FakeCapture:
    opened = 0

    def __init__(self, *args, refuse_settings=False):
        FakeCapture.opened += 1
        self.refuse_settings, self.released = refuse_settings, False

    def isOpened(self):
        return True

    def set(self, *args):
        if self.refuse_settings:
            raise cv2.error('Unknown C++ exception from OpenCV code')
        return True

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.released = True


def test_new_run_takes_over_without_reopening():
    FakeCapture.opened = 0
    cam = SharedCamera(opener=lambda i: FakeCapture())
    first = cam.acquire(0)
    assert cam.read(first)[0] is True
    second = cam.acquire(0)                      # a rerun starts while the old loop is still running
    assert cam.read(first) == (None, None)       # the old loop is told to stop
    assert cam.read(second)[0] is True
    assert FakeCapture.opened == 1               # the device was opened only once


def test_switching_off_releases_and_changing_index_reopens():
    caps = []
    cam = SharedCamera(opener=lambda i: caps.append(FakeCapture()) or caps[-1])
    t = cam.acquire(0)
    cam.acquire(1)
    assert caps[0].released and len(caps) == 2
    cam.release()
    assert caps[1].released and cam.read(t) == (None, None)


def test_refused_settings_do_not_crash(monkeypatch):
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a: FakeCapture(refuse_settings=True))
    assert open_device(0) is not None


def test_no_camera_gives_none():
    cam = SharedCamera(opener=lambda i: None, attempts=2, retry_s=0)
    assert cam.acquire(0) is None
