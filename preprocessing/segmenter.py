"""Cut a continuous stream of landmark frames into signed sentences.

A sentence starts when a hand appears and ends once both hands have been out of view for `pause_s`
seconds. Each finished sentence is resampled to a constant frame rate, because the camera loop runs
slower than the ~30 fps of the training videos and the model expects that signing speed.
"""
import numpy as np

from preprocessing.landmarks import has_hands


def resample(frames, times_ms, fps=30.0):
    """Linearly resample frames [T, ...] taken at times_ms onto a uniform `fps` grid.

    Where one neighbour is missing (NaN), the nearer frame is used instead of interpolating.
    """
    frames = np.asarray(frames, np.float32)
    t = np.asarray(times_ms, np.float64)
    if len(frames) < 2 or not fps or t[-1] <= t[0]:
        return frames
    grid = np.arange(t[0], t[-1] + 1e-6, 1000.0 / fps)
    i = np.clip(np.searchsorted(t, grid, side='right') - 1, 0, len(t) - 2)
    w = ((grid - t[i]) / np.maximum(t[i + 1] - t[i], 1e-6)).clip(0, 1)
    shape = (-1,) + (1,) * (frames.ndim - 1)
    a, b, w = frames[i], frames[i + 1], w.reshape(shape)
    mixed = (1 - w) * a + w * b
    nearest = np.where(w < 0.5, a, b)
    return np.where(np.isnan(mixed), nearest, mixed).astype(np.float32)


class SentenceSegmenter:
    def __init__(self, pause_s=0.7, min_sign_s=0.3, max_sentence_s=20.0, target_fps=30.0):
        self.pause_ms = pause_s * 1000
        self.min_sign_ms = min_sign_s * 1000
        self.max_ms = max_sentence_s * 1000
        self.target_fps = target_fps
        self._reset()

    def _reset(self):
        self._frames, self._times = [], []
        self._start = self._last_hand = None

    @property
    def active(self):
        """True while a sentence is being collected."""
        return bool(self._frames)

    def push(self, landmarks, t_ms):
        """Add one frame. Returns the finished sentence [T, 543, 3] at target_fps, or None."""
        if has_hands(landmarks):
            if not self._frames:
                self._start = t_ms
            self._frames.append(landmarks)
            self._times.append(t_ms)
            self._last_hand = t_ms
            if t_ms - self._start >= self.max_ms:
                return self._emit()
        elif self._frames:
            self._frames.append(landmarks)
            self._times.append(t_ms)
            if t_ms - self._last_hand >= self.pause_ms:
                return self._emit()
        return None

    def flush(self):
        """End of stream: return whatever sentence is pending, or None."""
        return self._emit() if self._frames else None

    def _emit(self):
        frames, times = self._frames, self._times
        long_enough = self._last_hand - self._start >= self.min_sign_ms
        self._reset()
        if not long_enough:   # a hand flickered into view; not a sign
            return None
        return resample(np.stack(frames), times, self.target_fps)
