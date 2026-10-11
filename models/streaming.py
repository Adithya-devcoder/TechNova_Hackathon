"""Live word-by-word recognition: words are added to the sentence the moment each sign is recognised.

Every `step_s` seconds the last `window_s` seconds of landmarks are resampled to 30 fps and classified by
the pretrained isolated-sign model. A word is added when `agree` checks in a row name the same sign with
at least `min_conf` confidence; holding a sign does not add it twice. When the hands are down for
`end_s` seconds the sentence is finished.
"""
from collections import deque

import numpy as np

from models.pretrained_isolated import classify_sign
from preprocessing.landmarks import has_hands
from preprocessing.segmenter import resample


class WordStream:
    def __init__(self, iso, vocab=None, threshold=0.6, min_conf=0.3, window_s=1.0, step_s=0.3, agree=2,
                 end_s=1.5, fps=30.0):
        self.iso, self.vocab, self.threshold, self.min_conf = iso, vocab, threshold, min_conf
        self.window_ms, self.step_ms, self.agree, self.end_ms, self.fps = window_s * 1000, step_s * 1000, agree, end_s * 1000, fps
        self.frames = deque()
        self.words = []          # the sentence being built: word dicts as in predict()
        self.finished = None     # set to the word list when a sentence ends
        self._candidate, self._count = None, 0
        self._last_added = None  # word added most recently (blocks duplicates until it changes)
        self._last_check = -1e9
        self._last_hand = None

    @property
    def text(self):
        return ' '.join(w['word'] for w in self.words)

    def push(self, landmarks, t_ms):
        """Add one frame. Returns a newly added word dict, or None. Sets .finished when a sentence ends."""
        self.finished = None
        hand = has_hands(landmarks)
        self.frames.append((t_ms, landmarks))
        while self.frames and t_ms - self.frames[0][0] > self.window_ms:
            self.frames.popleft()
        if hand:
            self._last_hand = t_ms
        elif self._last_hand is not None and t_ms - self._last_hand >= self.end_ms:
            self._finish()
            return None
        elif self._last_hand is not None and t_ms - self._last_hand >= self.step_ms * 2:
            self._candidate, self._count, self._last_added = None, 0, None   # a pause: the same sign may follow
        if not hand or t_ms - self._last_check < self.step_ms:
            return None
        self._last_check = t_ms
        times = [t for t, _ in self.frames]
        clip = resample(np.stack([lm for _, lm in self.frames]), times, self.fps)
        word = classify_sign(clip, self.iso, self.vocab, self.threshold)
        if word is None or word['confidence'] < self.min_conf:
            self._candidate, self._count = None, 0
            return None
        if word['word'] == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = word['word'], 1
        if self._count >= self.agree and word['word'] != self._last_added:
            self._last_added = word['word']
            self.words.append(word)
            return word
        return None

    def _finish(self):
        if self.words:
            self.finished = self.words
        self.words = []
        self._candidate, self._count, self._last_added, self._last_hand = None, 0, None, None
        self.frames.clear()

    def flush(self):
        """End the current sentence now (e.g. the camera was switched off)."""
        self._finish()
        return self.finished
