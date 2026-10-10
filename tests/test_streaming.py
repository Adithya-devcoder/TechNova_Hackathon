"""Sentence builder: words appear one after another as they are signed, without pauses."""
import numpy as np

from models.streaming import WordStream


class FakeByPosition:
    """Names the sign from where the right hand is for most of the window: x<0.35 water, x>0.65 mom, else zebra."""
    labels = ['water', 'zebra', 'mom', 'other']

    def probabilities(self, clip):
        hand = clip[:, 522:543, 0]
        if not np.isfinite(hand).any():   # no right hand: unsure, like a real model
            return np.array([0.25, 0.25, 0.25, 0.25])
        x = np.nanmedian(np.nanmean(hand, axis=1))   # the sign filling most of the window, like a real model
        if x > 0.95:   # a gesture that is not a sign
            return np.array([0.26, 0.25, 0.25, 0.24])
        k = 0 if x < 0.35 else 2 if x > 0.65 else 1
        p = np.full(4, 0.03)
        p[k] = 0.91
        return p


def frame(x=None):
    lm = np.full((543, 3), np.nan, np.float32)
    lm[489:522] = 0.5
    if x is not None:
        lm[522:543] = x
    return lm


def feed(stream, xs, fps=15, t0=0.0):
    """Feed one frame per value at `fps`; returns the words added, in order."""
    added = []
    for i, x in enumerate(xs):
        w = stream.push(frame(x), t0 + i * 1000 / fps)
        if w:
            added.append(w['word'])
    return added


def test_words_appear_in_order_without_pauses():
    s = WordStream(FakeByPosition())
    added = feed(s, [0.2] * 30 + [0.5] * 30 + [0.8] * 30)          # three 2 s signs back to back
    assert added == ['water', 'zebra', 'mom']
    assert s.text == 'water zebra mom'


def test_holding_a_sign_adds_it_once():
    s = WordStream(FakeByPosition())
    assert feed(s, [0.2] * 60) == ['water']


def test_non_sign_gesture_adds_nothing():
    s = WordStream(FakeByPosition(), min_conf=0.3)
    assert feed(s, [0.99] * 45) == []


def test_hands_down_finishes_the_sentence():
    s = WordStream(FakeByPosition(), end_s=1.5)
    feed(s, [0.2] * 30 + [0.8] * 30)
    finished = None
    for i in range(30):                                          # 2 s with no hands
        s.push(frame(None), 4000 + i * 1000 / 15)
        finished = finished or s.finished
    assert [w['word'] for w in finished] == ['water', 'mom']
    assert s.words == []


def test_same_sign_twice_after_a_short_pause():
    s = WordStream(FakeByPosition(), end_s=1.5)
    added = feed(s, [0.2] * 30 + [None] * 10 + [0.2] * 30)       # ~0.7 s pause, then the same sign again
    assert added == ['water', 'water']
