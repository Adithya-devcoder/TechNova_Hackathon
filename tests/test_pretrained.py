"""Hybrid recogniser: our CTC model finds the words, the Kaggle 1st-place model names each one."""
import numpy as np
import pytest

from models import pretrained_isolated as pi
from models import signbridge_model as sb


class FakeIsolated:
    """Stands in for the TFLite model: always most confident about 'zebra', then 'water'."""
    labels = ['water', 'zebra', 'mom']

    def __init__(self):
        self.calls = []

    def probabilities(self, clip):
        self.calls.append(len(clip))
        return np.array([0.3, 0.6, 0.1])


@pytest.fixture(scope='module')
def bundle():
    return sb.load_model()


def clip_with_hands(n=60, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.full((n, 543, 3), np.nan, np.float32)
    raw[:, sb.POSE] = rng.uniform(0.3, 0.7, (1, 7, 3))
    raw[5:55, sb.RH] = rng.uniform(0.4, 0.6, (1, 21, 3)) + rng.normal(0, 0.01, (50, 21, 3))
    return raw


def test_segments_cover_clip_without_gaps():
    segs = pi._segments(100, [10, 50, 90])
    assert segs[0][0] == 0 and segs[-1][1] == 100
    assert all(a[1] == b[0] for a, b in zip(segs, segs[1:]))


def test_hybrid_output_contract(bundle):
    iso = FakeIsolated()
    out = pi.predict_hybrid(clip_with_hands(), bundle, iso)
    assert set(out) == {'text', 'words', 'latency_ms'} and out['words']
    assert all(w['word'] == 'zebra' and w['confidence'] == 0.6 for w in out['words'])
    assert all(5 <= w['start'] < w['end'] <= 55 for w in out['words'])   # positions in the input clip
    assert all(n >= 8 for n in iso.calls)                                 # no tiny pieces sent to the model


def test_hybrid_vocab_restriction(bundle):
    out = pi.predict_hybrid(clip_with_hands(), bundle, FakeIsolated(), vocab=['water', 'mom'])
    assert {w['word'] for w in out['words']} == {'water'}
    assert out['words'][0]['confidence'] == pytest.approx(0.75)           # 0.3 / (0.3 + 0.1)


def test_hybrid_no_hands_returns_nothing(bundle):
    assert pi.predict_hybrid(np.full((20, 543, 3), np.nan, np.float32), bundle, FakeIsolated())['text'] == ''


@pytest.mark.skipif(not pi.available(), reason='Kaggle 1st-place files not in models/kaggle_1st/')
def test_real_pretrained_model_runs(bundle):
    iso = pi.IsolatedSignModel()
    assert len(iso.labels) == 250
    p = iso.probabilities(clip_with_hands())
    assert p.shape == (250,) and abs(p.sum() - 1) < 1e-3
    out = pi.predict_hybrid(clip_with_hands(), bundle, iso)
    assert all(w['word'] in iso.labels for w in out['words'])
