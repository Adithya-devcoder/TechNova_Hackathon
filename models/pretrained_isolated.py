"""Pretrained isolated-sign recogniser: the 1st-place model of Kaggle's ASL Signs competition (Hoyeol Sohn).

It classifies ONE sign clip into one of 250 ASL signs. We combine it with our BiGRU-CTC model:
our model finds where each word is in a sentence, and this model decides which sign each piece is.

Needs two files from Kaggle in models/kaggle_1st/ (not committed; see README):
  model.tflite                       output of https://www.kaggle.com/code/hoyso48/1st-place-solution-inference
  sign_to_prediction_index_map.json  from https://www.kaggle.com/competitions/asl-signs/data

Note: this model was trained on every ASL Signs signer, including our test signers, so it must not be
scored on our Kaggle test split (leakage). Evaluate it only on live signing by new people.
"""
import json
import time
from pathlib import Path

import numpy as np

from models import signbridge_model as sb

FOLDER = Path(__file__).resolve().parent / 'kaggle_1st'
MODEL_FILE, LABEL_FILE = 'model.tflite', 'sign_to_prediction_index_map.json'


def available(folder=FOLDER):
    return (Path(folder) / MODEL_FILE).exists() and (Path(folder) / LABEL_FILE).exists()


class IsolatedSignModel:
    def __init__(self, folder=FOLDER):
        folder = Path(folder)
        try:   # LiteRT is the small, supported TFLite runtime; full TensorFlow also works
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
        self.interpreter = Interpreter(model_path=str(folder / MODEL_FILE))
        signatures = self.interpreter.get_signature_list()
        key = 'serving_default' if 'serving_default' in signatures else next(iter(signatures))
        self.runner = self.interpreter.get_signature_runner(key)
        self.input_name = signatures[key]['inputs'][0]
        self.output_name = signatures[key]['outputs'][0]
        index = json.loads((folder / LABEL_FILE).read_text())
        self.labels = [s for s, _ in sorted(index.items(), key=lambda kv: kv[1])]

    def probabilities(self, clip):
        """[T, 543, 3] landmarks of one sign (NaN = missing) -> probabilities over the 250 signs."""
        out = np.asarray(self.runner(**{self.input_name: np.asarray(clip, np.float32)})[self.output_name],
                         np.float64).reshape(-1)
        if out.min() < 0 or abs(out.sum() - 1) > 1e-3:   # logits -> probabilities
            out = np.exp(out - out.max())
            out /= out.sum()
        return out


# MediaPipe pose points that swap when the image is mirrored (left <-> right)
_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
               (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
_MIRROR = np.arange(543)
_MIRROR[468:489], _MIRROR[522:543] = np.arange(522, 543), np.arange(468, 489)
for _a, _b in _POSE_PAIRS:
    _MIRROR[489 + _a], _MIRROR[489 + _b] = 489 + _b, 489 + _a


def mirror(clip):
    """Left-right mirror of [T, 543, 3] landmarks: flip x and swap the hands and left/right pose points."""
    out = np.asarray(clip, np.float32)[:, _MIRROR].copy()
    out[..., 0] = 1.0 - out[..., 0]
    return out


def _hand_span(clip):
    hands = np.isfinite(clip[:, 468:489, 0]).any(1) | np.isfinite(clip[:, 522:543, 0]).any(1)
    if not hands.any():
        return None
    first, last = np.where(hands)[0][[0, -1]]
    return int(first), int(last) + 1


def classify_sign(clip, iso, vocab=None, threshold=0.6, try_mirror=True):
    """One sign clip -> word dict (as in predict()), or None if no hand is visible.

    try_mirror: also score the mirrored clip and keep the more confident reading, in case the camera's
    left/right convention differs from the training videos.
    """
    clip = np.asarray(clip, np.float32)
    span = _hand_span(clip)
    if span is None:
        return None
    a, b = span
    piece = clip[a:b]
    allowed = None if vocab is None else np.isin(iso.labels, list(vocab))

    def score(x):
        p = iso.probabilities(x)
        if allowed is not None:
            p = np.where(allowed, p, 0.0)
            p = p / p.sum() if p.sum() > 0 else p
        return p

    p = score(piece)
    if try_mirror:
        pm = score(mirror(piece))
        if pm.max() > p.max():
            p = pm
    top = np.argsort(-p)[:3]
    conf = float(p[top[0]])
    return {'word': iso.labels[top[0]], 'confidence': round(conf, 3), 'uncertain': conf < threshold,
            'start': a, 'end': b, 'alternatives': [(iso.labels[k], round(float(p[k]), 3)) for k in top]}


def _result(words, t0, min_confidence):
    kept = [w for w in words if w is not None and w['confidence'] >= min_confidence]
    return {'text': ' '.join(w['word'] for w in kept), 'words': kept,
            'latency_ms': round((time.perf_counter() - t0) * 1000, 1)}


def predict_signs(sign_clips, iso, vocab=None, threshold=0.6, min_confidence=0.0, offsets=None):
    """Sign-by-sign recognition: one clip per sign (the signer pauses between signs).

    Words below min_confidence are dropped, so gestures that are not signs produce no word.
    """
    t0 = time.perf_counter()
    words = []
    for i, clip in enumerate(sign_clips):
        w = classify_sign(clip, iso, vocab, threshold)
        if w is not None and offsets is not None:
            w['start'] += offsets[i]
            w['end'] += offsets[i]
        words.append(w)
    return _result(words, t0, min_confidence)


def _segments(n_frames, centres):
    """Split [0, n_frames) into one piece per word, cutting halfway between neighbouring word positions."""
    cuts = [0] + [(a + b) // 2 for a, b in zip(centres, centres[1:])] + [n_frames]
    return [(cuts[i], max(cuts[i + 1], cuts[i] + 1)) for i in range(len(centres))]


def predict_hybrid(landmarks, bundle, iso, vocab=None, min_sign_frames=8, min_confidence=0.0):
    """Same output as signbridge_model.predict(), with each word re-recognised by the pretrained model.

    vocab: optional list of allowed signs (e.g. our 52); None allows all 250.
    """
    t0 = time.perf_counter()
    base = sb.predict(landmarks, bundle)
    pts = np.asarray(landmarks, np.float32)
    span = _hand_span(pts)
    if span is None:
        return base
    first, last = span
    clip = pts[first:last]
    centres = [w['start'] - first for w in base['words']] or [len(clip) // 2]   # no word found: one sign
    words = []
    for (a, b), centre in zip(_segments(len(clip), centres), centres):
        if b - a < min_sign_frames:   # widen very short pieces around the word position
            a, b = max(0, centre - min_sign_frames // 2), min(len(clip), centre + min_sign_frames // 2 + 1)
        w = classify_sign(clip[a:b], iso, vocab, bundle['threshold'])
        if w is not None:
            w['start'] += first + a
            w['end'] += first + a
        words.append(w)
    return _result(words, t0, min_confidence)
