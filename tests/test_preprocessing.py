"""Member 2: landmark layout, sentence segmentation and resampling."""
from types import SimpleNamespace

import numpy as np
import pytest

from models import signbridge_model as sb
from preprocessing.landmarks import N_ROWS, has_hands, result_to_array
from preprocessing.segmenter import SentenceSegmenter, resample


def frame(hand=True):
    lm = np.full((N_ROWS, 3), np.nan, np.float32)
    lm[489:522] = 0.5
    if hand:
        lm[522:543] = 0.4
    return lm


def points(n, value):
    return [SimpleNamespace(x=value, y=value + 0.01, z=0.0) for _ in range(n)]


def test_result_layout_matches_training_rows():
    # The Tasks face mesh has 478 points; only the first 468 belong in the training layout
    res = SimpleNamespace(face_landmarks=points(478, 0.1), left_hand_landmarks=[],
                          pose_landmarks=points(33, 0.3), right_hand_landmarks=points(21, 0.4))
    out = result_to_array(res)
    assert out.shape == (543, 3)
    assert np.allclose(out[:468, 0], 0.1)
    assert np.isnan(out[468:489]).all()            # left hand not detected
    assert np.allclose(out[489:522, 0], 0.3)
    assert np.allclose(out[522:543, 0], 0.4)
    assert out[sb.RH].shape == (21, 3)             # the model reads exactly these rows
    assert has_hands(out) and not has_hands(result_to_array(SimpleNamespace()))


def run(segmenter, hand_pattern, fps=15):
    """Feed frames at `fps`; return the sentences emitted (including a final flush)."""
    out = []
    for i, hand in enumerate(hand_pattern):
        s = segmenter.push(frame(hand), i * 1000 / fps)
        if s is not None:
            out.append(s)
    tail = segmenter.flush()
    return out + ([tail] if tail is not None else [])


def test_sentence_ends_after_pause():
    seg = SentenceSegmenter(pause_s=0.7, target_fps=30)
    sentences = run(seg, [False] * 5 + [True] * 30 + [False] * 15, fps=15)   # 2 s signing, then 1 s down
    assert len(sentences) == 1
    assert not seg.active


def test_two_sentences_split_at_pause():
    pattern = [True] * 15 + [False] * 15 + [True] * 15 + [False] * 15
    assert len(run(SentenceSegmenter(pause_s=0.7), pattern, fps=15)) == 2


def test_short_hand_flicker_is_ignored():
    assert run(SentenceSegmenter(pause_s=0.7, min_sign_s=0.3), [False] * 5 + [True] * 2 + [False] * 20) == []


def test_sentence_resampled_to_model_rate():
    sentence = run(SentenceSegmenter(pause_s=0.7, target_fps=30), [True] * 30 + [False] * 15, fps=15)[0]
    # frames span from the first hand to the end of the pause: about 2.7 s, i.e. ~81 frames at 30 fps
    assert 80 <= len(sentence) <= 90


def test_resample_interpolates_and_keeps_missing_parts():
    frames = np.array([[0.0, np.nan], [1.0, np.nan], [2.0, 5.0]], np.float32)
    out = resample(frames, [0, 100, 200], fps=20)          # 0, 50, 100, 150, 200 ms
    assert np.allclose(out[:, 0], [0, 0.5, 1, 1.5, 2])
    assert np.isnan(out[1, 1])                             # between two missing values stays missing
    assert out[4, 1] == pytest.approx(5.0)
