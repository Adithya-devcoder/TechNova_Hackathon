"""Member 1: predict() contract, WER, and the signer split."""
import json
from pathlib import Path

import numpy as np
import pytest

from models import signbridge_model as sb

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def bundle():
    return sb.load_model()


def sentence_with_hands(n=60, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.full((n, 543, 3), np.nan, np.float32)
    raw[:, sb.POSE] = rng.uniform(0.3, 0.7, (1, 7, 3))
    raw[:, sb.RH] = rng.uniform(0.4, 0.6, (1, 21, 3)) + rng.normal(0, 0.01, (n, 21, 3))
    return raw


def test_vocabulary_meets_ps41(bundle):
    assert len(bundle['vocab']) >= 50
    assert len(set(bundle['vocab'])) == len(bundle['vocab'])


def test_predict_output_contract(bundle):
    out = sb.predict(sentence_with_hands(), bundle)
    assert set(out) == {'text', 'words', 'latency_ms'}
    assert out['text'] == ' '.join(w['word'] for w in out['words'])
    for w in out['words']:
        assert set(w) == {'word', 'confidence', 'uncertain', 'start', 'end', 'alternatives'}
        assert w['word'] in bundle['vocab']
        assert 0.0 <= w['confidence'] <= 1.0
        assert w['uncertain'] == (w['confidence'] < bundle['threshold'])


@pytest.mark.parametrize('raw', [np.zeros((0, 543, 3), np.float32), np.full((30, 543, 3), np.nan, np.float32)],
                         ids=['empty', 'no-person'])
def test_no_hands_gives_no_words(bundle, raw):
    assert sb.predict(raw, bundle)['text'] == ''


def test_word_positions_refer_to_input_frames(bundle):
    raw = sentence_with_hands(60)
    raw[:10, sb.RH] = np.nan    # hands appear only in frames 10-49
    raw[50:, sb.RH] = np.nan
    for w in sb.predict(raw, bundle)['words']:
        assert 10 <= w['start'] < w['end'] <= 50


@pytest.mark.parametrize('ref,hyp,expected', [
    ('a b c', 'a b c', 0.0),
    ('a b c', 'a x c', 1 / 3),      # substitution
    ('a b c', 'a c', 1 / 3),        # deletion
    ('a b c', 'a b c d e', 2 / 3),  # insertions
    ('a b', '', 1.0),
])
def test_wer(ref, hyp, expected):
    assert sb.wer([ref], [hyp]) == pytest.approx(expected)


def test_split_has_no_signer_leakage():
    split = json.loads((ROOT / 'data' / 'splits' / 'split.json').read_text())
    train, val, test = (set(split[k]) for k in ('train', 'val', 'test'))
    assert not (train & val or train & test or val & test)
    assert len(test) >= 3


def test_demo_sentences_use_only_vocabulary(bundle):
    sentences = json.loads((ROOT / 'data' / 'demo_sentences.json').read_text())
    assert len(sentences) >= 10
    for s in sentences:
        assert set(s.split()) <= set(bundle['vocab']), s
