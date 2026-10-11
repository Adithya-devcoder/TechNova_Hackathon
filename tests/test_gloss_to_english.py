"""Readable English from recognised sign glosses."""
import pytest

from models.gloss_to_english import to_english


@pytest.mark.parametrize('glosses,english', [
    ('minemy brother sick', 'My brother is sick.'),
    ('mom drink water now', 'Mom drinks water now.'),
    ('dad go store tomorrow', 'Dad will go to the store tomorrow.'),
    ('yesterday weus ride car home', 'We rode the car home yesterday.'),
    ('hello grandma thankyou', 'Hello grandma, thank you.'),
    ('boy have toy car', 'The boy has a toy car.'),
    ('hesheit sad cry', 'They are sad and cry.'),
    ('minemy dad give apple', 'My dad gives an apple.'),
    ('mom', 'Mom.'),
    ('', ''),
])
def test_to_english(glosses, english):
    assert to_english(glosses) == english


def test_every_demo_sentence_reads_as_a_sentence():
    import json
    from pathlib import Path
    for s in json.loads((Path(__file__).resolve().parents[1] / 'data' / 'demo_sentences.json').read_text()):
        out = to_english(s)
        assert out[0].isupper() and out.endswith('.') and '_' not in out
