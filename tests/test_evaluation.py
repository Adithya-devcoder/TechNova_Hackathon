"""Member 4: live results log and summary."""
from evaluation.live_log import append_result, load_results, summarize


def fake_result(text, conf=0.9):
    words = [{'word': w, 'confidence': conf, 'uncertain': conf < 0.6, 'start': 0, 'end': 1, 'alternatives': []}
             for w in text.split()]
    return {'text': text, 'words': words, 'latency_ms': 20.0}


def test_log_and_live_wer(tmp_path):
    path = tmp_path / 'live.csv'
    append_result('mom drink water', fake_result('mom drink water'), 60, path)
    append_result('girl like pizza', fake_result('girl pizza', 0.5), 50, path)
    append_result('hello mom', fake_result(''), 40, path)          # nothing recognised
    df = load_results(path)
    assert len(df) == 3 and df.loc[1, 'uncertain_words'] == 'girl pizza'
    s = summarize(df)
    assert s['attempts'] == 3 and s['exact_sentences'] == 1
    assert abs(s['wer'] - 3 / 8) < 1e-9                            # 1 deletion + 2 deletions over 8 words


def test_empty_log(tmp_path):
    assert summarize(load_results(tmp_path / 'missing.csv')) == {'attempts': 0}
