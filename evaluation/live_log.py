"""Live demo evaluation: log each signed test sentence with what the model recognised, and score WER.

The app's "Live evaluation" mode appends a row per sentence. Summarise from the repo root with:
    python -m evaluation.live_log
"""
import csv
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.signbridge_model import wer  # noqa: E402  (same WER as the Kaggle test set)

LOG_PATH = ROOT / 'evaluation' / 'live_results.csv'
COLUMNS = ['time', 'reference', 'hypothesis', 'mean_confidence', 'uncertain_words', 'latency_ms', 'frames']


def append_result(reference, result, frames, path=LOG_PATH):
    """Append one live attempt. `result` is the dict returned by predict()."""
    words = result['words']
    row = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'reference': reference.strip().lower(),
        'hypothesis': result['text'],
        'mean_confidence': round(sum(w['confidence'] for w in words) / len(words), 3) if words else 0.0,
        'uncertain_words': ' '.join(w['word'] for w in words if w['uncertain']),
        'latency_ms': result['latency_ms'],
        'frames': frames,
    }
    path = Path(path)
    is_new = not path.exists()
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row


def load_results(path=LOG_PATH):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, keep_default_na=False)


def summarize(df):
    """Live WER and related numbers for a results table."""
    if df.empty:
        return {'attempts': 0}
    refs = df['reference'].astype(str).tolist()
    hyps = df['hypothesis'].astype(str).tolist()
    return {
        'attempts': len(df),
        'distinct_sentences': len(set(refs)),
        'wer': wer(refs, hyps),
        'exact_sentences': sum(r == h for r, h in zip(refs, hyps)),
        'median_latency_ms': float(pd.to_numeric(df['latency_ms']).median()),
    }


if __name__ == '__main__':
    results = load_results(sys.argv[1] if len(sys.argv) > 1 else LOG_PATH)
    if results.empty:
        print(f'no live results yet ({LOG_PATH})')
        sys.exit(0)
    s = summarize(results)
    print(results[['reference', 'hypothesis', 'mean_confidence', 'latency_ms']].to_string(index=False))
    print(f"\nlive WER {s['wer']:.1%} over {s['attempts']} attempts ({s['distinct_sentences']} distinct sentences), "
          f"{s['exact_sentences']} exactly right, median latency {s['median_latency_ms']:.0f} ms")
