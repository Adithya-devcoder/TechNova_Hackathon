"""Error analysis of the held-out-signer test predictions (evaluation/test_predictions.csv from Kaggle).

    python -m evaluation.error_analysis

Writes evaluation/error_analysis.md: error types, most-confused signs, hardest signs,
WER by sentence length, and whether low confidence predicts mistakes.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.signbridge_model import wer  # noqa: E402

PRED_PATH = ROOT / 'evaluation' / 'test_predictions.csv'
OUT_PATH = ROOT / 'evaluation' / 'error_analysis.md'


def align(ref, hyp):
    """Levenshtein alignment -> list of (op, ref_word, hyp_word); op in ok/sub/del/ins."""
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0], d[0, :] = range(len(r) + 1), range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    ops, i, j = [], len(r), len(h)
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(('ok' if r[i - 1] == h[j - 1] else 'sub', r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(('del', r[i - 1], None))
            i -= 1
        else:
            ops.append(('ins', None, h[j - 1]))
            j -= 1
    return ops[::-1]


def table(rows, header):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


def main():
    df = pd.read_csv(PRED_PATH, keep_default_na=False)
    ops = Counter()
    confusions, ref_count, ref_errors = Counter(), Counter(), Counter()
    for ref, hyp in zip(df.reference, df.model):
        for op, r, h in align(ref, hyp):
            ops[op] += 1
            if r is not None:
                ref_count[r] += 1
                ref_errors[r] += op != 'ok'
            if op == 'sub':
                confusions[(r, h)] += 1
    n_ref = sum(ref_count.values())

    df['n_words'] = df.reference.str.split().str.len()
    df['correct'] = df.reference == df.model
    by_len = [(n, len(g), f"{wer(g.reference, g.model):.1%}", f"{g.correct.mean():.0%}")
              for n, g in df.groupby('n_words')]
    hardest = sorted(((w, ref_count[w], f'{ref_errors[w] / ref_count[w]:.0%}') for w in ref_count if ref_count[w] >= 20),
                     key=lambda r: -float(r[2][:-1]))[:10]
    easiest = sorted(((w, ref_count[w], f'{ref_errors[w] / ref_count[w]:.0%}') for w in ref_count if ref_count[w] >= 20),
                     key=lambda r: float(r[2][:-1]))[:5]

    # Does low sentence confidence flag wrong sentences?
    conf = df.mean_confidence.astype(float)
    rows_conf = []
    for lo, hi in [(0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        g = df[(conf >= lo) & (conf < hi)]
        if len(g):
            rows_conf.append((f'{lo:.1f}–{min(hi, 1):.1f}', len(g), f'{g.correct.mean():.0%}', f'{wer(g.reference, g.model):.1%}'))

    report = f"""# Error analysis: BiGRU-CTC on held-out signers

Source: `evaluation/test_predictions.csv` (1,000 synthetic continuous sentences from the 3 test signers,
who never appear in training). Overall WER {wer(df.reference, df.model):.1%}; {df.correct.sum()} of {len(df)} sentences exactly right.

## Error types
{table([('substitution (wrong sign)', ops['sub'], f"{ops['sub'] / n_ref:.1%}"),
        ('deletion (sign missed)', ops['del'], f"{ops['del'] / n_ref:.1%}"),
        ('insertion (extra sign)', ops['ins'], f"{ops['ins'] / n_ref:.1%}")], ['error', 'count', 'share of reference words'])}

## Most confused sign pairs (reference → predicted)
{table([(f'{r} → {h}', c) for (r, h), c in confusions.most_common(12)], ['confusion', 'count'])}

## Hardest signs (word error rate, signs seen at least 20 times)
{table(hardest, ['sign', 'times in test', 'error rate'])}

Easiest: {', '.join(f'{w} ({e})' for w, _, e in easiest)}.

## WER by sentence length
{table(by_len, ['signs in sentence', 'sentences', 'WER', 'exactly right'])}

## Does confidence warn about mistakes?
{table(rows_conf, ['mean word confidence', 'sentences', 'exactly right', 'WER'])}

## Takeaways
- Most errors are **{max(('substitutions', ops['sub']), ('deletions', ops['del']), ('insertions', ops['ins']), key=lambda x: x[1])[0]}**.
- Some confusions are related signs (sleepy → sleep, child → boy); others (pizza → brother) have no
  obvious link and need the clips inspected. The model sees only hand and upper-body points, not the face.
- Sentence length is mixed up with sentence type: every 2-sign sentence is a random sequence (templates have
  3–5 signs), so its higher WER partly reflects the lack of familiar context, not length alone.
- Low-confidence sentences are wrong far more often, so flagging uncertain words in the app is justified.
- These are synthetic sentences built from isolated clips; live signing adds real co-articulation and a
  different camera, so live accuracy must be measured separately (app: Live evaluation mode).
"""
    OUT_PATH.write_text(report, encoding='utf-8')
    print(f'wrote {OUT_PATH.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
