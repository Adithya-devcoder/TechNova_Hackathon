# Error analysis: BiGRU-CTC on held-out signers

Source: `evaluation/test_predictions.csv` (1,000 synthetic continuous sentences from the 3 test signers,
who never appear in training). Overall WER 21.0%; 510 of 1000 sentences exactly right.

## Error types
| error | count | share of reference words |
|---|---|---|
| substitution (wrong sign) | 635 | 17.5% |
| deletion (sign missed) | 79 | 2.2% |
| insertion (extra sign) | 46 | 1.3% |

## Most confused sign pairs (reference → predicted)
| confusion | count |
|---|---|
| pizza → brother | 21 |
| go → outside | 14 |
| girl → toy | 12 |
| like → hungry | 11 |
| girl → yesterday | 11 |
| sleepy → sleep | 11 |
| milk → car | 10 |
| book → grandma | 9 |
| child → boy | 9 |
| pizza → hello | 8 |
| night → store | 7 |
| minemy → please | 6 |

## Hardest signs (word error rate, signs seen at least 20 times)
| sign | times in test | error rate |
|---|---|---|
| pizza | 81 | 64% |
| girl | 67 | 55% |
| give | 74 | 51% |
| yes | 30 | 43% |
| go | 124 | 40% |
| book | 73 | 40% |
| sleep | 30 | 37% |
| sleepy | 72 | 36% |
| like | 80 | 34% |
| weus | 87 | 32% |

Easiest: boy (0%), find (2%), home (3%), store (3%), icecream (3%).

## WER by sentence length
| signs in sentence | sentences | WER | exactly right |
|---|---|---|---|
| 2 | 130 | 32.3% | 44% |
| 3 | 292 | 22.6% | 55% |
| 4 | 407 | 15.9% | 58% |
| 5 | 171 | 25.6% | 32% |

## Does confidence warn about mistakes?
| mean word confidence | sentences | exactly right | WER |
|---|---|---|---|
| 0.0–0.6 | 42 | 2% | 74.6% |
| 0.6–0.8 | 213 | 13% | 42.3% |
| 0.8–0.9 | 232 | 32% | 25.3% |
| 0.9–1.0 | 513 | 80% | 6.7% |

## Takeaways
- Most errors are **substitutions**.
- Some confusions are related signs (sleepy → sleep, child → boy); others (pizza → brother) have no
  obvious link and need the clips inspected. The model sees only hand and upper-body points, not the face.
- Sentence length is mixed up with sentence type: every 2-sign sentence is a random sequence (templates have
  3–5 signs), so its higher WER partly reflects the lack of familiar context, not length alone.
- Low-confidence sentences are wrong far more often, so flagging uncertain words in the app is justified.
- These are synthetic sentences built from isolated clips; live signing adds real co-articulation and a
  different camera, so live accuracy must be measured separately (app: Live evaluation mode).
