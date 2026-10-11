# SignBridge model results (measured, 2026-10-10 11:21)

- **Data:** Google ASL Signs (Kaggle `asl-signs`), 52 signs, 19590 isolated clips.
- **Split by signer:** train [2044, 4718, 16069, 18796, 22343, 25571, 26734, 27610, 28656, 29302, 32319, 36257, 53618, 55372, 61333], val [34503, 37055, 62590], **test [30680, 37779, 49445]**. Test signers are never seen in training.
- **Test set:** 1000 synthetic continuous sentences built only from the test signers' clips:
  460 demo-style template sentences and 540 random sign sequences.

| Method | WER all | WER template sentences | WER random sequences |
|---|---|---|---|
| Baseline: sliding window + LogisticRegression | 175.8% | 186.7% | 166.2% |
| Baseline + smoothing (K=3, tuned on val signers) | 96.2% | 95.9% | 96.6% |
| Ablation: BiGRU + CTC without signer normalisation | 24.6% | 12.8% | 35.0% |
| **BiGRU + CTC (ours)** | 21.0% | 11.5% | 29.4% |

- Sentences exactly right (model): 510/1000
- Mean word confidence: 0.94 on correct sentences, 0.79 on wrong ones
- predict() latency for one sentence of 42 frames: cpu 12.8 ms, cuda 4.2 ms
- Hardware: x86_64, GPU Tesla T4; 25 epochs

**Limitations:** sentences are synthesised from isolated clips, so real co-articulation is only approximated
by interpolated transitions. Template sentences also appear in training (with other signers), so their WER is
optimistic; the random-sequence WER is the purer recognition number. No live webcam test yet.
