# SignBridge: 3-minute demo script

Record in one take if possible. Keep the terminal and the app visible; show real outputs, including mistakes.

## Setup before recording (not on camera)
- Repo on branch `test-abe`, `.venv` installed, app started with `.venv\Scripts\python -m streamlit run app/main.py`.
- Sidebar: **Mode = Live evaluation**, camera index set, threshold 0.6, landmarks overlay on.
- Signer sits facing the camera, shoulders and both hands in frame, plain background, good light.
- Signer has practised the demo sentences in `data/demo_sentences.json` (ASL signs from our 52-sign vocabulary).
- Delete old attempts (checkbox + Confirm delete) so the live WER starts from zero.

## 0:00–0:25 — Problem (slide or README on screen)
"Recognising a single sign is easy. Understanding a whole signed sentence is not: signs flow into each other, and every signer moves differently. SignBridge turns continuous signing from a laptop webcam into text, and tells you which words it is unsure about."

## 0:25–0:55 — How it works (architecture diagram in README)
"MediaPipe gives us 543 landmarks per frame. We keep the hands and upper body, normalise them to the signer's shoulders, and a bidirectional GRU trained with CTC loss reads the whole sentence. The dataset only has isolated signs, so we build continuous sentences ourselves: clips from the same signer, joined with interpolated transitions and random speed changes."

## 0:55–2:15 — Live demo (app, Live evaluation mode)
Sign at least 10 sentences. Suggested order (easy signs first):
1. `hello grandma thankyou`
2. `mom drink water now`
3. `boy have toy car`
4. `dad go store tomorrow`
5. `minemy brother sick`
6. `child happy find toy`
7. `grandpa read book night`
8. `weus go outside later`
9. `girl like pizza`  (pizza is our hardest sign: 64% error on the test set — say so)
10. `hesheit sad cry`
11. `minemy dad give apple`
12. `grandma hungry icecream`

For each: pick the reference in the dropdown, sign it, lower hands, read out what appeared.
- Point at a green word: "confident". Point at an amber word with "?": "the model is unsure; hover shows its other guesses."
- After the last sentence, show the **Live WER** line and the table. Say the number as it is.

## 2:15–2:40 — Honest evaluation (evaluation/results.md on screen)
"On 1,000 test sentences from three signers the model never saw, word error rate is 21%, against 96% for a tuned sliding-window baseline. Removing our signer normalisation makes it worse, 24.6%. Our live number today was [X]%, [higher/lower] than the test set, because live signing has real co-articulation and a different camera."

## 2:40–3:00 — Limitations and next steps
"It knows 52 ASL signs and nothing else: an unknown gesture is mapped to the closest sign, usually with low confidence. Next: train with other signs as negatives so it can say 'not a sign', measure live WER with more signers, and swap in an Indian Sign Language dataset through the same pipeline."

## If the camera fails
Switch to **Upload video** mode and run a pre-recorded clip of the same sentences. Say on camera that it is a recording.
