"""SignBridge core: landmark features, BiGRU-CTC model, CTC decoding and predict().

The training notebook and the live app both import this file, so the features match exactly.
Input to predict(): float32 array [T, 543, 3] in MediaPipe Holistic order
(face 0-467, left hand 468-488, pose 489-521, right hand 522-542), with NaN for missing points.
"""
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

LH = list(range(468, 489))
RH = list(range(522, 543))
POSE = [489 + i for i in (0, 11, 12, 13, 14, 15, 16)]   # nose, shoulders, elbows, wrists
POINTS = LH + RH + POSE                                  # 49 points
N_PTS = len(POINTS)
I_LH, I_RH = slice(0, 21), slice(21, 42)
I_LSH, I_RSH = 43, 44
# Swap the hands and the left/right pose points when mirroring
MIRROR_PERM = list(range(21, 42)) + list(range(0, 21)) + [42, 44, 43, 46, 45, 48, 47]
FEAT_DIM = N_PTS * 2 * 2 + 2   # positions + velocities + two hand-presence flags


def select_points(raw):
    """[T, 543, 3] Holistic landmarks -> [T, 49, 2] (x, y) float32, NaN where missing."""
    return np.asarray(raw, dtype=np.float32)[:, POINTS, :2]


def make_features(pts, normalize=True):
    """[T, 49, 2] points -> [T, FEAT_DIM] float32 features.

    normalize=True: mirror so the dominant hand is always in the right-hand slot, then centre
    on the shoulder midpoint and scale by shoulder width (signer normalisation).
    """
    p = np.array(pts, dtype=np.float32)
    if normalize:
        lh_frames = np.isfinite(p[:, I_LH, 0]).any(1).sum()
        rh_frames = np.isfinite(p[:, I_RH, 0]).any(1).sum()
        if lh_frames > rh_frames:
            p[..., 0] = -p[..., 0]
            p = p[:, MIRROR_PERM]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            sh = p[:, [I_LSH, I_RSH]]
            center = np.nanmedian(sh.mean(1), axis=0)
            width = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
            if not np.all(np.isfinite(center)):
                center = np.nanmean(p.reshape(-1, 2), axis=0) if np.isfinite(p).any() else np.zeros(2, np.float32)
        if not np.isfinite(width) or width < 1e-3:
            width = 0.25
        p = (p - center) / width
    lh = np.isfinite(p[:, I_LH, 0]).any(1, keepdims=True).astype(np.float32)
    rh = np.isfinite(p[:, I_RH, 0]).any(1, keepdims=True).astype(np.float32)
    vel = np.nan_to_num(np.diff(p, axis=0, prepend=p[:1]))
    pos = np.nan_to_num(p)
    T = len(p)
    return np.concatenate([pos.reshape(T, -1), vel.reshape(T, -1), lh, rh], axis=1).astype(np.float32)


class GRUCTC(nn.Module):
    """Linear projection -> residual Conv1D -> 2-layer BiGRU -> per-frame classes (0 = CTC blank)."""

    def __init__(self, in_dim, n_classes, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x, lengths):
        h = self.inp(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.head(out)


def greedy_decode(probs, vocab, threshold=0.6):
    """Greedy CTC decoding of per-frame probabilities [T, C].

    A word's confidence is the mean probability of that word over the frames that produced it.
    """
    best = probs.argmax(1)
    words, t, T = [], 0, len(best)
    while t < T:
        c, s = best[t], t
        while t < T and best[t] == c:
            t += 1
        if c != 0:
            seg = probs[s:t]
            conf = float(seg[:, c].mean())
            mean = seg.mean(0)
            mean[0] = 0
            top = np.argsort(-mean)[:3]
            words.append({
                'word': vocab[c - 1], 'confidence': round(conf, 3), 'uncertain': conf < threshold,
                'start': int(s), 'end': int(t),
                'alternatives': [(vocab[k - 1], round(float(mean[k]), 3)) for k in top if k != 0],
            })
    return words


def wer(refs, hyps):
    """Word error rate = (substitutions + deletions + insertions) / reference words."""
    errors = n_words = 0
    for ref, hyp in zip(refs, hyps):
        r, h = ref.split(), hyp.split()
        d = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(h) + 1):
                cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
                prev, d[j] = d[j], cur
        errors += d[len(h)]
        n_words += len(r)
    return errors / max(n_words, 1)


DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_main.pt')


def load_model(path=DEFAULT_MODEL, device='cpu'):
    if device == 'cpu':
        torch.set_num_threads(1)   # one sentence at a time: a single thread was 2-3x faster than 8 on a laptop
    ck = torch.load(path, map_location=device)
    model = GRUCTC(ck['in_dim'], len(ck['vocab']) + 1, **ck['model_cfg']).to(device).eval()
    model.load_state_dict(ck['state'])
    return {'model': model, 'vocab': ck['vocab'], 'normalize': ck['normalize'],
            'threshold': ck['threshold'], 'device': device}


_BUNDLE = None


@torch.no_grad()
def predict(landmarks, bundle=None):
    """[T, 543, 3] landmarks of one signed sentence -> {'text', 'words', 'latency_ms'}."""
    global _BUNDLE
    if bundle is None:
        _BUNDLE = _BUNDLE or load_model()
        bundle = _BUNDLE
    t0 = time.perf_counter()
    pts = select_points(landmarks) if len(landmarks) else np.zeros((0, N_PTS, 2), np.float32)
    hands = np.isfinite(pts[:, :42, 0]).any(1)
    if not hands.any():   # nobody signing: no hand in any frame
        return {'text': '', 'words': [], 'latency_ms': round((time.perf_counter() - t0) * 1000, 1)}
    first, last = np.where(hands)[0][[0, -1]]
    pts = pts[first: last + 1]   # trim hand-free frames at both ends, exactly as the training clips were
    feats = make_features(pts, bundle['normalize'])
    x = torch.from_numpy(feats)[None].to(bundle['device'])
    logits = bundle['model'](x, torch.tensor([len(feats)]))
    probs = logits.float().softmax(-1)[0].cpu().numpy()
    words = greedy_decode(probs, bundle['vocab'], bundle['threshold'])
    for w in words:   # report frame positions in the caller's input, not the trimmed sequence
        w['start'] += int(first)
        w['end'] += int(first)
    return {'text': ' '.join(w['word'] for w in words), 'words': words,
            'latency_ms': round((time.perf_counter() - t0) * 1000, 1)}
