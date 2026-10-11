# %% [markdown]
# # SignBridge: continuous sign-to-text (PS-41), Member 1 model notebook
# Run on Kaggle with the `asl-signs` competition data attached and **Accelerator = GPU** and **Internet = On**.
#
# Pipeline: isolated ASL Signs clips → **signer-disjoint split** → synthetic continuous sentences
# (same signer, interpolated transitions, speed warp) → baseline (sliding window + LogisticRegression)
# vs **BiGRU + CTC** → WER on held-out signers → `predict()` for the app.
#
# Run the cells top to bottom. Each cell prints what it produced.

# %%
# CELL 1: setup and a first look at the data
!pip install -q jiwer
import os, sys, json, time, glob, random, warnings
import numpy as np, pandas as pd

def find_data():
    if os.environ.get('SIGNBRIDGE_DATA'):
        return os.environ['SIGNBRIDGE_DATA']
    hits = glob.glob('/kaggle/input/*/train.csv') + glob.glob('/kaggle/input/*/*/train.csv')
    assert hits, 'train.csv not found: attach the asl-signs competition data to this notebook'
    return os.path.dirname(hits[0])

DATA = find_data()
OUT = os.environ.get('SIGNBRIDGE_OUT', '/kaggle/working')
os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get('SIGNBRIDGE_SMOKE') == '1'   # tiny sizes, used only for local testing
N_TRAIN, N_VAL, N_TEST, N_BASE, EPOCHS = (128, 32, 48, 60, 1) if SMOKE else (20000, 600, 1000, 3000, 25)
NW = 0 if os.name == 'nt' else 4                     # DataLoader workers
NJOBS = 1 if os.name == 'nt' else os.cpu_count()      # parallel parquet reading

df = pd.read_csv(f'{DATA}/train.csv')
print('data dir:', DATA)
print(df.head())
print('clips:', len(df), '| signers:', df.participant_id.nunique(), '| signs:', df.sign.nunique())
print(sorted(df.sign.unique()))

# %%
# CELL 2: vocabulary (52 signs) and demo sentences in gloss order
VOCAB = [
    'mom', 'dad', 'grandma', 'grandpa', 'brother', 'girl', 'boy', 'child',          # people
    'minemy', 'weus', 'hesheit',                                                    # pronouns
    'go', 'drink', 'sleep', 'read', 'like', 'have', 'see', 'give', 'find', 'wait', 'ride', 'cry',  # actions
    'water', 'milk', 'food', 'apple', 'pizza', 'icecream',                          # food
    'home', 'outside', 'store', 'bed', 'car', 'book', 'toy',                        # places and things
    'happy', 'sad', 'hungry', 'thirsty', 'sleepy', 'sick',                          # feelings
    'now', 'later', 'tomorrow', 'yesterday', 'night',                               # time
    'please', 'thankyou', 'hello', 'yes', 'no',                                     # social
]
SENTENCES = [
    'hello grandma thankyou',
    'mom drink water now',
    'dad go store tomorrow',
    'brother sleepy go bed',
    'minemy brother sick',
    'girl like pizza',
    'boy have toy car',
    'weus go outside later',
    'child thirsty milk please',
    'grandpa read book night',
    'hesheit sad cry',
    'yesterday weus ride car home',
    'minemy dad give apple',
    'grandma hungry icecream',
    'child happy find toy',
    'wait please mom',
]
available = set(df.sign)
missing = [w for w in VOCAB if w not in available]
assert not missing, f'not in dataset, swap these for signs from the list above: {missing}'
assert len(set(VOCAB)) == len(VOCAB) >= 50
for s in SENTENCES:
    bad = [w for w in s.split() if w not in VOCAB]
    assert not bad, f'"{s}" uses words outside VOCAB: {bad}'
W2I = {w: i + 1 for i, w in enumerate(VOCAB)}   # 0 is the CTC blank
print(len(VOCAB), 'signs,', len(SENTENCES), 'sentences')
print(df[df.sign.isin(VOCAB)].sign.value_counts().describe()[['min', 'mean', 'max']].rename('clips per sign'))

# %%
%%writefile signbridge_model.py
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

# %%
# CELL 4: load the 52-sign subset and split by signer (no signer appears in two splits)
import importlib
sys.path.insert(0, os.getcwd())
import signbridge_model as sb
importlib.reload(sb)
from joblib import Parallel, delayed

sub = df[df.sign.isin(VOCAB)].reset_index(drop=True)

def load_clip(path):
    xyz = pd.read_parquet(f'{DATA}/{path}', columns=['x', 'y', 'z']).values
    pts = sb.select_points(xyz.reshape(-1, 543, 3))
    hands = np.isfinite(pts[:, :42, 0]).any(1)
    if not hands.any():
        return None                              # no hand ever detected, so the clip is useless
    idx = np.where(hands)[0]
    return pts[idx[0]: idx[-1] + 1].astype(np.float16)   # trim idle frames at the start and end

t0 = time.time()
loaded = Parallel(n_jobs=NJOBS, batch_size=64)(delayed(load_clip)(p) for p in sub.path)
keep = [i for i, c in enumerate(loaded) if c is not None]
CLIPS = [loaded[i] for i in keep]
sub = sub.iloc[keep].reset_index(drop=True)
print(f'loaded {len(CLIPS)} clips in {time.time() - t0:.0f}s (dropped {len(loaded) - len(keep)} without hands)')
print('frames per clip: median', int(np.median([len(c) for c in CLIPS])))

rng = np.random.default_rng(42)
parts = sorted(int(p) for p in sub.participant_id.unique())
rng.shuffle(parts)
SPLIT = {'test': sorted(parts[:3]), 'val': sorted(parts[3:6]), 'train': sorted(parts[6:])}
assert not (set(SPLIT['train']) & set(SPLIT['val']) or set(SPLIT['train']) & set(SPLIT['test'])
            or set(SPLIT['val']) & set(SPLIT['test'])), 'signer leakage'
json.dump(SPLIT, open(f'{OUT}/split.json', 'w'), indent=1)

POOLS = {k: {} for k in SPLIT}
p2split = {p: k for k, ps in SPLIT.items() for p in ps}
for i, (pid, sign) in enumerate(zip(sub.participant_id, sub.sign)):
    POOLS[p2split[int(pid)]].setdefault(int(pid), {}).setdefault(sign, []).append(i)
for k in SPLIT:
    n = sum(len(v) for d in POOLS[k].values() for v in d.values())
    absent = [w for w in VOCAB if not any(w in d for d in POOLS[k].values())]
    assert not absent, f'{k} signers never sign {absent}: swap those words in CELL 2'
    print(f'{k:5s}: signers {SPLIT[k]} -> {n} clips')

# %%
# CELL 5: synthetic continuous sentences built inside each split
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def sample_words(rng):
    """Half are demo-style template sentences, half are random sign sequences (so the model can't just memorise sentences)."""
    if rng.random() < 0.5:
        return SENTENCES[rng.integers(len(SENTENCES))].split(), 'template'
    return [str(w) for w in rng.choice(VOCAB, size=int(rng.integers(2, 6)), replace=False)], 'random'

def warp(c, rng):
    n = max(2, int(round(len(c) * rng.uniform(0.8, 1.25))))
    return c[np.round(np.linspace(0, len(c) - 1, n)).astype(int)]

def synth_sentence(words, split, rng, transitions=True):
    """Join clips of `words` from one signer of `split`, with interpolated transitions between signs."""
    pool = POOLS[split]
    owners = [p for p in pool if all(w in pool[p] for w in words)]
    signer = owners[rng.integers(len(owners))] if owners else list(pool)[rng.integers(len(pool))]
    segs, spans, t = [], [], 0
    for i, w in enumerate(words):
        if w in pool[signer]:
            src = signer
        else:   # this signer never signed w, so borrow another signer from the same split
            alt = [p for p in pool if w in pool[p]]
            src = alt[rng.integers(len(alt))]
        clips = pool[src][w]
        c = warp(CLIPS[clips[rng.integers(len(clips))]].astype(np.float32), rng)
        if i > 0 and transitions:
            k = int(rng.integers(3, 9))
            wgt = np.linspace(0, 1, k + 2)[1:-1, None, None].astype(np.float32)
            tr = (1 - wgt) * segs[-1][-1][None] + wgt * c[0][None]
            segs.append(tr)
            t += k
        segs.append(c)
        spans.append((t, t + len(c)))
        t += len(c)
    return np.concatenate(segs), spans

def augment(p, rng):
    th = np.deg2rad(rng.uniform(-12, 12))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32) * rng.uniform(0.9, 1.1)
    p = (p - 0.5) @ R.T + 0.5 + rng.uniform(-0.05, 0.05, 2).astype(np.float32)
    p = p + rng.normal(0, 0.002, p.shape).astype(np.float32)
    if rng.random() < 0.3:   # simulate a hand the detector lost for a few frames
        a = int(rng.integers(len(p)))
        p[a: a + int(rng.integers(1, 8)), sb.I_LH if rng.random() < 0.5 else sb.I_RH] = np.nan
    return p.astype(np.float32)

class SentenceDS(Dataset):
    """seed=None draws fresh sentences every time (training); a fixed seed gives a fixed set (val/test)."""
    def __init__(self, split, n, seed=None, augment=False, normalize=True, transitions=True):
        self.split, self.n, self.seed = split, n, seed
        self.augment, self.normalize, self.transitions = augment, normalize, transitions
    def __len__(self):
        return self.n
    def item(self, i):
        rng = np.random.default_rng(None if self.seed is None else self.seed * 100003 + i)
        words, kind = sample_words(rng)
        pts, spans = synth_sentence(words, self.split, rng, self.transitions)
        if self.augment:
            pts = augment(pts, rng)
        return pts, spans, words, kind
    def __getitem__(self, i):
        pts, _, words, kind = self.item(i)
        x = sb.make_features(pts, self.normalize)
        return torch.from_numpy(x), torch.tensor([W2I[w] for w in words]), ' '.join(words), kind

def collate(batch):
    xs, ys, texts, kinds = zip(*batch)
    return (nn.utils.rnn.pad_sequence(xs, batch_first=True), torch.tensor([len(x) for x in xs]),
            torch.cat(ys), torch.tensor([len(y) for y in ys]), list(texts), list(kinds))

x, y, text, kind = SentenceDS('train', 1, seed=0)[0]
print('example:', text, f'({kind}) ->', tuple(x.shape), 'features')
test_ds = SentenceDS('test', N_TEST, seed=123)   # one fixed test set for the baseline and the model

# %%
# CELL 6: baseline = sliding window + LogisticRegression, scored on the same held-out-signer test set
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

WIN, STRIDE = 20, 5

def window_stats(x):
    pos = x[:, :sb.N_PTS * 2]
    starts = list(range(0, max(len(x) - WIN, 0) + 1, STRIDE))
    rows = [np.concatenate([w.mean(0), w.std(0), w.min(0), w.max(0)]) for w in (pos[s:s + WIN] for s in starts)]
    return np.array(rows), starts

def window_labels(starts, spans, words, T):
    """A window's label is the sign covering at least 60% of it, otherwise 0 (transition / no sign)."""
    labels = []
    for s in starts:
        e, lab = min(s + WIN, T), 0
        for (a, b), w in zip(spans, words):
            if min(e, b) - max(s, a) >= 0.6 * (e - s):
                lab = W2I[w]
        labels.append(lab)
    return labels

t0 = time.time()
rng = np.random.default_rng(0)
Xb, yb = [], []
for _ in range(N_BASE):
    words, _ = sample_words(rng)
    pts, spans = synth_sentence(words, 'train', rng)
    x = sb.make_features(pts)
    rows, starts = window_stats(x)
    Xb.append(rows)
    yb += window_labels(starts, spans, words, len(x))
baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
baseline.fit(np.concatenate(Xb), yb)

def baseline_predict(x):
    out, prev = [], 0
    for p in baseline.predict(window_stats(x)[0]):
        if p != 0 and p != prev:
            out.append(VOCAB[p - 1])
        prev = p
    return ' '.join(out)

base_rows = []
for i in range(len(test_ds)):
    x, _, text, kind = test_ds[i]
    base_rows.append((text, baseline_predict(x.numpy()), kind))
BASE_WER = {k: sb.wer([r for r, _, kk in base_rows if k in ('all', kk)], [h for _, h, kk in base_rows if k in ('all', kk)])
            for k in ('all', 'template', 'random')}
print(f'baseline trained in {time.time() - t0:.0f}s | held-out-signer WER: ' + ', '.join(f'{k} {v:.1%}' for k, v in BASE_WER.items()))

# %%
# CELL 7: train BiGRU + CTC (one epoch should take well under a minute on a Kaggle GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_CFG = {'hidden': 256, 'layers': 2, 'dropout': 0.2}
print('device:', device)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    rows = []
    for X, lx, _, _, texts, kinds in loader:
        probs = model(X.to(device), lx).float().softmax(-1).cpu().numpy()
        for b in range(len(texts)):
            words = sb.greedy_decode(probs[b, :lx[b]], VOCAB)
            conf = float(np.mean([w['confidence'] for w in words])) if words else 0.0
            rows.append((texts[b], ' '.join(w['word'] for w in words), kinds[b], conf))
    w = {k: sb.wer([r[0] for r in rows if k in ('all', r[2])], [r[1] for r in rows if k in ('all', r[2])])
         for k in ('all', 'template', 'random')}
    return w, rows

def run_training(tag, normalize=True, transitions=True, epochs=EPOCHS):
    tl = DataLoader(SentenceDS('train', N_TRAIN, augment=True, normalize=normalize, transitions=transitions),
                    batch_size=64, shuffle=True, num_workers=NW, collate_fn=collate, persistent_workers=NW > 0)
    vl = DataLoader(SentenceDS('val', N_VAL, seed=7, normalize=normalize), batch_size=128,
                    num_workers=NW, collate_fn=collate)
    torch.manual_seed(0)
    model = sb.GRUCTC(sb.FEAT_DIM, len(VOCAB) + 1, **MODEL_CFG).to(device)
    print(f'[{tag}] parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(tl))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best = float('inf')
    for ep in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for X, lx, Y, ly, _, _ in tl:
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(X.to(device), lx)
            loss = ctc(logits.float().log_softmax(-1).transpose(0, 1), Y.to(device), lx, ly)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        val_wer, _ = evaluate(model, vl)
        mark = ''
        if val_wer['all'] < best:
            best, mark = val_wer['all'], '  <- best, saved'
            torch.save({'state': {k: v.cpu() for k, v in model.state_dict().items()}, 'vocab': VOCAB,
                        'in_dim': sb.FEAT_DIM, 'model_cfg': MODEL_CFG, 'normalize': normalize,
                        'threshold': 0.6, 'split': SPLIT}, f'{OUT}/model_{tag}.pt')
        print(f'[{tag}] epoch {ep:2d}/{epochs} loss {total / len(tl):.3f} | val WER {val_wer["all"]:.1%} '
              f'| {time.time() - t0:.0f}s{mark}')
    return best

run_training('main')

# %%
# CELL 8: final test on held-out signers, latency, an example prediction, and results.md
import platform
test_dl = DataLoader(test_ds, batch_size=128, num_workers=NW, collate_fn=collate)
bundle = sb.load_model(f'{OUT}/model_main.pt', device)
MODEL_WER, model_rows = evaluate(bundle['model'], test_dl)

pd.DataFrame([{'reference': r[0], 'model': r[1], 'baseline': b[1], 'kind': r[2], 'mean_confidence': round(r[3], 3)}
              for r, b in zip(model_rows, base_rows)]).to_csv(f'{OUT}/test_predictions.csv', index=False)
correct = [r[3] for r in model_rows if r[0] == r[1]]
wrong = [r[3] for r in model_rows if r[0] != r[1]]

# Latency: one whole sentence through predict(), from raw [T, 543, 3] landmarks to text
pts, _, words, _ = test_ds.item(0)
raw = np.full((len(pts), 543, 3), np.nan, np.float32)
raw[:, sb.POINTS, :2] = pts
lat = {}
for dev in (['cpu', 'cuda'] if device == 'cuda' else ['cpu']):
    b = sb.load_model(f'{OUT}/model_main.pt', dev)
    sb.predict(raw, b)   # warm-up
    lat[dev] = float(np.median([sb.predict(raw, b)['latency_ms'] for _ in range(30)]))
example = sb.predict(raw, bundle)
print('reference :', ' '.join(words))
print('prediction:', json.dumps(example, indent=1)[:800])

try:
    import jiwer   # independent cross-check of our WER
    print(f'jiwer check, model WER all: {jiwer.wer([r[0] for r in model_rows], [r[1] or "<empty>" for r in model_rows]):.1%}')
except ImportError:
    pass

fmt = lambda d: ' | '.join(f'{d[k]:.1%}' for k in ('all', 'template', 'random'))
results = f"""# SignBridge model results (measured, {time.strftime('%Y-%m-%d %H:%M')})

- **Data:** Google ASL Signs (Kaggle `asl-signs`), {len(VOCAB)} signs, {len(CLIPS)} isolated clips.
- **Split by signer:** train {SPLIT['train']}, val {SPLIT['val']}, **test {SPLIT['test']}**. Test signers are never seen in training.
- **Test set:** {len(test_ds)} synthetic continuous sentences built only from the test signers' clips:
  {sum(r[2] == 'template' for r in model_rows)} demo-style template sentences and {sum(r[2] == 'random' for r in model_rows)} random sign sequences.

| Method | WER all | WER template sentences | WER random sequences |
|---|---|---|---|
| Baseline: sliding window + LogisticRegression | {fmt(BASE_WER)} |
| **BiGRU + CTC (ours)** | {fmt(MODEL_WER)} |

- Sentences exactly right (model): {len(correct)}/{len(model_rows)}
- Mean word confidence: {np.mean(correct) if correct else float('nan'):.2f} on correct sentences, {np.mean(wrong) if wrong else float('nan'):.2f} on wrong ones
- predict() latency for one sentence of {len(pts)} frames: """ + ', '.join(f'{k} {v:.1f} ms' for k, v in lat.items()) + f"""
- Hardware: {platform.processor() or platform.machine()}{', GPU ' + torch.cuda.get_device_name(0) if device == 'cuda' else ''}; {EPOCHS} epochs

**Limitations:** sentences are synthesised from isolated clips, so real co-articulation is only approximated
by interpolated transitions. Template sentences also appear in training (with other signers), so their WER is
optimistic; the random-sequence WER is the purer recognition number. No live webcam test yet.
"""
open(f'{OUT}/results.md', 'w').write(results)
print(results)

# %%
# CELL 8b: fairer baseline - output a word only when K consecutive windows agree (K tuned on VALIDATION signers)
def decode_runs(pred, k):
    runs = []
    for p in pred:
        if runs and runs[-1][0] == p:
            runs[-1][1] += 1
        else:
            runs.append([p, 1])
    words = [VOCAB[c - 1] for c, n in runs if c != 0 and n >= k]
    return ' '.join(w for i, w in enumerate(words) if i == 0 or w != words[i - 1])

def window_preds(ds):
    out = []
    for i in range(len(ds)):
        x, _, text, kind = ds[i]
        out.append((text, kind, baseline.predict(window_stats(x.numpy())[0])))
    return out

val_preds = window_preds(SentenceDS('val', N_VAL, seed=7))
val_scores = {k: sb.wer([t for t, _, _ in val_preds], [decode_runs(p, k) for _, _, p in val_preds]) for k in range(1, 7)}
BEST_K = min(val_scores, key=val_scores.get)
print('val WER by K:', {k: f'{v:.1%}' for k, v in val_scores.items()}, '-> K =', BEST_K)

test_preds = window_preds(test_ds)
SMOOTH_WER = {kind: sb.wer([t for t, kk, _ in test_preds if kind in ('all', kk)],
                           [decode_runs(p, BEST_K) for _, kk, p in test_preds if kind in ('all', kk)])
              for kind in ('all', 'template', 'random')}
line = f'| Baseline + smoothing (K={BEST_K}, tuned on val signers) | {fmt(SMOOTH_WER)} |\n'
results = results.replace('| **BiGRU + CTC (ours)**', line + '| **BiGRU + CTC (ours)**')
open(f'{OUT}/results.md', 'w').write(results)
print(results)

# %%
# CELL 9 (optional, only if time allows): ablation without signer normalisation
run_training('no_norm', normalize=False)
abl = sb.load_model(f'{OUT}/model_no_norm.pt', device)
abl_dl = DataLoader(SentenceDS('test', N_TEST, seed=123, normalize=False), batch_size=128, num_workers=NW, collate_fn=collate)
ABL_WER, _ = evaluate(abl['model'], abl_dl)
line = f'| Ablation: BiGRU + CTC without signer normalisation | {fmt(ABL_WER)} |\n'
results = results.replace('| **BiGRU + CTC (ours)**', line + '| **BiGRU + CTC (ours)**')
open(f'{OUT}/results.md', 'w').write(results)
print(line)

# %%
# CELL 10: bundle the files to download from the Output panel
import zipfile
with zipfile.ZipFile(f'{OUT}/signbridge_member1.zip', 'w') as z:
    for f in ['model_main.pt', 'split.json', 'results.md', 'test_predictions.csv']:
        if os.path.exists(f'{OUT}/{f}'):
            z.write(f'{OUT}/{f}', f)
    z.write('signbridge_model.py', 'signbridge_model.py')
print('download /kaggle/working/signbridge_member1.zip from the Output panel on the right')
