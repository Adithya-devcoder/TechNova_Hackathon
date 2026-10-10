"""Measure predict() latency on this machine for typical sentence lengths.

    python -m evaluation.benchmark_latency
"""
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import signbridge_model as sb  # noqa: E402

LENGTHS = (30, 60, 90, 150, 300)   # frames at 30 fps: 1 s to 10 s of signing
RUNS = 20


def fake_sentence(n_frames, rng):
    """Landmarks with pose and both hands present and moving, the costliest realistic input."""
    raw = np.full((n_frames, 543, 3), np.nan, np.float32)
    rows = sb.LH + sb.RH + sb.POSE
    base = rng.uniform(0.3, 0.7, (1, len(rows), 3))
    raw[:, rows] = base + np.cumsum(rng.normal(0, 0.003, (n_frames, len(rows), 3)), axis=0)
    return raw


def main():
    bundle = sb.load_model()
    rng = np.random.default_rng(0)
    print(f'CPU: {platform.processor() or platform.machine()} | torch threads: {torch.get_num_threads()}\n')
    print('| frames | seconds of signing | median latency (ms) | p90 (ms) |')
    print('|---|---|---|---|')
    for n in LENGTHS:
        x = fake_sentence(n, rng)
        sb.predict(x, bundle)   # warm-up
        times = [sb.predict(x, bundle)['latency_ms'] for _ in range(RUNS)]
        print(f'| {n} | {n / 30:.1f} | {np.median(times):.1f} | {np.percentile(times, 90):.1f} |')


if __name__ == '__main__':
    main()
