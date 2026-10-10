"""Run the trained model on one saved clip and check that the clip follows the input contract.

    python -m models.predict_clip clip.npy

clip.npy: float32 array [T, 543, 3] for ONE whole signed sentence, MediaPipe Holistic row order
(face 0-467, left hand 468-488, pose 489-521, right hand 522-542), raw normalised x, y, z, NaN = missing.
"""
import json
import sys

import numpy as np

from models import signbridge_model as sb

PARTS = {'left hand': slice(468, 489), 'pose': slice(489, 522), 'right hand': slice(522, 543)}


def check_clip(clip):
    """Return a list of problems with the clip; an empty list means it matches the contract."""
    problems = []
    if clip.ndim != 3 or clip.shape[1:] != (543, 3):
        return [f'shape is {clip.shape}, expected (T, 543, 3)']
    for name, rows in PARTS.items():
        zero_frames = int((clip[:, rows] == 0).all(axis=(1, 2)).sum())
        if zero_frames:
            problems.append(f'{name} is all zeros in {zero_frames} frames: missing landmarks must be NaN, not 0')
    finite = clip[np.isfinite(clip[..., 0])]
    if len(finite) and not (-0.5 <= np.median(finite[:, 0]) <= 1.5):
        problems.append('x values are not normalised image coordinates (expected roughly 0-1)')
    if not np.isfinite(clip[:, PARTS['pose'], 0]).any():
        problems.append('pose is never detected: the model needs the shoulders to normalise the signer')
    return problems


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    clip = np.load(sys.argv[1]).astype(np.float32)
    problems = check_clip(clip)
    for p in problems:
        print('WARNING:', p)
    if clip.ndim != 3 or clip.shape[1:] != (543, 3):
        sys.exit(1)
    hands = np.isfinite(clip[:, 468:489, 0]).any(1) | np.isfinite(clip[:, 522:543, 0]).any(1)
    print(f'clip: {len(clip)} frames, hands visible in {int(hands.sum())}')
    print(json.dumps(sb.predict(clip, sb.load_model()), indent=1))


if __name__ == '__main__':
    main()
