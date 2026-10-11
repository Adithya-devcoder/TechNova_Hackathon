"""
Tests for signbridge.preprocessing.sentence_buffer
====================================================
All tests use synthetic FrameLandmarks — no webcam or MediaPipe required.

Covered
-------
SentenceResult validation
  - Correct shape and dtype.
  - len(frames) == holistic.shape[0].
  - Frozen (immutable).
  - Empty frames raises ValueError.
  - Shape mismatch raises ValueError.

Constructor validation
  - Default construction succeeds.
  - max_frames=None allowed.
  - max_frames < 1 raises ValueError.
  - min_frames < 1 raises ValueError.
  - min_frames > max_frames raises ValueError.

State machine
  - Initial state is IDLE.
  - start() → RECORDING.
  - flush() → IDLE.
  - reset() → IDLE from any state.
  - start() when already RECORDING resets and continues (fresh sentence).

append()
  - Ignored (no error) when IDLE.
  - Non-FrameLandmarks raises TypeError.
  - Frames are stored in chronological order.
  - Frame count increments correctly.
  - Max-frames cap evicts oldest frame.

flush()
  - Returns None when IDLE.
  - Returns None when frame_count < min_frames (empty buffer).
  - Returns None when frame_count < min_frames (partial buffer).
  - Returns SentenceResult when frame_count >= min_frames.
  - Result has correct T (frame count), shape, dtype.
  - State is IDLE after flush regardless of outcome.
  - Buffer is empty after flush.
  - Calling flush() twice in a row: second call returns None.

Frame ordering
  - Frames appear in append() order (oldest = row 0).
  - Distinct per-frame values are preserved through flush().

Missing-landmark handling (NaN propagation)
  - Frame with left_hand_detected=False → LH slots in holistic are NaN.
  - Frame with right_hand_detected=False → RH slots in holistic are NaN.
  - Frame with pose_detected=False → Pose slots in holistic are NaN.
  - Frame with all detected → no NaN outside face block.
  - Mixed detection across frames handled independently per frame.
  - Face slots always NaN regardless of flags.

reset()
  - Discards frames.
  - State becomes IDLE.
  - reset() when already IDLE is safe (no error).
  - Frames after reset() are not returned by flush().

Detection flags preserved end-to-end
  - After flush(), SentenceResult.frames[i] retains the original flags.
  - Values in frames list are independent copies (not shared references).
"""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.interface.adapter import (
    LH_SLOT_END,
    LH_SLOT_START,
    POSE_SLOT_END,
    POSE_SLOT_START,
    RH_SLOT_END,
    RH_SLOT_START,
    HOLISTIC_SLOTS,
    HOLISTIC_VALUES,
)
from signbridge.preprocessing.landmark_extractor import FEATURE_DIM, FrameLandmarks
from signbridge.preprocessing.sentence_buffer import SentenceBuffer, SentenceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lm(
    left: bool = False,
    right: bool = False,
    pose: bool = False,
    fill: float = 0.0,
) -> FrameLandmarks:
    """Return a FrameLandmarks with a constant-fill vector."""
    vec = np.full(FEATURE_DIM, fill, dtype=np.float32)
    return FrameLandmarks(
        left_hand_detected=left,
        right_hand_detected=right,
        pose_detected=pose,
        vector=vec,
    )


def make_distinct_lm(value: float, **flags) -> FrameLandmarks:
    """Return a FrameLandmarks whose vector is ``value`` repeated (unique per frame)."""
    vec = np.full(FEATURE_DIM, value, dtype=np.float32)
    return FrameLandmarks(
        left_hand_detected=flags.get("left", True),
        right_hand_detected=flags.get("right", True),
        pose_detected=flags.get("pose", True),
        vector=vec,
    )


def fill_and_flush(n: int, **lm_kwargs) -> SentenceResult:
    """Start, append n identical frames, flush, and return the result."""
    buf = SentenceBuffer()
    buf.start()
    for _ in range(n):
        buf.append(make_lm(**lm_kwargs))
    return buf.flush()


# ---------------------------------------------------------------------------
# SentenceResult
# ---------------------------------------------------------------------------

class TestSentenceResult:
    def test_valid_construction(self):
        frames = [make_lm(pose=True)]
        holistic = np.zeros((1, HOLISTIC_SLOTS, HOLISTIC_VALUES), dtype=np.float32)
        r = SentenceResult(frames=frames, holistic=holistic)
        assert len(r.frames) == 1
        assert r.holistic.shape == (1, HOLISTIC_SLOTS, HOLISTIC_VALUES)

    def test_empty_frames_raises(self):
        holistic = np.zeros((0, HOLISTIC_SLOTS, HOLISTIC_VALUES), dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            SentenceResult(frames=[], holistic=holistic)

    def test_shape_mismatch_raises(self):
        frames = [make_lm(), make_lm()]
        holistic = np.zeros((3, HOLISTIC_SLOTS, HOLISTIC_VALUES), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            SentenceResult(frames=frames, holistic=holistic)

    def test_wrong_dtype_raises(self):
        frames = [make_lm()]
        holistic = np.zeros((1, HOLISTIC_SLOTS, HOLISTIC_VALUES), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            SentenceResult(frames=frames, holistic=holistic)

    def test_frozen(self):
        """SentenceResult is immutable."""
        frames = [make_lm()]
        holistic = np.zeros((1, HOLISTIC_SLOTS, HOLISTIC_VALUES), dtype=np.float32)
        r = SentenceResult(frames=frames, holistic=holistic)
        with pytest.raises((AttributeError, TypeError)):
            r.frames = []  # type: ignore


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_defaults(self):
        buf = SentenceBuffer()
        assert buf.state == "IDLE"
        assert buf.frame_count == 0
        assert buf.max_frames == 600
        assert buf.min_frames == 1

    def test_max_frames_none(self):
        buf = SentenceBuffer(max_frames=None)
        assert buf.max_frames is None

    def test_max_frames_zero_raises(self):
        with pytest.raises(ValueError, match="max_frames"):
            SentenceBuffer(max_frames=0)

    def test_max_frames_negative_raises(self):
        with pytest.raises(ValueError, match="max_frames"):
            SentenceBuffer(max_frames=-5)

    def test_min_frames_zero_raises(self):
        with pytest.raises(ValueError, match="min_frames"):
            SentenceBuffer(min_frames=0)

    def test_min_frames_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="min_frames"):
            SentenceBuffer(max_frames=10, min_frames=11)

    def test_min_equals_max_allowed(self):
        buf = SentenceBuffer(max_frames=5, min_frames=5)
        assert buf.min_frames == 5


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_initial_state_is_idle(self):
        assert SentenceBuffer().state == "IDLE"
        assert not SentenceBuffer().is_recording

    def test_start_enters_recording(self):
        buf = SentenceBuffer()
        buf.start()
        assert buf.state == "RECORDING"
        assert buf.is_recording

    def test_flush_returns_to_idle(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(pose=True))
        buf.flush()
        assert buf.state == "IDLE"

    def test_reset_from_recording_to_idle(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm())
        buf.reset()
        assert buf.state == "IDLE"

    def test_reset_from_idle_is_safe(self):
        buf = SentenceBuffer()
        buf.reset()   # must not raise
        assert buf.state == "IDLE"

    def test_start_when_already_recording_resets_buffer(self):
        """start() during RECORDING discards old frames — fresh sentence."""
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_distinct_lm(0.1))
        buf.append(make_distinct_lm(0.2))
        assert buf.frame_count == 2

        buf.start()   # ← restarts recording
        assert buf.state == "RECORDING"
        assert buf.frame_count == 0   # old frames discarded

    def test_flush_after_reset_returns_none(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(pose=True))
        buf.reset()
        assert buf.flush() is None


# ---------------------------------------------------------------------------
# append()
# ---------------------------------------------------------------------------

class TestAppend:
    def test_append_while_idle_ignored(self):
        buf = SentenceBuffer()
        buf.append(make_lm())   # must not raise
        assert buf.frame_count == 0

    def test_append_wrong_type_raises(self):
        buf = SentenceBuffer()
        buf.start()
        with pytest.raises(TypeError):
            buf.append(np.zeros(FEATURE_DIM, dtype=np.float32))  # type: ignore

    def test_frame_count_increments(self):
        buf = SentenceBuffer()
        buf.start()
        for i in range(5):
            buf.append(make_lm())
            assert buf.frame_count == i + 1

    def test_max_frames_cap_evicts_oldest(self):
        buf = SentenceBuffer(max_frames=3)
        buf.start()
        for i in range(5):
            buf.append(make_distinct_lm(float(i), left=True))
        assert buf.frame_count == 3
        # After capping, the 3 newest frames (values 2.0, 3.0, 4.0) remain.
        result = buf.flush()
        assert result.holistic[0, LH_SLOT_START, 0] == pytest.approx(2.0)
        assert result.holistic[1, LH_SLOT_START, 0] == pytest.approx(3.0)
        assert result.holistic[2, LH_SLOT_START, 0] == pytest.approx(4.0)

    def test_max_frames_none_no_eviction(self):
        buf = SentenceBuffer(max_frames=None)
        buf.start()
        N = 200
        for _ in range(N):
            buf.append(make_lm(pose=True))
        assert buf.frame_count == N


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------

class TestFlush:
    def test_flush_while_idle_returns_none(self):
        assert SentenceBuffer().flush() is None

    def test_flush_empty_buffer_returns_none(self):
        buf = SentenceBuffer()
        buf.start()
        assert buf.flush() is None   # 0 frames < min_frames=1

    def test_flush_below_min_frames_returns_none(self):
        buf = SentenceBuffer(min_frames=5)
        buf.start()
        buf.append(make_lm(pose=True))
        buf.append(make_lm(pose=True))
        assert buf.frame_count == 2
        result = buf.flush()
        assert result is None

    def test_flush_at_min_frames_returns_result(self):
        buf = SentenceBuffer(min_frames=3)
        buf.start()
        for _ in range(3):
            buf.append(make_lm(pose=True))
        result = buf.flush()
        assert result is not None

    def test_flush_returns_sentence_result(self):
        result = fill_and_flush(5, pose=True)
        assert isinstance(result, SentenceResult)

    def test_flush_result_shape(self):
        result = fill_and_flush(7, pose=True)
        assert result.holistic.shape == (7, HOLISTIC_SLOTS, HOLISTIC_VALUES)

    def test_flush_result_dtype(self):
        result = fill_and_flush(3, pose=True)
        assert result.holistic.dtype == np.float32

    def test_flush_state_idle_after_success(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(pose=True))
        buf.flush()
        assert buf.state == "IDLE"

    def test_flush_state_idle_after_none(self):
        buf = SentenceBuffer(min_frames=10)
        buf.start()
        buf.append(make_lm())
        buf.flush()
        assert buf.state == "IDLE"

    def test_flush_clears_buffer(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(pose=True))
        buf.flush()
        assert buf.frame_count == 0

    def test_second_flush_returns_none(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(pose=True))
        buf.flush()
        assert buf.flush() is None   # state is IDLE; nothing buffered


# ---------------------------------------------------------------------------
# Frame ordering
# ---------------------------------------------------------------------------

class TestFrameOrdering:
    def test_chronological_order_in_holistic(self):
        """Row 0 of holistic corresponds to the first appended frame."""
        buf = SentenceBuffer()
        buf.start()
        # Each frame has a unique fill value in the left-hand segment.
        VALUES = [0.10, 0.20, 0.30, 0.40, 0.50]
        for v in VALUES:
            buf.append(make_distinct_lm(v, left=True, right=False, pose=False))
        result = buf.flush()
        for i, v in enumerate(VALUES):
            # LH wrist x is at slot 468, axis-1, axis-2 index 0
            assert result.holistic[i, LH_SLOT_START, 0] == pytest.approx(v)

    def test_chronological_order_in_frames_list(self):
        """frames list matches append() order."""
        buf = SentenceBuffer()
        buf.start()
        lms = [make_distinct_lm(float(i), pose=True) for i in range(6)]
        for lm in lms:
            buf.append(lm)
        result = buf.flush()
        for i, lm in enumerate(lms):
            assert result.frames[i] is lm

    def test_frames_list_length_matches_t(self):
        result = fill_and_flush(12, pose=True)
        assert len(result.frames) == result.holistic.shape[0] == 12


# ---------------------------------------------------------------------------
# Missing-landmark handling (NaN propagation)
# ---------------------------------------------------------------------------

class TestMissingLandmarkNaN:
    def test_left_hand_absent_slots_nan(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(left=False, right=True, pose=True, fill=1.0))
        result = buf.flush()
        assert np.all(np.isnan(result.holistic[0, LH_SLOT_START:LH_SLOT_END]))

    def test_right_hand_absent_slots_nan(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(left=True, right=False, pose=True, fill=1.0))
        result = buf.flush()
        assert np.all(np.isnan(result.holistic[0, RH_SLOT_START:RH_SLOT_END]))

    def test_pose_absent_slots_nan(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(left=True, right=True, pose=False, fill=1.0))
        result = buf.flush()
        assert np.all(np.isnan(result.holistic[0, POSE_SLOT_START:POSE_SLOT_END]))

    def test_all_detected_no_nan_outside_face(self):
        """When all segments detected, non-face slots must contain no NaN."""
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(left=True, right=True, pose=True, fill=0.5))
        result = buf.flush()
        non_face = np.concatenate([
            result.holistic[0, LH_SLOT_START:LH_SLOT_END].ravel(),
            result.holistic[0, POSE_SLOT_START:POSE_SLOT_END].ravel(),
            result.holistic[0, RH_SLOT_START:RH_SLOT_END].ravel(),
        ])
        assert not np.any(np.isnan(non_face))

    def test_face_always_nan(self):
        """Face slots [0:468] are always NaN."""
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(left=True, right=True, pose=True, fill=1.0))
        result = buf.flush()
        assert np.all(np.isnan(result.holistic[0, :468]))

    def test_mixed_detection_per_frame(self):
        """NaN is determined per-frame, independently."""
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(left=True,  right=False, pose=False, fill=1.0))  # frame 0
        buf.append(make_lm(left=False, right=True,  pose=False, fill=1.0))  # frame 1
        buf.append(make_lm(left=False, right=False, pose=True,  fill=1.0))  # frame 2
        result = buf.flush()

        # frame 0: LH present, RH and pose NaN
        assert not np.any(np.isnan(result.holistic[0, LH_SLOT_START:LH_SLOT_END]))
        assert np.all(np.isnan(result.holistic[0, RH_SLOT_START:RH_SLOT_END]))
        assert np.all(np.isnan(result.holistic[0, POSE_SLOT_START:POSE_SLOT_END]))

        # frame 1: RH present, LH and pose NaN
        assert np.all(np.isnan(result.holistic[1, LH_SLOT_START:LH_SLOT_END]))
        assert not np.any(np.isnan(result.holistic[1, RH_SLOT_START:RH_SLOT_END]))
        assert np.all(np.isnan(result.holistic[1, POSE_SLOT_START:POSE_SLOT_END]))

        # frame 2: pose present, hands NaN
        assert np.all(np.isnan(result.holistic[2, LH_SLOT_START:LH_SLOT_END]))
        assert np.all(np.isnan(result.holistic[2, RH_SLOT_START:RH_SLOT_END]))
        assert not np.any(np.isnan(result.holistic[2, POSE_SLOT_START:POSE_SLOT_END]))


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_frames(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(pose=True))
        buf.append(make_lm(pose=True))
        buf.reset()
        assert buf.frame_count == 0

    def test_reset_sets_idle(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm())
        buf.reset()
        assert buf.state == "IDLE"

    def test_reset_idle_is_safe(self):
        SentenceBuffer().reset()   # must not raise

    def test_frames_after_reset_not_flushed(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(pose=True))
        buf.reset()
        # Need to start again for flush to not return None immediately
        result = buf.flush()  # IDLE → None
        assert result is None


# ---------------------------------------------------------------------------
# Detection flags preserved end-to-end
# ---------------------------------------------------------------------------

class TestFlagsPreservedEndToEnd:
    def test_left_hand_flag_preserved(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(left=True, right=False, pose=False))
        result = buf.flush()
        assert result.frames[0].left_hand_detected is True
        assert result.frames[0].right_hand_detected is False
        assert result.frames[0].pose_detected is False

    def test_pose_flag_preserved(self):
        buf = SentenceBuffer()
        buf.start()
        buf.append(make_lm(left=False, right=False, pose=True))
        result = buf.flush()
        assert result.frames[0].pose_detected is True

    def test_frames_are_same_objects(self):
        """The frames list in SentenceResult contains the exact objects appended."""
        buf = SentenceBuffer()
        buf.start()
        lm = make_lm(left=True, pose=True)
        buf.append(lm)
        result = buf.flush()
        assert result.frames[0] is lm

    def test_flags_consistent_with_nan_pattern(self):
        """For every frame, NaN slots in holistic are consistent with the flags."""
        buf = SentenceBuffer()
        buf.start()
        lms = [
            make_lm(left=True,  right=False, pose=True,  fill=0.5),
            make_lm(left=False, right=True,  pose=False, fill=0.5),
        ]
        for lm in lms:
            buf.append(lm)
        result = buf.flush()

        for i, lm in enumerate(lms):
            if lm.left_hand_detected:
                assert not np.any(np.isnan(result.holistic[i, LH_SLOT_START:LH_SLOT_END]))
            else:
                assert np.all(np.isnan(result.holistic[i, LH_SLOT_START:LH_SLOT_END]))

            if lm.right_hand_detected:
                assert not np.any(np.isnan(result.holistic[i, RH_SLOT_START:RH_SLOT_END]))
            else:
                assert np.all(np.isnan(result.holistic[i, RH_SLOT_START:RH_SLOT_END]))

            if lm.pose_detected:
                assert not np.any(np.isnan(result.holistic[i, POSE_SLOT_START:POSE_SLOT_END]))
            else:
                assert np.all(np.isnan(result.holistic[i, POSE_SLOT_START:POSE_SLOT_END]))
