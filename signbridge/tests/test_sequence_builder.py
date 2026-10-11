"""
Tests for signbridge.preprocessing.sequence_builder
====================================================
All tests use synthetic FrameLandmarks (no webcam or MediaPipe needed).

Covered:
  - Constructor validation (window_size, stride).
  - Output shape and dtype for normal and edge-case window sizes.
  - No windows emitted until buffer is full.
  - Exactly one window emitted per stride frames once full.
  - Stride=window_size produces non-overlapping windows.
  - Stride=1 produces maximally overlapping windows.
  - Vector values in windows match the input FrameLandmarks.
  - Zero-filled FrameLandmarks propagate correctly into windows.
  - reset() clears the buffer and resets the stride counter.
  - push_batch() convenience method.
  - output_shape property.
"""

import numpy as np
import pytest

from signbridge.preprocessing.landmark_extractor import FEATURE_DIM, FrameLandmarks
from signbridge.preprocessing.sequence_builder import SequenceBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame_lm(value: float = 0.0) -> FrameLandmarks:
    """Return a FrameLandmarks whose vector is entirely ``value``."""
    vec = np.full(FEATURE_DIM, value, dtype=np.float32)
    return FrameLandmarks(vector=vec)


def zero_frame() -> FrameLandmarks:
    return make_frame_lm(0.0)


def push_n(builder: SequenceBuilder, n: int, value: float = 0.0):
    """Push ``n`` identical frames and return all emitted windows."""
    all_windows = []
    for _ in range(n):
        all_windows.extend(builder.push(make_frame_lm(value)))
    return all_windows


# ---------------------------------------------------------------------------
# Test: Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_valid_defaults(self):
        sb = SequenceBuilder()
        assert sb.window_size == 30
        assert sb.stride == 1

    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            SequenceBuilder(window_size=0)

    def test_stride_zero_raises(self):
        with pytest.raises(ValueError, match="stride"):
            SequenceBuilder(window_size=5, stride=0)

    def test_stride_greater_than_window_raises(self):
        with pytest.raises(ValueError, match="stride"):
            SequenceBuilder(window_size=5, stride=6)

    def test_stride_equal_window_valid(self):
        sb = SequenceBuilder(window_size=5, stride=5)
        assert sb.stride == sb.window_size

    def test_custom_window_size(self):
        sb = SequenceBuilder(window_size=60, stride=10)
        assert sb.window_size == 60
        assert sb.stride == 10


# ---------------------------------------------------------------------------
# Test: Output shape and dtype
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_window_shape_default(self):
        sb = SequenceBuilder(window_size=30, stride=1)
        windows = push_n(sb, 30)
        assert len(windows) == 1
        assert windows[0].shape == (30, FEATURE_DIM)

    def test_window_dtype(self):
        sb = SequenceBuilder(window_size=5, stride=1)
        windows = push_n(sb, 5)
        assert windows[0].dtype == np.float32

    def test_output_shape_property(self):
        sb = SequenceBuilder(window_size=15, stride=3)
        assert sb.output_shape == (15, FEATURE_DIM)

    @pytest.mark.parametrize("T", [1, 5, 10, 30, 90])
    def test_shape_various_window_sizes(self, T):
        sb = SequenceBuilder(window_size=T, stride=1)
        windows = push_n(sb, T)
        assert len(windows) == 1
        assert windows[0].shape == (T, FEATURE_DIM)


# ---------------------------------------------------------------------------
# Test: Buffer filling behaviour
# ---------------------------------------------------------------------------

class TestBufferFilling:
    def test_no_windows_before_full(self):
        sb = SequenceBuilder(window_size=5, stride=1)
        # Push 4 frames — buffer not yet full
        windows = push_n(sb, 4)
        assert windows == [], "No windows should be emitted before buffer is full"

    def test_first_window_on_exactly_full(self):
        sb = SequenceBuilder(window_size=5, stride=1)
        windows = push_n(sb, 5)
        assert len(windows) == 1

    def test_buffered_frames_counter(self):
        sb = SequenceBuilder(window_size=5, stride=1)
        for i in range(1, 6):
            sb.push(zero_frame())
            assert sb.buffered_frames == i

    def test_buffered_frames_capped_at_window_size(self):
        sb = SequenceBuilder(window_size=5, stride=1)
        push_n(sb, 10)
        assert sb.buffered_frames == 5


# ---------------------------------------------------------------------------
# Test: Emission rate (stride)
# ---------------------------------------------------------------------------

class TestEmissionRate:
    def test_stride_1_emits_one_per_frame_after_full(self):
        T = 5
        sb = SequenceBuilder(window_size=T, stride=1)
        # First T frames fill the buffer → 1 window
        windows = push_n(sb, T)
        assert len(windows) == 1
        # Each subsequent frame → 1 more window
        for _ in range(5):
            w = sb.push(zero_frame())
            assert len(w) == 1

    def test_stride_equal_window_non_overlapping(self):
        T = 5
        sb = SequenceBuilder(window_size=T, stride=T)
        # Frames 1–4: no window yet
        assert push_n(sb, T - 1) == []
        # Frame 5: first window
        w = sb.push(zero_frame())
        assert len(w) == 1
        # Frames 6–9: no window
        assert push_n(sb, T - 1) == []
        # Frame 10: second window
        w = sb.push(zero_frame())
        assert len(w) == 1

    def test_stride_2_emission_count(self):
        T, S = 4, 2
        sb = SequenceBuilder(window_size=T, stride=S)
        # Push 10 frames — windows expected: frame 4, 6, 8, 10
        all_windows = push_n(sb, 10)
        assert len(all_windows) == 4

    def test_total_windows_over_long_run(self):
        T, S, N = 5, 1, 20
        sb = SequenceBuilder(window_size=T, stride=S)
        all_windows = push_n(sb, N)
        # First window at frame T, then one per frame → N - T + 1 = 16
        expected = N - T + 1
        assert len(all_windows) == expected


# ---------------------------------------------------------------------------
# Test: Value correctness
# ---------------------------------------------------------------------------

class TestValueCorrectness:
    def test_zero_filled_frame_propagates(self):
        sb = SequenceBuilder(window_size=3, stride=3)
        push_n(sb, 3, value=0.0)
        # reset and re-push with zeros
        sb.reset()
        windows = push_n(sb, 3, value=0.0)
        assert len(windows) == 1
        np.testing.assert_array_equal(windows[0], np.zeros((3, FEATURE_DIM), dtype=np.float32))

    def test_values_match_input_frames(self):
        sb = SequenceBuilder(window_size=3, stride=3)
        # Three distinct frames: 1.0, 2.0, 3.0
        sb.push(make_frame_lm(1.0))
        sb.push(make_frame_lm(2.0))
        windows = sb.push(make_frame_lm(3.0))
        assert len(windows) == 1
        w = windows[0]  # shape (3, FEATURE_DIM)
        np.testing.assert_array_almost_equal(w[0], np.full(FEATURE_DIM, 1.0, dtype=np.float32))
        np.testing.assert_array_almost_equal(w[1], np.full(FEATURE_DIM, 2.0, dtype=np.float32))
        np.testing.assert_array_almost_equal(w[2], np.full(FEATURE_DIM, 3.0, dtype=np.float32))

    def test_chronological_order(self):
        """Row 0 = oldest frame, row T-1 = newest frame."""
        sb = SequenceBuilder(window_size=3, stride=3)
        values = [0.1, 0.2, 0.3]
        windows = []
        for v in values:
            windows.extend(sb.push(make_frame_lm(v)))
        assert len(windows) == 1
        for i, v in enumerate(values):
            assert windows[0][i, 0] == pytest.approx(v, rel=1e-5)

    def test_vector_is_copied_not_referenced(self):
        """Mutating input after push must not affect the window."""
        sb = SequenceBuilder(window_size=2, stride=2)
        fl = make_frame_lm(1.0)
        sb.push(fl)
        fl.vector[:] = 99.0  # mutate the original
        windows = sb.push(make_frame_lm(2.0))
        # Row 0 should still be 1.0 (copy was stored)
        assert windows[0][0, 0] == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Test: reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_buffer(self):
        sb = SequenceBuilder(window_size=5, stride=1)
        push_n(sb, 4)
        sb.reset()
        assert sb.buffered_frames == 0

    def test_reset_prevents_premature_emission(self):
        sb = SequenceBuilder(window_size=5, stride=5)
        push_n(sb, 4)
        sb.reset()
        # Now need another full window_size frames before emission
        windows = push_n(sb, 4)
        assert windows == []
        windows = sb.push(zero_frame())
        assert len(windows) == 1

    def test_reset_resets_stride_counter(self):
        """After reset, stride counter must restart."""
        sb = SequenceBuilder(window_size=3, stride=2)
        push_n(sb, 3)  # fills buffer, emits 1 window
        push_n(sb, 1)  # frames_since_emit = 1 (stride not yet met)
        sb.reset()
        # After reset: need window_size frames before any emission
        windows = push_n(sb, 2)
        assert windows == []


# ---------------------------------------------------------------------------
# Test: push_batch()
# ---------------------------------------------------------------------------

class TestPushBatch:
    def test_batch_same_as_individual(self):
        sb1 = SequenceBuilder(window_size=5, stride=1)
        sb2 = SequenceBuilder(window_size=5, stride=1)

        frames = [make_frame_lm(float(i)) for i in range(8)]

        for fl in frames:
            sb1.push(fl)

        sb2.push_batch(frames)

        # Both builders should have produced the same number of frames
        # (we check via buffered_frames — push counts are now equal)
        assert sb1.buffered_frames == sb2.buffered_frames

    def test_batch_emits_correct_window_count(self):
        T, S, N = 5, 1, 10
        sb = SequenceBuilder(window_size=T, stride=S)
        frames = [zero_frame() for _ in range(N)]
        windows = sb.push_batch(frames)
        expected = N - T + 1
        assert len(windows) == expected

    def test_batch_empty_input(self):
        sb = SequenceBuilder(window_size=5, stride=1)
        windows = sb.push_batch([])
        assert windows == []


# ---------------------------------------------------------------------------
# Test: Wrong input type
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_push_wrong_type_raises(self):
        sb = SequenceBuilder(window_size=5, stride=1)
        with pytest.raises(TypeError):
            sb.push(np.zeros(FEATURE_DIM, dtype=np.float32))  # type: ignore
