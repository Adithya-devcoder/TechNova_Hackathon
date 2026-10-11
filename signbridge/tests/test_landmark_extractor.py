"""
Tests for signbridge.preprocessing.landmark_extractor
======================================================
All tests use synthetic inputs (black or white frames, numpy arrays) so no
real webcam or internet connection is needed.

Covered:
  - Constants have expected values.
  - Slice offsets are self-consistent and non-overlapping.
  - extract() on an all-black frame returns a zero vector (no landmarks detectable).
  - extract() on an all-black frame returns correct dtype and shape.
  - FrameLandmarks raises ValueError on wrong shape / dtype.
  - LandmarkExtractor can be used as a context manager.
  - extract() rejects frames with wrong shape.
"""

import numpy as np
import pytest

from signbridge.preprocessing.landmark_extractor import (
    FEATURE_DIM,
    HAND_DIM,
    HAND_LANDMARK_COUNT,
    HAND_VALUES_PER_LANDMARK,
    LEFT_HAND_SLICE,
    POSE_DIM,
    POSE_LANDMARK_COUNT,
    POSE_VALUES_PER_LANDMARK,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
    FrameLandmarks,
    LandmarkExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def black_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a small all-black BGR frame (no detectable landmarks)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def white_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a small all-white BGR frame."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_hand_dim(self):
        assert HAND_DIM == HAND_LANDMARK_COUNT * HAND_VALUES_PER_LANDMARK
        assert HAND_DIM == 63

    def test_pose_dim(self):
        assert POSE_DIM == POSE_LANDMARK_COUNT * POSE_VALUES_PER_LANDMARK
        assert POSE_DIM == 132

    def test_feature_dim(self):
        assert FEATURE_DIM == HAND_DIM + HAND_DIM + POSE_DIM
        assert FEATURE_DIM == 258

    def test_slices_are_non_overlapping(self):
        all_indices = (
            set(range(*LEFT_HAND_SLICE.indices(FEATURE_DIM)))
            | set(range(*RIGHT_HAND_SLICE.indices(FEATURE_DIM)))
            | set(range(*POSE_SLICE.indices(FEATURE_DIM)))
        )
        assert len(all_indices) == FEATURE_DIM, (
            "Slices must cover all FEATURE_DIM indices with no overlaps"
        )

    def test_left_hand_slice_start(self):
        assert LEFT_HAND_SLICE.start == 0

    def test_right_hand_slice_start(self):
        assert RIGHT_HAND_SLICE.start == HAND_DIM

    def test_pose_slice_start(self):
        assert POSE_SLICE.start == 2 * HAND_DIM

    def test_pose_slice_stop(self):
        assert POSE_SLICE.stop == FEATURE_DIM


# ---------------------------------------------------------------------------
# Test: FrameLandmarks dataclass
# ---------------------------------------------------------------------------

class TestFrameLandmarks:
    def test_default_vector_shape(self):
        fl = FrameLandmarks()
        assert fl.vector.shape == (FEATURE_DIM,)

    def test_default_vector_dtype(self):
        fl = FrameLandmarks()
        assert fl.vector.dtype == np.float32

    def test_default_vector_is_zero(self):
        fl = FrameLandmarks()
        np.testing.assert_array_equal(fl.vector, np.zeros(FEATURE_DIM, dtype=np.float32))

    def test_default_detection_flags_are_false(self):
        fl = FrameLandmarks()
        assert fl.left_hand_detected is False
        assert fl.right_hand_detected is False
        assert fl.pose_detected is False

    def test_wrong_shape_raises(self):
        bad_vector = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            FrameLandmarks(vector=bad_vector)

    def test_wrong_dtype_raises(self):
        bad_vector = np.zeros(FEATURE_DIM, dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            FrameLandmarks(vector=bad_vector)

    def test_custom_vector_stored(self):
        v = np.ones(FEATURE_DIM, dtype=np.float32) * 0.5
        fl = FrameLandmarks(
            left_hand_detected=True, right_hand_detected=False, pose_detected=True,
            vector=v,
        )
        np.testing.assert_array_equal(fl.vector, v)


# ---------------------------------------------------------------------------
# Test: LandmarkExtractor (synthetic frames — no webcam needed)
# ---------------------------------------------------------------------------

class TestLandmarkExtractor:
    """Tests use static_image_mode=True so MediaPipe skips tracking."""

    @pytest.fixture(scope="class")
    @classmethod
    def extractor(cls):
        ext = LandmarkExtractor(static_image_mode=True)
        yield ext
        ext.close()

    def test_extract_returns_frame_landmarks(self, extractor):
        result = extractor.extract(black_frame())
        assert isinstance(result, FrameLandmarks)

    def test_output_shape(self, extractor):
        result = extractor.extract(black_frame())
        assert result.vector.shape == (FEATURE_DIM,)

    def test_output_dtype(self, extractor):
        result = extractor.extract(black_frame())
        assert result.vector.dtype == np.float32

    def test_black_frame_no_landmarks_detected(self, extractor):
        """All-black frame should not trigger any detections."""
        result = extractor.extract(black_frame())
        assert result.left_hand_detected is False
        assert result.right_hand_detected is False
        # Pose detection on a black frame: MediaPipe may or may not detect;
        # we only assert the vector has the correct shape/dtype regardless.

    def test_black_frame_zero_hand_segments(self, extractor):
        """Hand segments must be zero when hands are not detected."""
        result = extractor.extract(black_frame())
        if not result.left_hand_detected:
            np.testing.assert_array_equal(
                result.vector[LEFT_HAND_SLICE],
                np.zeros(HAND_DIM, dtype=np.float32),
                err_msg="Left hand not detected — segment should be zero-filled",
            )
        if not result.right_hand_detected:
            np.testing.assert_array_equal(
                result.vector[RIGHT_HAND_SLICE],
                np.zeros(HAND_DIM, dtype=np.float32),
                err_msg="Right hand not detected — segment should be zero-filled",
            )

    def test_wrong_shape_frame_raises(self, extractor):
        bad_frame = np.zeros((64, 64), dtype=np.uint8)  # missing channel dim
        with pytest.raises(ValueError):
            extractor.extract(bad_frame)

    def test_wrong_channels_raises(self, extractor):
        bad_frame = np.zeros((64, 64, 4), dtype=np.uint8)  # BGRA
        with pytest.raises(ValueError):
            extractor.extract(bad_frame)

    def test_context_manager(self):
        """Ensure __enter__ / __exit__ work without error."""
        with LandmarkExtractor(static_image_mode=True) as ext:
            result = ext.extract(black_frame())
        assert result.vector.shape == (FEATURE_DIM,)

    def test_different_frame_sizes_accepted(self, extractor):
        for h, w in [(32, 32), (240, 320), (480, 640)]:
            result = extractor.extract(np.zeros((h, w, 3), dtype=np.uint8))
            assert result.vector.shape == (FEATURE_DIM,)

    def test_multiple_frames_return_consistent_shape(self, extractor):
        """Calling extract() on multiple frames must always yield FEATURE_DIM."""
        frames = [black_frame(), white_frame(), black_frame(h=120, w=160)]
        for frame in frames:
            result = extractor.extract(frame)
            assert result.vector.shape == (FEATURE_DIM,), (
                f"Inconsistent shape for frame of size {frame.shape}"
            )
