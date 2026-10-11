"""
Integration tests for the full CV pipeline (capture → extract → sequence)
==========================================================================
These tests exercise CVPipeline.process_frame() with synthetic BGR frames
so no real webcam is required.

Covered:
  - process_frame() produces windows of the correct shape and dtype.
  - Two-stage integration: extractor feeds sequence builder correctly.
  - output_shape property matches actual window shape.
  - reset() resets the pipeline builder state.
  - Pipeline with stride=window_size produces non-overlapping windows.
  - Passing enough synthetic frames through produces at least one window.
"""

import numpy as np
import pytest

from signbridge.preprocessing.landmark_extractor import FEATURE_DIM
from signbridge.preprocessing.pipeline import CVPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WINDOW_SIZE = 10  # keep small so tests run fast


def black_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_pipeline(**kwargs) -> CVPipeline:
    """Create a CVPipeline without opening the webcam."""
    kw = dict(window_size=WINDOW_SIZE, stride=1)
    kw.update(kwargs)
    return CVPipeline(**kw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineProcessFrame:
    @pytest.fixture(scope="class")
    @classmethod
    def pipeline(cls):
        p = CVPipeline(window_size=WINDOW_SIZE, stride=1)
        yield p
        p.close()

    def test_no_window_before_buffer_full(self, pipeline):
        pipeline.reset()
        for _ in range(WINDOW_SIZE - 1):
            windows = pipeline.process_frame(black_frame())
            assert windows == [], "No windows before buffer is full"

    def test_first_window_on_frame_T(self, pipeline):
        pipeline.reset()
        windows = []
        for _ in range(WINDOW_SIZE):
            windows.extend(pipeline.process_frame(black_frame()))
        assert len(windows) == 1

    def test_window_shape(self, pipeline):
        pipeline.reset()
        windows = []
        for _ in range(WINDOW_SIZE):
            windows.extend(pipeline.process_frame(black_frame()))
        assert windows[0].shape == (WINDOW_SIZE, FEATURE_DIM)

    def test_window_dtype(self, pipeline):
        pipeline.reset()
        windows = []
        for _ in range(WINDOW_SIZE):
            windows.extend(pipeline.process_frame(black_frame()))
        assert windows[0].dtype == np.float32

    def test_output_shape_property(self, pipeline):
        assert pipeline.output_shape == (WINDOW_SIZE, FEATURE_DIM)

    def test_stride_equal_window_non_overlapping(self):
        p = CVPipeline(window_size=WINDOW_SIZE, stride=WINDOW_SIZE)
        # Push 2 * WINDOW_SIZE frames — expect exactly 2 windows
        all_windows = []
        for _ in range(2 * WINDOW_SIZE):
            all_windows.extend(p.process_frame(black_frame()))
        p.close()
        assert len(all_windows) == 2

    def test_reset_clears_pipeline_buffer(self, pipeline):
        pipeline.reset()
        # Push WINDOW_SIZE - 1 frames
        for _ in range(WINDOW_SIZE - 1):
            pipeline.process_frame(black_frame())
        pipeline.reset()
        # After reset, need another WINDOW_SIZE frames
        windows = []
        for _ in range(WINDOW_SIZE - 1):
            windows.extend(pipeline.process_frame(black_frame()))
        assert windows == [], "Buffer should be empty after reset"

    def test_sustained_stream_produces_windows(self, pipeline):
        """100 frames with stride=1 should produce windows after warmup."""
        pipeline.reset()
        all_windows = []
        for _ in range(100):
            all_windows.extend(pipeline.process_frame(black_frame()))
        expected = 100 - WINDOW_SIZE + 1
        assert len(all_windows) == expected

    def test_different_frame_sizes_accepted(self, pipeline):
        pipeline.reset()
        for _ in range(WINDOW_SIZE):
            pipeline.process_frame(np.zeros((120, 160, 3), dtype=np.uint8))
        # If we got here without exception, frame sizes are handled correctly.


class TestPipelineContractCompatibility:
    """Verify that pipeline output is directly compatible with the ML contract."""

    def test_output_matches_contract_feature_dim(self):
        from signbridge.interface.contract import FEATURE_DIM as CONTRACT_FEATURE_DIM
        assert FEATURE_DIM == CONTRACT_FEATURE_DIM

    def test_mock_model_accepts_pipeline_output(self):
        from signbridge.interface.contract import MockModel
        p = CVPipeline(window_size=WINDOW_SIZE, stride=WINDOW_SIZE)
        model = MockModel()
        all_windows = []
        for _ in range(WINDOW_SIZE):
            all_windows.extend(p.process_frame(black_frame()))
        p.close()
        assert len(all_windows) == 1
        result = model.predict(all_windows[0])
        assert result.predicted_sign is not None
        assert 0.0 <= result.confidence <= 1.0
