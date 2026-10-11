"""
Tests for signbridge.interface.adapter
=======================================
All tests use synthetic ``FrameLandmarks`` — no webcam or MediaPipe needed.

Covered
-------
Slot layout constants
  - Model-side slot boundaries (LH, Pose, RH, Face) match Member 1's spec.
  - Slot counts equal the CV-side landmark counts.
  - All 543 slots are accounted for with no overlaps or gaps.

Output shape and dtype
  - frame_landmarks_to_holistic() always returns (543, 3), float32.
  - sequence_to_holistic() returns (T, 543, 3), float32.

Missing-landmark handling (detection flags)
  - Face slots are always NaN (not captured by CV pipeline).
  - Left hand absent  → slots 468–488 are NaN; right hand and pose unaffected.
  - Right hand absent → slots 522–542 are NaN; left hand and pose unaffected.
  - Pose absent       → slots 489–521 are NaN; hand slots unaffected.
  - All absent        → entire array is NaN (except face, which is always NaN).
  - All present       → no NaN outside the face block.

Zero-fill ambiguity (documented limitation)
  - When detection flag is True but vector contains 0.0, coordinates are
    written as 0.0 (not NaN) — this is correct behaviour; genuine origin
    coordinates must not be masked.
  - When detection flag is False, 0.0 vector values are correctly masked as NaN.

Landmark value mapping (per-landmark correctness)
  - Left hand: vector[0:63] maps to slots 468–488, xyz per landmark.
  - Right hand: vector[63:126] maps to slots 522–542, xyz per landmark.
  - Pose: vector[126:258] maps to slots 489–521, xyz only (visibility dropped).
  - Pose wrist landmark (index 15 = left wrist) value verified end-to-end.
  - Hand wrist landmark (index 0) value verified end-to-end.

Visibility drop
  - The 4th value (visibility) in each pose entry is NOT copied to the output.
  - Only x, y, z appear in pose slots.

sequence_to_holistic
  - Empty sequence raises ValueError.
  - T=1 produces (1, 543, 3).
  - T=N preserves chronological order.
  - Mixed detection across frames works correctly.

Input validation
  - Wrong shape vector raises ValueError.
  - Wrong dtype vector raises ValueError.
  - Non-FrameLandmarks input to frame_landmarks_to_holistic raises TypeError.
"""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.interface.adapter import (
    FACE_SLOT_END,
    FACE_SLOT_START,
    HOLISTIC_SLOTS,
    HOLISTIC_VALUES,
    LH_SLOT_END,
    LH_SLOT_START,
    POSE_SLOT_END,
    POSE_SLOT_START,
    RH_SLOT_END,
    RH_SLOT_START,
    frame_landmarks_to_holistic,
    frame_vector_to_holistic,
    sequence_to_holistic,
)
from signbridge.preprocessing.landmark_extractor import (
    FEATURE_DIM,
    HAND_LANDMARK_COUNT,
    HAND_VALUES_PER_LANDMARK,
    POSE_LANDMARK_COUNT,
    POSE_VALUES_PER_LANDMARK,
    FrameLandmarks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame_lm(
    left_detected: bool = False,
    right_detected: bool = False,
    pose_detected: bool = False,
    fill: float = 0.0,
) -> FrameLandmarks:
    """Return a FrameLandmarks whose vector is entirely ``fill``."""
    vec = np.full(FEATURE_DIM, fill, dtype=np.float32)
    return FrameLandmarks(
        left_hand_detected=left_detected,
        right_hand_detected=right_detected,
        pose_detected=pose_detected,
        vector=vec,
    )


def make_distinct_frame_lm(
    left_detected: bool = True,
    right_detected: bool = True,
    pose_detected: bool = True,
) -> tuple[FrameLandmarks, np.ndarray]:
    """Return a FrameLandmarks with unique, recognisable per-landmark values.

    Each feature element i carries the value ``float(i) * 0.001`` so that
    values can be traced back to their source position in the 258-vector.
    Returns (FrameLandmarks, vector_copy).
    """
    vec = np.arange(FEATURE_DIM, dtype=np.float32) * 0.001
    fl = FrameLandmarks(
        left_hand_detected=left_detected,
        right_hand_detected=right_detected,
        pose_detected=pose_detected,
        vector=vec.copy(),
    )
    return fl, vec


# ---------------------------------------------------------------------------
# Slot layout constants
# ---------------------------------------------------------------------------

class TestSlotConstants:
    def test_face_slots(self):
        assert FACE_SLOT_START == 0
        assert FACE_SLOT_END == 468

    def test_lh_slots(self):
        assert LH_SLOT_START == 468
        assert LH_SLOT_END == 489
        assert LH_SLOT_END - LH_SLOT_START == HAND_LANDMARK_COUNT

    def test_pose_slots(self):
        assert POSE_SLOT_START == 489
        assert POSE_SLOT_END == 522
        assert POSE_SLOT_END - POSE_SLOT_START == POSE_LANDMARK_COUNT

    def test_rh_slots(self):
        assert RH_SLOT_START == 522
        assert RH_SLOT_END == 543
        assert RH_SLOT_END - RH_SLOT_START == HAND_LANDMARK_COUNT

    def test_total_slots(self):
        assert HOLISTIC_SLOTS == 543

    def test_holistic_values(self):
        assert HOLISTIC_VALUES == 3

    def test_no_gaps_or_overlaps(self):
        """All 543 slots must be covered by exactly one segment."""
        segments = (
            set(range(FACE_SLOT_START, FACE_SLOT_END)),
            set(range(LH_SLOT_START, LH_SLOT_END)),
            set(range(POSE_SLOT_START, POSE_SLOT_END)),
            set(range(RH_SLOT_START, RH_SLOT_END)),
        )
        total = set()
        for seg in segments:
            assert total.isdisjoint(seg), "Overlapping slot ranges detected"
            total |= seg
        assert total == set(range(HOLISTIC_SLOTS)), (
            f"Slots covered: {len(total)}, expected: {HOLISTIC_SLOTS}"
        )


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_single_frame_shape(self):
        fl = make_frame_lm()
        out = frame_landmarks_to_holistic(fl)
        assert out.shape == (HOLISTIC_SLOTS, HOLISTIC_VALUES)

    def test_single_frame_dtype(self):
        fl = make_frame_lm()
        out = frame_landmarks_to_holistic(fl)
        assert out.dtype == np.float32

    def test_sequence_shape_t1(self):
        out = sequence_to_holistic([make_frame_lm()])
        assert out.shape == (1, HOLISTIC_SLOTS, HOLISTIC_VALUES)

    def test_sequence_shape_t30(self):
        frames = [make_frame_lm() for _ in range(30)]
        out = sequence_to_holistic(frames)
        assert out.shape == (30, HOLISTIC_SLOTS, HOLISTIC_VALUES)

    def test_sequence_dtype(self):
        out = sequence_to_holistic([make_frame_lm()])
        assert out.dtype == np.float32

    @pytest.mark.parametrize("T", [1, 5, 30, 90])
    def test_sequence_various_lengths(self, T):
        out = sequence_to_holistic([make_frame_lm() for _ in range(T)])
        assert out.shape == (T, HOLISTIC_SLOTS, HOLISTIC_VALUES)


# ---------------------------------------------------------------------------
# Missing-landmark handling via detection flags
# ---------------------------------------------------------------------------

class TestMissingLandmarkHandling:
    def test_face_always_nan(self):
        """Face slots are always NaN regardless of detection flags."""
        fl = make_frame_lm(left_detected=True, right_detected=True, pose_detected=True, fill=1.0)
        out = frame_landmarks_to_holistic(fl)
        face_block = out[FACE_SLOT_START:FACE_SLOT_END]
        assert np.all(np.isnan(face_block)), "All face slots must be NaN"

    def test_all_absent_all_nan(self):
        """No detection → everything should be NaN."""
        fl = make_frame_lm(left_detected=False, right_detected=False, pose_detected=False)
        out = frame_landmarks_to_holistic(fl)
        assert np.all(np.isnan(out)), "All slots should be NaN when nothing detected"

    def test_left_hand_absent_gives_nan(self):
        fl = make_frame_lm(left_detected=False, right_detected=True, pose_detected=True, fill=1.0)
        out = frame_landmarks_to_holistic(fl)
        lh_block = out[LH_SLOT_START:LH_SLOT_END]
        assert np.all(np.isnan(lh_block)), "LH slots must be NaN when left_detected=False"

    def test_right_hand_absent_gives_nan(self):
        fl = make_frame_lm(left_detected=True, right_detected=False, pose_detected=True, fill=1.0)
        out = frame_landmarks_to_holistic(fl)
        rh_block = out[RH_SLOT_START:RH_SLOT_END]
        assert np.all(np.isnan(rh_block)), "RH slots must be NaN when right_detected=False"

    def test_pose_absent_gives_nan(self):
        fl = make_frame_lm(left_detected=True, right_detected=True, pose_detected=False, fill=1.0)
        out = frame_landmarks_to_holistic(fl)
        pose_block = out[POSE_SLOT_START:POSE_SLOT_END]
        assert np.all(np.isnan(pose_block)), "Pose slots must be NaN when pose_detected=False"

    def test_left_absent_does_not_affect_right_or_pose(self):
        fl = make_frame_lm(left_detected=False, right_detected=True, pose_detected=True, fill=1.0)
        out = frame_landmarks_to_holistic(fl)
        assert not np.any(np.isnan(out[RH_SLOT_START:RH_SLOT_END])), "RH must not be NaN"
        assert not np.any(np.isnan(out[POSE_SLOT_START:POSE_SLOT_END])), "Pose must not be NaN"

    def test_right_absent_does_not_affect_left_or_pose(self):
        fl = make_frame_lm(left_detected=True, right_detected=False, pose_detected=True, fill=1.0)
        out = frame_landmarks_to_holistic(fl)
        assert not np.any(np.isnan(out[LH_SLOT_START:LH_SLOT_END])), "LH must not be NaN"
        assert not np.any(np.isnan(out[POSE_SLOT_START:POSE_SLOT_END])), "Pose must not be NaN"

    def test_pose_absent_does_not_affect_hands(self):
        fl = make_frame_lm(left_detected=True, right_detected=True, pose_detected=False, fill=1.0)
        out = frame_landmarks_to_holistic(fl)
        assert not np.any(np.isnan(out[LH_SLOT_START:LH_SLOT_END])), "LH must not be NaN"
        assert not np.any(np.isnan(out[RH_SLOT_START:RH_SLOT_END])), "RH must not be NaN"

    def test_all_detected_no_nan_outside_face(self):
        """When all are detected, only face slots should be NaN."""
        fl = make_frame_lm(left_detected=True, right_detected=True, pose_detected=True, fill=0.5)
        out = frame_landmarks_to_holistic(fl)
        non_face = np.concatenate([
            out[LH_SLOT_START:LH_SLOT_END].ravel(),
            out[POSE_SLOT_START:POSE_SLOT_END].ravel(),
            out[RH_SLOT_START:RH_SLOT_END].ravel(),
        ])
        assert not np.any(np.isnan(non_face)), "No NaN outside face block when all detected"


# ---------------------------------------------------------------------------
# Zero-fill ambiguity (documented limitation)
# ---------------------------------------------------------------------------

class TestZeroFillAmbiguity:
    def test_zero_coords_with_flag_true_are_not_nan(self):
        """If detection=True but vector is 0.0, coordinates must NOT become NaN.

        A landmark genuinely at origin (0, 0, 0) is valid; we must not mask it.
        """
        fl = make_frame_lm(left_detected=True, right_detected=True, pose_detected=True, fill=0.0)
        out = frame_landmarks_to_holistic(fl)
        lh_block = out[LH_SLOT_START:LH_SLOT_END]
        assert not np.any(np.isnan(lh_block)), (
            "Zero-valued coordinates with detected=True must be written as 0.0, not NaN"
        )
        np.testing.assert_array_equal(lh_block, np.zeros_like(lh_block))

    def test_zero_coords_with_flag_false_become_nan(self):
        """If detection=False and vector is 0.0, the result must be NaN.

        The detection flag is authoritative; the zero is a fill artefact.
        """
        fl = make_frame_lm(left_detected=False, right_detected=False, pose_detected=False, fill=0.0)
        out = frame_landmarks_to_holistic(fl)
        assert np.all(np.isnan(out)), (
            "Zero-fill with all detected=False must produce all NaN"
        )


# ---------------------------------------------------------------------------
# Landmark value mapping — per-landmark correctness
# ---------------------------------------------------------------------------

class TestLandmarkValueMapping:

    def test_left_hand_wrist_xyz_mapping(self):
        """Left hand landmark 0 (wrist): vector[0:3] → out[468, 0:3]."""
        fl, vec = make_distinct_frame_lm(left_detected=True, right_detected=False, pose_detected=False)
        out = frame_landmarks_to_holistic(fl)
        # vector[0] = lh_lm0_x, [1] = lh_lm0_y, [2] = lh_lm0_z
        np.testing.assert_array_almost_equal(
            out[LH_SLOT_START, :],           # slot 468
            [vec[0], vec[1], vec[2]],
            err_msg="Left hand wrist (lm 0) xyz must map to slot 468",
        )

    def test_left_hand_pinky_tip_xyz_mapping(self):
        """Left hand landmark 20 (pinky tip): vector[60:63] → out[488, 0:3]."""
        fl, vec = make_distinct_frame_lm(left_detected=True, right_detected=False, pose_detected=False)
        out = frame_landmarks_to_holistic(fl)
        base = 20 * HAND_VALUES_PER_LANDMARK   # = 60
        np.testing.assert_array_almost_equal(
            out[LH_SLOT_START + 20, :],        # slot 488
            [vec[base], vec[base + 1], vec[base + 2]],
            err_msg="Left hand pinky tip (lm 20) xyz must map to slot 488",
        )

    def test_right_hand_wrist_xyz_mapping(self):
        """Right hand landmark 0 (wrist): vector[63:66] → out[522, 0:3]."""
        fl, vec = make_distinct_frame_lm(left_detected=False, right_detected=True, pose_detected=False)
        out = frame_landmarks_to_holistic(fl)
        base = 63   # RIGHT_HAND_SLICE.start
        np.testing.assert_array_almost_equal(
            out[RH_SLOT_START, :],             # slot 522
            [vec[base], vec[base + 1], vec[base + 2]],
            err_msg="Right hand wrist (lm 0) xyz must map to slot 522",
        )

    def test_right_hand_pinky_tip_xyz_mapping(self):
        """Right hand landmark 20 (pinky tip): vector[123:126] → out[542, 0:3]."""
        fl, vec = make_distinct_frame_lm(left_detected=False, right_detected=True, pose_detected=False)
        out = frame_landmarks_to_holistic(fl)
        base = 63 + 20 * HAND_VALUES_PER_LANDMARK   # = 123
        np.testing.assert_array_almost_equal(
            out[RH_SLOT_START + 20, :],             # slot 542
            [vec[base], vec[base + 1], vec[base + 2]],
            err_msg="Right hand pinky tip (lm 20) xyz must map to slot 542",
        )

    def test_pose_nose_xyz_mapping(self):
        """Pose landmark 0 (nose): vector[126:129] → out[489, 0:3]."""
        fl, vec = make_distinct_frame_lm(left_detected=False, right_detected=False, pose_detected=True)
        out = frame_landmarks_to_holistic(fl)
        pose_base = 126   # POSE_SLICE.start
        # Pose lm 0: x=vector[126], y=vector[127], z=vector[128], vis=vector[129] (dropped)
        np.testing.assert_array_almost_equal(
            out[POSE_SLOT_START, :],           # slot 489
            [vec[pose_base], vec[pose_base + 1], vec[pose_base + 2]],
            err_msg="Pose nose (lm 0) xyz must map to slot 489",
        )

    def test_pose_last_landmark_xyz_mapping(self):
        """Pose landmark 32 (right foot index): vector[254:257] → out[521, 0:3]."""
        fl, vec = make_distinct_frame_lm(left_detected=False, right_detected=False, pose_detected=True)
        out = frame_landmarks_to_holistic(fl)
        pose_base = 126
        lm_idx = 32
        base = pose_base + lm_idx * POSE_VALUES_PER_LANDMARK   # 126 + 32*4 = 254
        np.testing.assert_array_almost_equal(
            out[POSE_SLOT_START + 32, :],      # slot 521
            [vec[base], vec[base + 1], vec[base + 2]],
            err_msg="Pose right foot index (lm 32) xyz must map to slot 521",
        )

    def test_all_left_hand_landmarks_round_trip(self):
        """All 21 left-hand landmarks are mapped to the correct slots with correct values."""
        fl, vec = make_distinct_frame_lm(left_detected=True, right_detected=False, pose_detected=False)
        out = frame_landmarks_to_holistic(fl)
        for lm_i in range(HAND_LANDMARK_COUNT):
            src_base = lm_i * HAND_VALUES_PER_LANDMARK          # in vector[0:63]
            dst_slot = LH_SLOT_START + lm_i                     # in out[468:489]
            np.testing.assert_array_almost_equal(
                out[dst_slot, :],
                [vec[src_base], vec[src_base + 1], vec[src_base + 2]],
                err_msg=f"LH landmark {lm_i} mismatch at slot {dst_slot}",
            )

    def test_all_right_hand_landmarks_round_trip(self):
        """All 21 right-hand landmarks are mapped to the correct slots with correct values."""
        fl, vec = make_distinct_frame_lm(left_detected=False, right_detected=True, pose_detected=False)
        out = frame_landmarks_to_holistic(fl)
        rh_start_in_vec = 63
        for lm_i in range(HAND_LANDMARK_COUNT):
            src_base = rh_start_in_vec + lm_i * HAND_VALUES_PER_LANDMARK
            dst_slot = RH_SLOT_START + lm_i
            np.testing.assert_array_almost_equal(
                out[dst_slot, :],
                [vec[src_base], vec[src_base + 1], vec[src_base + 2]],
                err_msg=f"RH landmark {lm_i} mismatch at slot {dst_slot}",
            )

    def test_all_pose_landmarks_round_trip(self):
        """All 33 pose landmarks are mapped to the correct slots (xyz only)."""
        fl, vec = make_distinct_frame_lm(left_detected=False, right_detected=False, pose_detected=True)
        out = frame_landmarks_to_holistic(fl)
        pose_vec_start = 126
        for lm_i in range(POSE_LANDMARK_COUNT):
            src_base = pose_vec_start + lm_i * POSE_VALUES_PER_LANDMARK   # step of 4
            dst_slot = POSE_SLOT_START + lm_i
            np.testing.assert_array_almost_equal(
                out[dst_slot, :],
                [vec[src_base], vec[src_base + 1], vec[src_base + 2]],
                err_msg=f"Pose landmark {lm_i} mismatch at slot {dst_slot}",
            )


# ---------------------------------------------------------------------------
# Visibility drop
# ---------------------------------------------------------------------------

class TestVisibilityDrop:
    def test_pose_visibility_not_copied(self):
        """The 4th value (visibility) per pose landmark must NOT appear in output."""
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        # Write a distinct sentinel to visibility columns of all 33 pose landmarks.
        SENTINEL = 99.0
        for lm_i in range(POSE_LANDMARK_COUNT):
            vis_idx = 126 + lm_i * POSE_VALUES_PER_LANDMARK + 3   # column 3 = visibility
            vec[vis_idx] = SENTINEL

        fl = FrameLandmarks(left_hand_detected=False, right_hand_detected=False,
                            pose_detected=True, vector=vec)
        out = frame_landmarks_to_holistic(fl)

        pose_block = out[POSE_SLOT_START:POSE_SLOT_END]   # (33, 3)
        assert not np.any(pose_block == SENTINEL), (
            "Visibility sentinel value must not appear in output — visibility must be dropped"
        )

    def test_pose_output_has_exactly_3_values_per_landmark(self):
        """Each pose slot in output has exactly 3 values (x, y, z)."""
        fl = make_frame_lm(pose_detected=True, fill=0.5)
        out = frame_landmarks_to_holistic(fl)
        assert out[POSE_SLOT_START:POSE_SLOT_END].shape == (POSE_LANDMARK_COUNT, 3)


# ---------------------------------------------------------------------------
# sequence_to_holistic
# ---------------------------------------------------------------------------

class TestSequenceToHolistic:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            sequence_to_holistic([])

    def test_t1_shape(self):
        out = sequence_to_holistic([make_frame_lm()])
        assert out.shape == (1, HOLISTIC_SLOTS, HOLISTIC_VALUES)

    def test_t30_shape(self):
        out = sequence_to_holistic([make_frame_lm() for _ in range(30)])
        assert out.shape == (30, HOLISTIC_SLOTS, HOLISTIC_VALUES)

    def test_chronological_order(self):
        """Row 0 = oldest frame (value=0.1), row 1 = newest (value=0.2)."""
        vec0 = np.full(FEATURE_DIM, 0.1, dtype=np.float32)
        vec1 = np.full(FEATURE_DIM, 0.2, dtype=np.float32)
        fl0 = FrameLandmarks(left_hand_detected=True, right_hand_detected=False,
                             pose_detected=False, vector=vec0)
        fl1 = FrameLandmarks(left_hand_detected=True, right_hand_detected=False,
                             pose_detected=False, vector=vec1)
        out = sequence_to_holistic([fl0, fl1])
        assert out[0, LH_SLOT_START, 0] == pytest.approx(0.1)
        assert out[1, LH_SLOT_START, 0] == pytest.approx(0.2)

    def test_mixed_detection_across_frames(self):
        """Different detection flags per frame are handled independently."""
        fl_lh_only = make_frame_lm(left_detected=True,  right_detected=False, pose_detected=False, fill=1.0)
        fl_rh_only = make_frame_lm(left_detected=False, right_detected=True,  pose_detected=False, fill=1.0)
        fl_pose_only = make_frame_lm(left_detected=False, right_detected=False, pose_detected=True, fill=1.0)

        out = sequence_to_holistic([fl_lh_only, fl_rh_only, fl_pose_only])
        assert out.shape == (3, HOLISTIC_SLOTS, HOLISTIC_VALUES)

        # Frame 0: LH filled, RH NaN, Pose NaN
        assert not np.any(np.isnan(out[0, LH_SLOT_START:LH_SLOT_END]))
        assert np.all(np.isnan(out[0, RH_SLOT_START:RH_SLOT_END]))
        assert np.all(np.isnan(out[0, POSE_SLOT_START:POSE_SLOT_END]))

        # Frame 1: RH filled, LH NaN, Pose NaN
        assert np.all(np.isnan(out[1, LH_SLOT_START:LH_SLOT_END]))
        assert not np.any(np.isnan(out[1, RH_SLOT_START:RH_SLOT_END]))
        assert np.all(np.isnan(out[1, POSE_SLOT_START:POSE_SLOT_END]))

        # Frame 2: Pose filled, hands NaN
        assert np.all(np.isnan(out[2, LH_SLOT_START:LH_SLOT_END]))
        assert np.all(np.isnan(out[2, RH_SLOT_START:RH_SLOT_END]))
        assert not np.any(np.isnan(out[2, POSE_SLOT_START:POSE_SLOT_END]))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            frame_vector_to_holistic(
                np.zeros(100, dtype=np.float32), True, True, True
            )

    def test_wrong_dtype_raises(self):
        with pytest.raises(ValueError, match="float32"):
            frame_vector_to_holistic(
                np.zeros(FEATURE_DIM, dtype=np.float64), True, True, True
            )

    def test_non_frame_landmarks_raises(self):
        with pytest.raises(TypeError):
            frame_landmarks_to_holistic(np.zeros(FEATURE_DIM, dtype=np.float32))  # type: ignore
