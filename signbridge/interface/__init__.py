"""
signbridge.interface
=====================
Data contract between Member 2 (Computer Vision) and Member 1 (AI/ML Model).

⚠ **PROPOSED CONTRACT — NOT YET APPROVED**

This module is documentation-only.  It defines:
  - the exact tensor format the CV pipeline produces,
  - the function signature Member 1's model must implement,
  - the output format the frontend (Member 3) will consume.

Member 1 **must review and approve** this contract before integrating the model.
Update the ``STATUS`` constant below once agreed.

Nothing in this file performs landmark extraction or ML inference.
"""
from signbridge.interface.adapter import (
    frame_landmarks_to_holistic,
    sequence_to_holistic,
    frame_vector_to_holistic,
)

__all__ = [
    "frame_landmarks_to_holistic",
    "sequence_to_holistic",
    "frame_vector_to_holistic",
]
