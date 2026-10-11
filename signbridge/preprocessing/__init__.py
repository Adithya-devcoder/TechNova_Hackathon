"""
signbridge.preprocessing
========================
Webcam capture, MediaPipe landmark extraction, and landmark-sequence building.

Public API
----------
WebcamCapture       — context-managed webcam frame reader  (capture.py)
LandmarkExtractor   — per-frame MediaPipe Holistic wrapper (landmark_extractor.py)
SequenceBuilder     — accumulates frames into NumPy windows (sequence_builder.py)
CVPipeline          — orchestrates all three stages         (pipeline.py)
"""
from signbridge.preprocessing.capture import WebcamCapture
from signbridge.preprocessing.landmark_extractor import LandmarkExtractor, FrameLandmarks
from signbridge.preprocessing.sequence_builder import SequenceBuilder
from signbridge.preprocessing.pipeline import CVPipeline

__all__ = [
    "WebcamCapture",
    "LandmarkExtractor",
    "FrameLandmarks",
    "SequenceBuilder",
    "CVPipeline",
]
