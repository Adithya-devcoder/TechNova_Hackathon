"""FastAPI adapter for the existing SignBridge BiGRU-CTC model.

Run from the TechNova-ML repository root with:
    python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from models import signbridge_model as sb
from models.gloss_to_english import to_english
from preprocessing.landmarks import HolisticExtractor, draw_landmarks


# Local NLLB-200 model downloaded through Hugging Face.
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
TARGET_LANGUAGES = {
    "ta": "tam_Taml",
    "hi": "hin_Deva",
    "te": "tel_Telu",
}


@lru_cache(maxsize=1)
def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained(
        NLLB_MODEL, local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL, local_files_only=True
    )
    model.eval()
    return tokenizer, model


app = FastAPI(title="SignBridge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.1.2:8080",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://192.168.1.2:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = None
extractor = None
model_bundle = None
worker = None
stop_worker = threading.Event()
frame_lock = threading.Lock()
latest_frame = None
latest_landmarks = None
recording_lock = threading.Lock()
recording = False

live_frames: deque[np.ndarray] = deque(maxlen=60)
recorded_frames: list[np.ndarray] = []
recorded_times: list[float] = []
sessions: list[dict] = []
active_session_id: str | None = None
active_started_at: str | None = None
boot_error: str | None = None


class StopRequest(BaseModel):
    session_id: str


class TranslateRequest(BaseModel):
    text: str
    language: Literal["en", "ta", "hi", "te"] = "en"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prediction_result(landmarks: np.ndarray) -> dict:
    """Run the existing sign model and adapt its output to the frontend API."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Recognition model is not loaded")

    prediction = sb.predict(landmarks, model_bundle)
    gloss = str(prediction.get("text", "")).strip()
    sentence = to_english(gloss) if gloss else ""

    raw_words = prediction.get("words", [])
    words = [
        {
            "text": str(word.get("word", "")),
            "confidence": word.get("confidence"),
            "uncertain": word.get("uncertain", False),
        }
        for word in raw_words
    ]
    confidence_values = [
        float(word["confidence"])
        for word in words
        if word.get("confidence") is not None
    ]
    return {
        "sentence": sentence,
        "words": words,
        "confidence": (
            round(sum(confidence_values) / len(confidence_values), 3)
            if confidence_values else None
        ),
        "latency_ms": prediction.get("latency_ms"),
    }


def _camera_loop() -> None:
    global latest_frame, latest_landmarks, boot_error
    timestamp_ms = 0
    last_clock = time.perf_counter()

    try:
        while not stop_worker.is_set():
            if camera is None or extractor is None:
                time.sleep(0.1)
                continue

            ok, frame = camera.read()
            if not ok:
                boot_error = (
                    "Could not read from webcam. Check camera permission "
                    "or camera index."
                )
                time.sleep(0.15)
                continue

            now = time.perf_counter()
            timestamp_ms += max(1, int((now - last_clock) * 1000))
            last_clock = now
            frame = cv2.flip(frame, 1)
            landmarks = extractor.process(frame, timestamp_ms)

            with frame_lock:
                latest_frame = frame
                latest_landmarks = landmarks

            with recording_lock:
                copied = np.asarray(landmarks, dtype=np.float32).copy()
                live_frames.append(copied)
                if recording:
                    recorded_frames.append(copied)
                    recorded_times.append(float(timestamp_ms))

    except Exception as exc:
        boot_error = f"Camera processing failed: {exc}"


def _mjpeg(landmarks_overlay: bool):
    while not stop_worker.is_set():
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
            points = (
                None if latest_landmarks is None
                else np.asarray(latest_landmarks).copy()
            )

        if frame is None:
            time.sleep(0.05)
            continue

        if landmarks_overlay and points is not None:
            frame = draw_landmarks(frame, points)

        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if ok:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
        time.sleep(0.025)


@app.on_event("startup")
def startup() -> None:
    global camera, extractor, model_bundle, worker, boot_error
    try:
        model_bundle = sb.load_model()
        extractor = HolisticExtractor()
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            boot_error = (
                "Webcam could not be opened. Check camera permissions "
                "or try camera index 1."
            )
            camera.release()
            camera = None
            return

        stop_worker.clear()
        worker = threading.Thread(
            target=_camera_loop, daemon=True, name="signbridge-camera"
        )
        worker.start()
    except Exception as exc:
        boot_error = str(exc)


@app.on_event("shutdown")
def shutdown() -> None:
    stop_worker.set()
    if worker and worker.is_alive():
        worker.join(timeout=2)
    if camera is not None:
        camera.release()
    if extractor is not None:
        extractor.close()


@app.get("/health")
def health():
    if boot_error:
        return {"status": "degraded", "detail": boot_error}
    if model_bundle is None:
        return {"status": "starting"}
    if camera is None or latest_frame is None:
        return {
            "status": "degraded",
            "detail": "Model loaded, but webcam stream is unavailable.",
        }
    return {"status": "ok"}


@app.get("/stream")
def stream(landmarks: int = Query(default=0, ge=0, le=1)):
    if camera is None:
        raise HTTPException(
            status_code=503,
            detail=boot_error or "Webcam unavailable",
        )
    return StreamingResponse(
        _mjpeg(bool(landmarks)),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.post("/recording/start")
def start_recording():
    global recording, recorded_frames, recorded_times
    global active_session_id, active_started_at

    if camera is None:
        raise HTTPException(
            status_code=503,
            detail=boot_error or "Webcam unavailable",
        )

    with recording_lock:
        if recording:
            return {
                "session_id": active_session_id,
                "started_at": active_started_at,
            }
        recorded_frames = []
        recorded_times = []
        recording = True
        active_session_id = str(uuid.uuid4())
        active_started_at = _now_iso()

    return {
        "session_id": active_session_id,
        "started_at": active_started_at,
    }


@app.post("/recording/stop")
def stop_recording(req: StopRequest):
    global recording, recorded_frames, recorded_times
    global active_session_id, active_started_at

    with recording_lock:
        if not recording:
            raise HTTPException(status_code=400, detail="No recording is active")
        if active_session_id and req.session_id != active_session_id:
            raise HTTPException(status_code=400, detail="Session ID does not match")

        recording = False
        frames = [frame.copy() for frame in recorded_frames]
        session_id = active_session_id or req.session_id
        started_at = active_started_at
        recorded_frames = []
        recorded_times = []
        active_session_id = None
        active_started_at = None

    created_at = _now_iso()
    if not frames:
        result = {
            "sentence": "",
            "words": [],
            "confidence": None,
            "latency_ms": 0,
        }
        status = "empty"
    else:
        try:
            result = _prediction_result(np.stack(frames).astype(np.float32))
            status = "success" if result["sentence"] else "empty"
        except HTTPException:
            raise
        except Exception as exc:
            sessions.insert(0, {
                "id": session_id,
                "text": "",
                "created_at": created_at,
                "language": "en",
                "status": "error",
            })
            raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}")

    sessions.insert(0, {
        "id": session_id,
        "text": result["sentence"],
        "created_at": created_at,
        "language": "en",
        "status": status,
    })
    return result


@app.get("/recording/live")
def recording_live():
    with recording_lock:
        frames = [frame.copy() for frame in live_frames]

    if not frames:
        return {
            "sentence": "",
            "words": [],
            "confidence": None,
            "latency_ms": 0,
        }

    try:
        return _prediction_result(np.stack(frames).astype(np.float32))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Live recognition failed: {exc}")


@app.get("/vocabulary")
def vocabulary():
    signs = list(model_bundle["vocab"]) if model_bundle else []
    return {"signs": signs, "sign_language": "ASL"}


@app.get("/sessions")
def get_sessions():
    return sessions


@app.post("/translate")
def translate_text(req: TranslateRequest):
    text = req.text.strip()
    language = req.language

    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")
    if language == "en":
        return {"text": text, "language": "en"}

    target_code = TARGET_LANGUAGES.get(language)
    if target_code is None:
        raise HTTPException(status_code=400, detail="Unsupported language")

    try:
        tokenizer, model = load_translation_model()
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        target_id = tokenizer.convert_tokens_to_ids(target_code)
        if target_id == tokenizer.unk_token_id:
            raise ValueError(f"Invalid target language code: {target_code}")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=128,
            )

        translated = tokenizer.batch_decode(
            output, skip_special_tokens=True
        )[0].strip()
        if not translated:
            raise ValueError("The model returned an empty translation")
        return {"text": translated, "language": language}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {exc}",
        )
