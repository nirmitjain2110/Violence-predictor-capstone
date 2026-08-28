"""
FastAPI Server: Real-Time Multi-Modal Violence Escalation Predictor & Early Warning API
========================================================================================
Provides REST endpoints for:
- Full video upload and multi-modal early warning trajectory forecasting
- Live camera buffer streaming with predictive lead-time (Delta t) calculations
- Real-time weapon detection with confusion object rejection
- Annotated video rendering with dynamic HUD telemetry
"""

import io
import json
import base64
import subprocess
import shutil
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import MultiModalViolencePipeline, NUM_FRAMES, IMG_SIZE, MEAN, STD
from src.models.audio_stream import extract_mel_from_audio, AUDIO_N_MELS, AUDIO_TIME_FRAMES

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Multi-Modal Violence Escalation Predictor & Early Warning API",
    description="Real-time multi-modal surveillance forecasting with lead-time metric (Delta t)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Multi-Modal Pipeline
pipeline: Optional[MultiModalViolencePipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    print("[API] Initializing Multi-Modal Violence Prediction Pipeline...")
    weights_dir = Path(__file__).parent / "weights"
    fusion_weights = str(weights_dir / "fusion_best.pt") if (weights_dir / "fusion_best.pt").exists() else None
    weapon_weights = str(weights_dir / "weapon_best.pt") if (weights_dir / "weapon_best.pt").exists() else None

    pipeline = MultiModalViolencePipeline(
        fusion_weights_path=fusion_weights,
        weapon_weights_path=weapon_weights
    )
    print("[API] Pipeline ready.")

    if shutil.which("ffmpeg") is None:
        print("[API] Note: ffmpeg not found on PATH. Browser re-encoding will use OpenCV fallback.")


# ---------- Helper Functions ----------
def decode_b64_image(b64_str: str) -> Optional[np.ndarray]:
    try:
        data = base64.b64decode(b64_str.split(",")[-1])
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None
    }


@app.post("/predict_frame")
async def predict_frame(file: UploadFile = File(...)):
    """
    Single webcam/CCTV frame -> rapid weapon detection with confusion object filtering.
    """
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)

    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    dets, threat_vec = pipeline.weapon_detector.detect_frame(frame)
    return {
        "detections": dets,
        "threat_vector": [round(float(v), 3) for v in threat_vec],
        "threat_level": "ELEVATED" if threat_vec[3] > 0.40 else "NORMAL"
    }


@app.post("/predict_clip")
async def predict_clip(
    frames: str = Form(...),
    audio: Optional[UploadFile] = File(None)
):
    """
    Live Camera Stream Buffer (16 rolling frames + optional audio chunk):
    Computes full progressive trajectory, escalation slope, predictive lead time, and hazard index.
    """
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)

    try:
        frame_list = json.loads(frames)
    except Exception:
        return JSONResponse({"error": "Invalid JSON in frames field"}, status_code=400)

    if len(frame_list) != NUM_FRAMES:
        return JSONResponse(
            {"error": f"Expected {NUM_FRAMES} frames, got {len(frame_list)}"}, status_code=400
        )

    decoded_frames = []
    for b64 in frame_list:
        frame = decode_b64_image(b64)
        if frame is None:
            return JSONResponse({"error": "Failed to decode frame"}, status_code=400)
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        decoded_frames.append(resized)

    clip_np = np.stack(decoded_frames, axis=0).astype(np.float32) / 255.0

    # Process Audio
    audio_bytes = None
    if audio is not None:
        audio_bytes = await audio.read()
    mel, has_audio = extract_mel_from_audio(audio_bytes if audio_bytes else b"")

    import torch
    mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(pipeline.device)
    has_audio_t = torch.tensor([has_audio], dtype=torch.bool).to(pipeline.device)

    # Detect weapon threats on latest frame
    latest_frame_bgr = cv2.cvtColor(decoded_frames[-1], cv2.COLOR_RGB2BGR)
    dets, threat_vec = pipeline.weapon_detector.detect_frame(latest_frame_bgr)
    threat_t = torch.from_numpy(threat_vec).unsqueeze(0).float().to(pipeline.device)

    # Compute trajectory
    trajectory = pipeline.compute_progressive_trajectory(clip_np, mel_t, has_audio_t, threat_t)

    # Classification & Future Prediction
    clip_norm = (clip_np - MEAN) / STD
    clip_t = torch.from_numpy(clip_norm).permute(3, 0, 1, 2).float().unsqueeze(0).to(pipeline.device)

    with torch.no_grad():
        if pipeline.is_legacy_model:
            out = pipeline.fusion_model(clip_t, mel_t, has_audio_t)
            probs = torch.softmax(out, dim=1)[0]
            current_prob = probs[1].item()
            future_prob = current_prob
        else:
            res = pipeline.fusion_model(clip_t, mel_t, has_audio_t, threat_t)
            current_prob = res["probabilities"][0, 1].item()
            future_prob = res["future_violence_prob"][0].item()

    audio_energy = float(np.mean(np.clip(mel, -80, 0)) + 80.0) / 80.0 if has_audio else 0.0
    metrics = pipeline.trajectory_analyzer.compute_trajectory_metrics(
        probabilities=trajectory,
        future_prob=future_prob,
        weapon_threat=float(threat_vec[3]),
        audio_anomaly=audio_energy
    )

    label = "Violence" if current_prob >= 0.50 else "NonViolence"

    return {
        "label": label,
        "violence_probability": round(current_prob, 4),
        "future_violence_probability": round(future_prob, 4),
        "audio_used": has_audio,
        "weapons_detected": [d["class"] for d in dets],
        "metrics": metrics
    }


@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    """
    Complete Multi-Modal Video Analysis:
    - 16-frame Spatio-Temporal dynamics extraction
    - Mel-spectrogram audio stream sequence modeling
    - Parallel YOLO weapon detection with confusion object rejection
    - Dynamic trajectory forecasting with lead-time metric (Delta t)
    - Full annotated video generation with telemetry HUD
    """
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)

    tmp_path = Path("uploaded_video_tmp.mp4")
    video_bytes = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(video_bytes)

    analysis = pipeline.analyze_video(tmp_path, audio_bytes=video_bytes, weapon_sample_every=4)

    if "error" in analysis:
        if tmp_path.exists():
            tmp_path.unlink()
        return JSONResponse({"error": analysis["error"]}, status_code=400)

    # Render Annotated Video
    raw_annotated_path = Path("annotated_video_raw.mp4")
    final_annotated_path = Path("annotated_video_final.mp4")

    pipeline.annotate_video_frames(
        all_frames=analysis["all_frames"],
        fps=analysis["fps"],
        analysis_result=analysis,
        output_path=raw_annotated_path
    )

    # Re-encode to H.264 for web playback if ffmpeg is present
    if shutil.which("ffmpeg") is not None:
        try:
            if final_annotated_path.exists():
                final_annotated_path.unlink()
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_annotated_path), "-vcodec", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_annotated_path)],
                check=True, capture_output=True
            )
            raw_annotated_path.unlink()
        except Exception:
            if raw_annotated_path.exists():
                shutil.move(str(raw_annotated_path), str(final_annotated_path))
    else:
        if raw_annotated_path.exists():
            shutil.move(str(raw_annotated_path), str(final_annotated_path))

    if tmp_path.exists():
        tmp_path.unlink()

    return {
        "label": analysis["label"],
        "violence_probability": analysis["violence_probability"],
        "future_violence_probability": analysis["future_violence_probability"],
        "audio_used": analysis["audio_used"],
        "audio_energy_score": analysis["audio_energy_score"],
        "weapons_detected": analysis["weapons_detected"],
        "weapon_threat_vector": analysis["weapon_threat_vector"],
        "metrics": analysis["metrics"],
        "annotated_video_url": "/download_annotated"
    }


@app.get("/download_annotated")
def download_annotated():
    out_file = Path("annotated_video_final.mp4")
    if out_file.exists():
        return FileResponse(str(out_file), media_type="video/mp4")
    raw_file = Path("annotated_video_raw.mp4")
    if raw_file.exists():
        return FileResponse(str(raw_file), media_type="video/mp4")
    return JSONResponse({"error": "Annotated video not found"}, status_code=404)


# Serve Static UI Frontend
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")