import io
import base64
import subprocess
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import librosa
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from src.model_fusion import MultimodalViolenceDetector

warnings.filterwarnings("ignore")

# ---------- config ----------
NUM_FRAMES = 16
IMG_SIZE = 112
AUDIO_N_MELS = 64
AUDIO_TIME_FRAMES = 130
AUDIO_SR = 22050
AUDIO_DURATION = 3.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

device = torch.device("cpu")  # free-tier hosting: CPU only

app = FastAPI(title="Violence & Weapon Detection API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- load models once at startup ----------
print("Loading fusion model...")
fusion_model = MultimodalViolenceDetector(video_backbone_ckpt=None, freeze_video_backbone=False)
ckpt = torch.load("weights/fusion_best.pt", map_location="cpu")
fusion_model.load_state_dict(ckpt["model_state"])
fusion_model.eval()
print("Fusion model loaded.")

print("Loading weapon model...")
weapon_model = YOLO("weights/weapon_best.pt")
print("Weapon model loaded.")

if shutil.which("ffmpeg") is None:
    print("WARNING: ffmpeg not found on PATH. Video re-encoding will fail on upload. "
          "Install it locally with 'brew install ffmpeg' (macOS) or 'apt install ffmpeg' (Linux). "
          "The Docker image used for deployment already includes it.")


# ---------- helpers ----------
def decode_b64_image(b64_str):
    data = base64.b64decode(b64_str.split(",")[-1])
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def frames_to_clip_tensor(frames_rgb_112):
    """frames_rgb_112: list of NUM_FRAMES numpy arrays, RGB, already resized to IMG_SIZE."""
    clip = np.stack(frames_rgb_112, axis=0).astype(np.float32) / 255.0
    clip = (clip - MEAN) / STD
    clip_t = torch.from_numpy(clip).permute(3, 0, 1, 2).float().unsqueeze(0)
    return clip_t


def mel_from_audio_bytes(audio_bytes):
    try:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=AUDIO_SR, duration=AUDIO_DURATION, mono=True)
    except Exception:
        return np.zeros((AUDIO_N_MELS, AUDIO_TIME_FRAMES), dtype=np.float32), False
    target_len = int(AUDIO_SR * AUDIO_DURATION)
    y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]
    if np.abs(y).sum() < 1e-6:
        return np.zeros((AUDIO_N_MELS, AUDIO_TIME_FRAMES), dtype=np.float32), False
    mel = librosa.feature.melspectrogram(y=y, sr=AUDIO_SR, n_mels=AUDIO_N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    t = mel_db.shape[1]
    if t < AUDIO_TIME_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, AUDIO_TIME_FRAMES - t)))
    else:
        mel_db = mel_db[:, :AUDIO_TIME_FRAMES]
    return mel_db, True


def sample_frames_from_video(video_path, num_frames=NUM_FRAMES, size=IMG_SIZE):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if total <= 0:
        cap.release()
        return None, None, fps
    idxs = set(int(i * total / num_frames) for i in range(num_frames))
    clip_frames, all_frames = [], []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        if i in idxs:
            resized = cv2.resize(frame, (size, size))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            clip_frames.append(resized)
    cap.release()
    if len(clip_frames) != num_frames:
        return None, all_frames, fps
    return clip_frames, all_frames, fps


def run_weapon_detection(frame_bgr, conf=0.4):
    results = weapon_model.predict(frame_bgr, conf=conf, verbose=False)
    boxes = results[0].boxes
    dets = []
    for b in boxes:
        cls_name = weapon_model.names[int(b.cls[0])]
        dets.append({
            "class": cls_name,
            "conf": float(b.conf[0]),
            "box": [float(v) for v in b.xyxy[0].cpu().numpy().tolist()],
        })
    return dets


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_frame")
async def predict_frame(file: UploadFile = File(...)):
    """Single webcam frame -> weapon boxes only (fast, safe to call frequently)."""
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "could not decode image"}, status_code=400)
    dets = run_weapon_detection(frame)
    return {"detections": dets}


@app.post("/predict_clip")
async def predict_clip(frames: str = Form(...), audio: UploadFile = File(None)):
    """
    frames: JSON string list of base64 JPEGs, length NUM_FRAMES, already ~IMG_SIZE from client
            (client should send exactly NUM_FRAMES frames spaced across its rolling buffer)
    audio: optional short audio blob (webm/ogg/wav) recorded alongside the frames
    """
    import json
    frame_list = json.loads(frames)
    if len(frame_list) != NUM_FRAMES:
        return JSONResponse(
            {"error": f"expected {NUM_FRAMES} frames, got {len(frame_list)}"}, status_code=400
        )

    decoded = []
    for b64 in frame_list:
        frame = decode_b64_image(b64)
        if frame is None:
            return JSONResponse({"error": "bad frame data"}, status_code=400)
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        decoded.append(resized)

    clip_t = frames_to_clip_tensor(decoded)

    has_audio = False
    mel = np.zeros((AUDIO_N_MELS, AUDIO_TIME_FRAMES), dtype=np.float32)
    if audio is not None:
        audio_bytes = await audio.read()
        mel, has_audio = mel_from_audio_bytes(audio_bytes)

    mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
    has_audio_t = torch.tensor([has_audio], dtype=torch.bool)

    with torch.no_grad():
        out = fusion_model(clip_t, mel_t, has_audio_t)
        probs = torch.softmax(out, dim=1)[0]

    label = "Violence" if probs[1] > probs[0] else "NonViolence"
    return {
        "label": label,
        "violence_prob": round(probs[1].item(), 4),
        "audio_used": has_audio,
    }


@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    """Full video upload -> violence label + weapon detections + annotated video."""
    tmp_path = Path("uploaded_tmp.mp4")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    clip_frames, all_frames, fps = sample_frames_from_video(tmp_path)
    if clip_frames is None:
        return JSONResponse({"error": "video too short or unreadable"}, status_code=400)

    clip_t = frames_to_clip_tensor(clip_frames)

    # try to pull audio directly from the uploaded video file via librosa
    mel, has_audio = mel_from_audio_bytes(tmp_path.read_bytes())
    mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
    has_audio_t = torch.tensor([has_audio], dtype=torch.bool)

    with torch.no_grad():
        out = fusion_model(clip_t, mel_t, has_audio_t)
        probs = torch.softmax(out, dim=1)[0]
    label = "Violence" if probs[1] > probs[0] else "NonViolence"
    violence_prob = probs[1].item()

    # weapon detection every 5th frame + annotate output video
    detections_per_frame, weapon_summary = {}, {}
    for i in range(0, len(all_frames), 5):
        dets = run_weapon_detection(all_frames[i])
        if dets:
            detections_per_frame[i] = dets
            for d in dets:
                weapon_summary[d["class"]] = weapon_summary.get(d["class"], 0) + 1

    h, w = all_frames[0].shape[:2]
    raw_path = Path("annotated_output_raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    label_color = (0, 0, 255) if label == "Violence" else (0, 200, 0)
    label_text = f"{label} ({violence_prob:.2f})"
    last_dets = []
    for i, frame in enumerate(all_frames):
        frame = frame.copy()
        if i in detections_per_frame:
            last_dets = detections_per_frame[i]
        for det in last_dets:
            x1, y1, x2, y2 = map(int, det["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"{det['class']} {det['conf']:.2f}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
        writer.write(frame)
    writer.release()

    # OpenCV's mp4v codec isn't playable in most browsers — re-encode to H.264 with ffmpeg
    out_path = Path("annotated_output.mp4")
    if out_path.exists():
        out_path.unlink()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_path), "-vcodec", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
        check=True, capture_output=True,
    )
    raw_path.unlink()

    return {
        "label": label,
        "violence_prob": round(violence_prob, 4),
        "audio_used": has_audio,
        "weapons_detected": weapon_summary,
        "annotated_video_url": "/download_annotated",
    }


@app.get("/download_annotated")
def download_annotated():
    return FileResponse("annotated_output.mp4", media_type="video/mp4")


# serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")