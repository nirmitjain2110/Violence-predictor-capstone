"""
Unified Real-Time Multi-Modal Violence Prediction Pipeline
===========================================================
Orchestrates:
1. Video frame sampling and normalization
2. Audio stream extraction & Mel-spectrogram processing
3. YOLO-based weapon detection with confusion object rejection
4. Multi-modal fusion & predictive probability P(V_{t+Delta t})
5. Real-time progressive trajectory calculation & lead-time estimation (Delta t)
6. Annotated video rendering with HUD bounding boxes and escalation indicators
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
import torch
import warnings

from .models.spatial_temporal import MobileNetV3SpatialEncoder, SpatioTemporalViolencePredictor
from .models.audio_stream import AudioAnomalyEncoder, extract_mel_from_audio, AUDIO_N_MELS, AUDIO_TIME_FRAMES, AUDIO_SR, AUDIO_DURATION
from .models.weapon_stream import WeaponThreatDetector, WeaponThreatEncoder, CONFUSION_CLASSES
from .models.multimodal_fusion import GatedCrossModalFusion, PredictiveEarlyWarningModel
from .models.early_warning import EscalationTrajectoryAnalyzer
from .model_fusion import MultimodalViolenceDetector

warnings.filterwarnings("ignore")

NUM_FRAMES = 16
IMG_SIZE = 112
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


class MultiModalViolencePipeline:
    """
    Complete inference pipeline for real-time video, audio, and weapon analysis.
    """
    def __init__(
        self,
        fusion_weights_path: Optional[str] = "weights/fusion_best.pt",
        weapon_weights_path: Optional[str] = "weights/weapon_best.pt",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trajectory_analyzer = EscalationTrajectoryAnalyzer(critical_threshold=0.65, alert_threshold=0.50)

        # 1. Weapon Detector with Confusion Object Filtering
        self.weapon_detector = WeaponThreatDetector(
            model_path=weapon_weights_path if (weapon_weights_path and Path(weapon_weights_path).exists()) else None,
            conf_thresh=0.35
        )

        # 2. Multimodal Fusion Model
        self.fusion_model = None
        self.is_legacy_model = True

        if fusion_weights_path and Path(fusion_weights_path).exists():
            try:
                # Attempt to load legacy baseline weights if present
                self.fusion_model = MultimodalViolenceDetector(video_backbone_ckpt=None, freeze_video_backbone=False, pretrained=False)
                ckpt = torch.load(fusion_weights_path, map_location="cpu")
                state_dict = ckpt.get("model_state", ckpt)
                self.fusion_model.load_state_dict(state_dict)
                self.fusion_model.to(self.device)
                self.fusion_model.eval()
                self.is_legacy_model = True
                print(f"[MultiModalPipeline] Loaded legacy fusion model from {fusion_weights_path}")
            except Exception as e:
                print(f"[MultiModalPipeline] Note: could not load as legacy model ({e}), initializing PredictiveEarlyWarningModel...")
                self.fusion_model = PredictiveEarlyWarningModel()
                self.fusion_model.to(self.device)
                self.fusion_model.eval()
                self.is_legacy_model = False
        else:
            # Initialize modern edge-optimized architecture
            self.fusion_model = PredictiveEarlyWarningModel()
            self.fusion_model.to(self.device)
            self.fusion_model.eval()
            self.is_legacy_model = False

    def sample_video_frames(
        self,
        video_path: Union[str, Path],
        num_frames: int = NUM_FRAMES,
        size: int = IMG_SIZE
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray], float]:
        """
        Samples uniformly spaced frames for spatio-temporal modeling while keeping all frames for YOLO & HUD.
        """
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        if total <= 0:
            cap.release()
            return None, [], fps

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

        clip = np.stack(clip_frames, axis=0).astype(np.float32) / 255.0
        return clip, all_frames, fps

    def compute_progressive_trajectory(
        self,
        clip_np: np.ndarray,
        mel_t: torch.Tensor,
        has_audio_t: torch.Tensor,
        threat_t: Optional[torch.Tensor] = None
    ) -> List[float]:
        """
        Computes progressive temporal violence probability trajectory across t = 1..16 frames.
        """
        probabilities = []
        clip_norm = (clip_np - MEAN) / STD

        with torch.no_grad():
            if self.is_legacy_model:
                # Progressive sub-window evaluation for 3D/legacy models
                # Replicate prefix up to 16 frames to evaluate progressive accumulation
                for t in range(1, NUM_FRAMES + 1):
                    prefix = clip_norm[:t].copy()
                    if t < NUM_FRAMES:
                        # Pad with the last available frame
                        pad_len = NUM_FRAMES - t
                        padding = np.repeat(prefix[-1:], pad_len, axis=0)
                        window = np.concatenate([prefix, padding], axis=0)
                    else:
                        window = prefix

                    window_t = torch.from_numpy(window).permute(3, 0, 1, 2).float().unsqueeze(0).to(self.device)
                    out = self.fusion_model(window_t, mel_t, has_audio_t)
                    if isinstance(out, dict):
                        prob = out["probabilities"][0, 1].item()
                    else:
                        probs = torch.softmax(out, dim=1)[0]
                        prob = probs[1].item()
                    probabilities.append(float(prob))
            else:
                # Direct prefix evaluation for SpatioTemporalViolencePredictor / PredictiveEarlyWarningModel
                clip_t = torch.from_numpy(clip_norm).permute(3, 0, 1, 2).float().unsqueeze(0).to(self.device)
                res = self.fusion_model(clip_t, mel_t, has_audio_t, threat_t)
                base_prob = res["future_violence_prob"][0].item()
                # Compute progressive trajectory using temporal attention weights
                v_attn = res.get("video_temporal_attn", None)
                if v_attn is not None:
                    weights = v_attn[0].cpu().numpy()
                    cum_weights = np.cumsum(weights)
                    cum_weights = cum_weights / max(1e-5, cum_weights[-1])
                    probabilities = [float(base_prob * (0.35 + 0.65 * w)) for w in cum_weights]
                else:
                    probabilities = [base_prob] * NUM_FRAMES

        return probabilities

    def analyze_video(
        self,
        video_path: Union[str, Path],
        audio_bytes: Optional[bytes] = None,
        weapon_sample_every: int = 4
    ) -> Dict[str, Any]:
        """
        Runs the full end-to-end multi-modal analysis on a video file.
        """
        video_path = Path(video_path)
        clip_np, all_frames, fps = self.sample_video_frames(video_path)

        if clip_np is None:
            return {"error": f"Video unreadable or too short (could not sample {NUM_FRAMES} frames)"}

        # 1. Audio Anomaly Extraction
        if audio_bytes is not None:
            mel, has_audio = extract_mel_from_audio(audio_bytes)
        else:
            mel, has_audio = extract_mel_from_audio(video_path)

        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(self.device)
        has_audio_t = torch.tensor([has_audio], dtype=torch.bool).to(self.device)

        # 2. Weapon Threat Detection & Feature Vector
        dets_per_frame, weapon_counts, threat_vec = self.weapon_detector.detect_video_frames(
            all_frames, sample_every=weapon_sample_every
        )
        threat_t = torch.from_numpy(threat_vec).unsqueeze(0).float().to(self.device)

        # 3. Progressive Trajectory Computation
        trajectory = self.compute_progressive_trajectory(clip_np, mel_t, has_audio_t, threat_t)

        # 4. Multi-Modal Fusion Prediction
        clip_norm = (clip_np - MEAN) / STD
        clip_t = torch.from_numpy(clip_norm).permute(3, 0, 1, 2).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.is_legacy_model:
                out = self.fusion_model(clip_t, mel_t, has_audio_t)
                probs = torch.softmax(out, dim=1)[0]
                current_prob = probs[1].item()
                future_prob = current_prob
                pred_label = "Violence" if current_prob >= 0.50 else "NonViolence"
            else:
                res = self.fusion_model(clip_t, mel_t, has_audio_t, threat_t)
                current_prob = res["probabilities"][0, 1].item()
                future_prob = res["future_violence_prob"][0].item()
                pred_label = "Violence" if current_prob >= 0.50 else "NonViolence"

        # 5. Dynamic Mathematical Early Warning Metrics
        audio_energy = float(np.mean(np.clip(mel, -80, 0)) + 80.0) / 80.0 if has_audio else 0.0
        max_weapon_conf = float(threat_vec[3])

        metrics = self.trajectory_analyzer.compute_trajectory_metrics(
            probabilities=trajectory,
            future_prob=future_prob,
            weapon_threat=max_weapon_conf,
            audio_anomaly=audio_energy
        )

        return {
            "label": pred_label,
            "violence_probability": round(current_prob, 4),
            "future_violence_probability": round(future_prob, 4),
            "audio_used": has_audio,
            "audio_energy_score": round(audio_energy, 4),
            "weapons_detected": weapon_counts,
            "weapon_threat_vector": [round(float(v), 3) for v in threat_vec],
            "metrics": metrics,
            "all_frames": all_frames,
            "fps": fps,
            "detections_per_frame": dets_per_frame
        }

    def annotate_video_frames(
        self,
        all_frames: List[np.ndarray],
        fps: float,
        analysis_result: Dict[str, Any],
        output_path: Union[str, Path]
    ):
        """
        Renders HUD telemetry overlays and bounding boxes on the video.
        """
        if not all_frames:
            return

        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        dets_per_frame = analysis_result.get("detections_per_frame", {})
        metrics = analysis_result.get("metrics", {})
        label = analysis_result.get("label", "NonViolence")
        prob = analysis_result.get("violence_probability", 0.0)
        lead_time = metrics.get("lead_time_seconds", None)
        hazard = metrics.get("hazard_index", 0.0)
        pattern = metrics.get("pattern", "NORMAL")

        # Color mapping: Green (Normal), Orange (Elevated), Red (Critical)
        if hazard >= 0.70 or label == "Violence":
            hud_color = (0, 0, 240)    # Red
        elif hazard >= 0.40:
            hud_color = (0, 165, 255)  # Orange
        else:
            hud_color = (0, 200, 0)    # Green

        last_dets = []
        for i, frame in enumerate(all_frames):
            frame_draw = frame.copy()
            if i in dets_per_frame:
                last_dets = dets_per_frame[i]

            # 1. Draw Weapon Bounding Boxes
            for det in last_dets:
                x1, y1, x2, y2 = map(int, det["box"])
                cls_name = det["class"]
                conf_score = det["conf"]
                box_color = (0, 69, 255) if cls_name.lower() in ("rifle", "pistol", "gun") else (0, 165, 255)
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame_draw, f"{cls_name.upper()} {conf_score:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2
                )

            # 2. Draw Top Telemetry Bar
            cv2.rectangle(frame_draw, (0, 0), (w, 42), (18, 20, 25), -1)
            cv2.putText(
                frame_draw, f"STATUS: {label.upper()} ({prob:.1%})",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2
            )
            cv2.putText(
                frame_draw, f"HAZARD: {hazard:.2f} | PATTERN: {pattern}",
                (w - 380, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1
            )

            # 3. Draw Predictive Lead-Time HUD Alert if Escalation is Active
            if lead_time is not None and lead_time > 0:
                alert_text = f"EARLY WARNING: Escalation in ~{lead_time:.1f}s"
                cv2.rectangle(frame_draw, (10, h - 38), (360, h - 10), (0, 0, 160), -1)
                cv2.putText(
                    frame_draw, alert_text,
                    (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
                )

            writer.write(frame_draw)

        writer.release()
