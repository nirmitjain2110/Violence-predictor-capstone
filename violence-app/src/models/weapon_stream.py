"""
Module 3: Parallel Multi-Class Weapon Detector & Threat Vector Engine
======================================================================
Implements real-time weapon and threat detection with extended taxonomy (firearms,
knives, blunt objects) and explicit confusion object rejection (smartphones, wallets, bottles)
to drastically minimize false positive alarms.

Mathematical Formulation:
-------------------------
1. Detection Bounding Box & Class Confidence:
   B_i = [x1, y1, x2, y2],  c_i in [0, 1],  k_i in Classes

2. Confusion Mitigation Filter:
   If k_i in ConfusionClasses (phone, wallet, bottle) and IoU(B_i, B_j) > tau_iou:
       Reject ambiguous weapon candidate B_j

3. Spatial Threat Density:
   D_threat = min(1.0, sum_{i in ThreatDetections} Area(B_i) / Area_frame)

4. Structured Multi-Modal Threat Vector:
   w_threat = [
       c_firearm,      # max confidence among firearms
       c_knife,        # max confidence among edged weapons
       c_blunt,        # max confidence among blunt objects
       c_max,          # global maximum threat confidence
       D_threat,       # normalized spatial threat area ratio
       N_threat_norm   # normalized count min(1.0, N_threat / 5.0)
   ] in R^6

5. Weapon Threat Embedding:
   z_w = MLP(w_threat) in R^{d_w}  (e.g., d_w = 64)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# Extended Weapon Taxonomy Categories
FIREARM_CLASSES = {"rifle", "pistol", "gun", "shotgun", "handgun", "firearm", "revolver"}
KNIFE_CLASSES = {"knife", "blade", "machete", "sword", "dagger", "cutter"}
BLUNT_CLASSES = {"bat", "club", "stick", "rod", "crowbar", "hammer", "pipe"}

# Explicit Confusion Objects to Reject False Positives
CONFUSION_CLASSES = {"cell phone", "phone", "mobile phone", "wallet", "bottle", "water bottle", "cup", "book", "umbrella"}


class WeaponThreatEncoder(nn.Module):
    """
    Learned neural projection for the structured threat feature vector.
    Maps w_threat (6D) -> continuous latent space d_w (64D).
    """
    def __init__(self, in_features: int = 6, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, threat_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            threat_vec: (B, 6) or (B, T, 6)
        Returns:
            (B, embed_dim) or (B, T, embed_dim)
        """
        return self.encoder(threat_vec)


class WeaponThreatDetector:
    """
    Detection and feature extraction engine wrapping YOLO with confusion filtering
    and threat dynamics estimation.
    """
    def __init__(self, model_path: Optional[str] = None, conf_thresh: float = 0.35):
        self.conf_thresh = conf_thresh
        self.model = None
        if model_path is not None and YOLO is not None:
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"[WeaponThreatDetector] Warning: could not load YOLO model from {model_path}: {e}")

    def load_model(self, model_path: str):
        if YOLO is not None:
            self.model = YOLO(model_path)

    def filter_confusion_objects(self, raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rejects weapon predictions that strongly overlap with or are superseded by confusion objects.
        """
        filtered = []
        confusion_boxes = [d for d in raw_detections if d["class"].lower() in CONFUSION_CLASSES]
        weapon_candidates = [d for d in raw_detections if d["class"].lower() not in CONFUSION_CLASSES]

        for w_det in weapon_candidates:
            wx1, wy1, wx2, wy2 = w_det["box"]
            w_area = (wx2 - wx1) * (wy2 - wy1)
            is_confused = False

            for c_det in confusion_boxes:
                cx1, cy1, cx2, cy2 = c_det["box"]
                ix1, iy1 = max(wx1, cx1), max(wy1, cy1)
                ix2, iy2 = min(wx2, cx2), min(wy2, cy2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                iou = inter_area / max(1.0, w_area + (cx2 - cx1) * (cy2 - cy1) - inter_area)

                if iou > 0.45 and c_det["conf"] > w_det["conf"]:
                    is_confused = True
                    break

            if not is_confused:
                filtered.append(w_det)

        return filtered

    def detect_frame(self, frame_bgr: np.ndarray, frame_h: int = 1, frame_w: int = 1) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Performs detection on a single frame and computes the 6D mathematical threat vector.
        """
        if self.model is None or frame_bgr is None:
            return [], np.zeros(6, dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        frame_area = float(max(1, h * w))

        try:
            results = self.model.predict(frame_bgr, conf=self.conf_thresh, verbose=False)
            raw_dets = []
            for b in results[0].boxes:
                cls_id = int(b.cls[0])
                cls_name = self.model.names[cls_id] if hasattr(self.model, "names") else str(cls_id)
                conf = float(b.conf[0])
                box = [float(v) for v in b.xyxy[0].cpu().numpy().tolist()]
                raw_dets.append({"class": cls_name, "conf": conf, "box": box})

            filtered_dets = self.filter_confusion_objects(raw_dets)
        except Exception:
            return [], np.zeros(6, dtype=np.float32)

        c_firearm = 0.0
        c_knife = 0.0
        c_blunt = 0.0
        total_threat_area = 0.0
        threat_count = 0

        for d in filtered_dets:
            cls_lower = d["class"].lower()
            conf = d["conf"]
            x1, y1, x2, y2 = d["box"]
            box_area = max(0.0, (x2 - x1) * (y2 - y1))

            is_threat = False
            if any(f in cls_lower for f in FIREARM_CLASSES):
                c_firearm = max(c_firearm, conf)
                is_threat = True
            elif any(k in cls_lower for k in KNIFE_CLASSES):
                c_knife = max(c_knife, conf)
                is_threat = True
            elif any(b in cls_lower for b in BLUNT_CLASSES):
                c_blunt = max(c_blunt, conf)
                is_threat = True
            else:
                # General default threat class (e.g. rifle, knife, pistol)
                c_firearm = max(c_firearm, conf)
                is_threat = True

            if is_threat:
                total_threat_area += box_area
                threat_count += 1

        c_max = max(c_firearm, c_knife, c_blunt)
        d_spatial = min(1.0, total_threat_area / frame_area)
        n_threat_norm = min(1.0, threat_count / 5.0)

        threat_vec = np.array([
            c_firearm,
            c_knife,
            c_blunt,
            c_max,
            d_spatial,
            n_threat_norm
        ], dtype=np.float32)

        return filtered_dets, threat_vec

    def detect_video_frames(
        self,
        frames: List[np.ndarray],
        sample_every: int = 4
    ) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, int], np.ndarray]:
        """
        Samples frames across the video and extracts weapon detections and temporal threat vector.
        """
        detections_per_frame = {}
        summary_counts = {}
        frame_threat_vectors = []

        for i in range(0, len(frames), sample_every):
            dets, t_vec = self.detect_frame(frames[i])
            if dets:
                detections_per_frame[i] = dets
                for d in dets:
                    summary_counts[d["class"]] = summary_counts.get(d["class"], 0) + 1
            frame_threat_vectors.append(t_vec)

        if len(frame_threat_vectors) > 0:
            stacked_threats = np.stack(frame_threat_vectors, axis=0)  # (N_sampled, 6)
            # Max pooling over time for threat presence + average spatial density
            agg_firearm = np.max(stacked_threats[:, 0])
            agg_knife = np.max(stacked_threats[:, 1])
            agg_blunt = np.max(stacked_threats[:, 2])
            agg_max = np.max(stacked_threats[:, 3])
            agg_spatial = np.mean(stacked_threats[:, 4])
            agg_count = np.max(stacked_threats[:, 5])
            video_threat_vec = np.array([agg_firearm, agg_knife, agg_blunt, agg_max, agg_spatial, agg_count], dtype=np.float32)
        else:
            video_threat_vec = np.zeros(6, dtype=np.float32)

        return detections_per_frame, summary_counts, video_threat_vec
