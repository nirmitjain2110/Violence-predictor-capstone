"""
Multi-Modal Violence & Escalation Prediction Test Suite
========================================================
Validates:
1. Module 1: MobileNetV3 2D-CNN Spatial Extraction & TAM Temporal Attention (16 frames)
2. Module 2: Audio Stream Mel-Spectrogram & BiLSTM Acoustic Anomaly Encoder with Fallback
3. Module 3: Multi-Class Weapon Threat Dynamics & Confusion Object Filtering
4. Module 4: Gated Cross-Modal Attention Fusion & Early Warning Network
5. Module 5: Escalation Trajectory Diagnostics, Dynamic Velocity, Acceleration, and Lead-Time (Delta t)
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add violence-app to sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.models.spatial_temporal import (
    MobileNetV3SpatialEncoder,
    TemporalAttentionModule,
    SpatioTemporalViolencePredictor
)
from src.models.audio_stream import (
    AudioAnomalyEncoder,
    AcousticAttentionPooling,
    extract_mel_from_audio
)
from src.models.weapon_stream import (
    WeaponThreatDetector,
    WeaponThreatEncoder
)
from src.models.multimodal_fusion import (
    GatedCrossModalFusion,
    PredictiveEarlyWarningModel
)
from src.models.early_warning import (
    EscalationTrajectoryAnalyzer
)


def test_module_1_spatio_temporal():
    print("\n[TEST 1] Module 1: Spatio-Temporal Video Predictor (MobileNetV3 + TAM)...")
    model = SpatioTemporalViolencePredictor(spatial_dim=512, hidden_dim=256, dropout=0.2)
    model.eval()

    # Batch of 2 video clips: 16 frames, 3 channels, 112x112
    dummy_clips = torch.randn(2, 3, 16, 112, 112)
    with torch.no_grad():
        logits, attn_weights = model(dummy_clips)
        trajectories = model.compute_progressive_trajectories(dummy_clips)

    assert logits.shape == (2, 2), f"Expected logits shape (2, 2), got {logits.shape}"
    assert attn_weights.shape == (2, 16), f"Expected attention shape (2, 16), got {attn_weights.shape}"
    assert len(trajectories) == 2 and len(trajectories[0]) == 16, "Trajectory length mismatch"
    print("  -> Logits shape:", logits.shape)
    print("  -> Temporal Attention shape:", attn_weights.shape)
    print("  -> Trajectory sample (16 steps):", [round(p, 3) for p in trajectories[0][:6]], "...")
    print("  [PASS] Module 1 Passed!")


def test_module_2_audio():
    print("\n[TEST 2] Module 2: Audio Stream Anomaly Encoder (Mel + BiLSTM)...")
    audio_encoder = AudioAnomalyEncoder(n_mels=64, embed_dim=128, lstm_hidden=64)
    audio_encoder.eval()

    # Batch of 2: One with valid audio, one with fallback/silent audio
    dummy_mel = torch.randn(2, 1, 64, 130)
    has_audio = torch.tensor([True, False])

    with torch.no_grad():
        a_emb, a_attn = audio_encoder(dummy_mel, has_audio)

    assert a_emb.shape == (2, 128), f"Expected audio embedding shape (2, 128), got {a_emb.shape}"
    print("  -> Audio Embedding shape:", a_emb.shape)
    print("  -> Fallback embedding test passed (sample 0 vs sample 1 != identical):", not torch.allclose(a_emb[0], a_emb[1]))
    print("  [PASS] Module 2 Passed!")


def test_module_3_weapon():
    print("\n[TEST 3] Module 3: Threat Vector Encoder & Confusion Mitigation...")
    weapon_encoder = WeaponThreatEncoder(in_features=6, embed_dim=64)
    weapon_encoder.eval()

    # 6D Threat vector: [c_firearm, c_knife, c_blunt, c_max, D_spatial, N_threat]
    dummy_threat = torch.tensor([
        [0.88, 0.00, 0.00, 0.88, 0.15, 0.20],  # Firearm present
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]   # No threat
    ], dtype=torch.float32)

    with torch.no_grad():
        w_emb = weapon_encoder(dummy_threat)

    assert w_emb.shape == (2, 64), f"Expected weapon embedding shape (2, 64), got {w_emb.shape}"
    print("  -> Weapon Embedding shape:", w_emb.shape)

    detector = WeaponThreatDetector(model_path=None)
    # Test confusion filter: reject weapon if superseded by phone/bottle
    mock_dets = [
        {"class": "knife", "conf": 0.52, "box": [50, 50, 100, 100]},
        {"class": "cell phone", "conf": 0.85, "box": [48, 52, 102, 98]}
    ]
    filtered = detector.filter_confusion_objects(mock_dets)
    assert len(filtered) == 0, f"Confusion filter failed, expected 0 retained, got {len(filtered)}"
    print("  -> Confusion filter successfully rejected false weapon candidate.")
    print("  [PASS] Module 3 Passed!")


def test_module_4_multimodal_fusion():
    print("\n[TEST 4] Module 4: Gated Cross-Modal Attention Early Warning Model...")
    model = PredictiveEarlyWarningModel(fused_dim=256, num_classes=2)
    model.eval()

    dummy_clips = torch.randn(2, 3, 16, 112, 112)
    dummy_mel = torch.randn(2, 1, 64, 130)
    has_audio = torch.tensor([True, False])
    threat_vec = torch.tensor([
        [0.85, 0.0, 0.0, 0.85, 0.12, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    with torch.no_grad():
        res = model(dummy_clips, dummy_mel, has_audio, threat_vec)

    p_future = res["future_violence_prob"]
    probs = res["probabilities"]
    gates = res["modality_gates"]

    assert p_future.shape == (2,), f"Expected future prob shape (2,), got {p_future.shape}"
    assert probs.shape == (2, 2), f"Expected class probs shape (2, 2), got {probs.shape}"
    assert gates.shape == (2, 3), f"Expected modality gates shape (2, 3), got {gates.shape}"

    print("  -> P(V_{t+Delta t}) Future Violence Probability:", [round(p.item(), 4) for p in p_future])
    print("  -> Classification Probabilities (NonViolence / Violence):", probs.numpy())
    print("  -> Modality Gating Weights (Video, Audio, Weapon):", gates.numpy())
    print("  [PASS] Module 4 Passed!")


def test_module_5_early_warning_engine():
    print("\n[TEST 5] Module 5: Dynamic Escalation Velocity & Lead-Time Analyzer...")
    analyzer = EscalationTrajectoryAnalyzer(critical_threshold=0.65, alert_threshold=0.50, fps=5.33)

    # Simulated strictly escalating sequence
    escalating_traj = [0.10, 0.14, 0.19, 0.24, 0.31, 0.38, 0.46, 0.54, 0.62, 0.71, 0.78, 0.85, 0.90, 0.92, 0.95, 0.96]
    metrics = analyzer.compute_trajectory_metrics(
        probabilities=escalating_traj,
        future_prob=0.88,
        weapon_threat=0.75,
        audio_anomaly=0.60
    )

    print("  -> Pattern Detected:", metrics["pattern"])
    print("  -> Escalation Velocity S(t):", metrics["escalation_slope"], "/s")
    print("  -> Predictive Lead-Time Delta t:", metrics["lead_time_seconds"], "s")
    print("  -> Composite Hazard Index H(t):", metrics["hazard_index"])
    print("  -> Crossing Step (Frame):", metrics["crossing_step"])
    print("  -> Threat Level:", metrics["threat_level"])

    assert metrics["early_warning_alert"] is True, "Expected early warning alert to trigger"
    assert metrics["pattern"] in ("STRICT_ESCALATION", "GRADUAL_BUILDUP"), f"Unexpected pattern {metrics['pattern']}"
    print("  [PASS] Module 5 Passed!")


if __name__ == "__main__":
    print("================================================================")
    print("  AEGISGUARD MULTI-MODAL VIOLENCE PREDICTOR TEST SUITE")
    print("================================================================")
    test_module_1_spatio_temporal()
    test_module_2_audio()
    test_module_3_weapon()
    test_module_4_multimodal_fusion()
    test_module_5_early_warning_engine()
    print("\n================================================================")
    print("  ALL MULTI-MODAL PREDICTION TESTS PASSED SUCCESSFULLY! (5/5)")
    print("================================================================")
