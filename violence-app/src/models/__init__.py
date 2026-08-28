"""
Multi-Modal Violence & Escalation Prediction Package
====================================================
Modules:
- Module 1: Spatio-Temporal Violence Predictor (MobileNetV3 + TAM)
- Module 2: Audio Stream Anomaly Encoder (Mel-Spectrogram + BiLSTM)
- Module 3: Parallel Weapon & Threat Vector Engine (YOLO + Confusion Filtering)
- Module 4: Multi-Modal Gated Cross-Attention Fusion & Early Warning Network
- Module 5: Escalation Trajectory Analyzer & Predictive Lead-Time Engine
"""

from .spatial_temporal import (
    MobileNetV3SpatialEncoder,
    TemporalAttentionModule,
    SpatioTemporalViolencePredictor
)
from .audio_stream import (
    AudioAnomalyEncoder,
    AcousticAttentionPooling,
    extract_mel_from_audio,
    AUDIO_N_MELS,
    AUDIO_TIME_FRAMES,
    AUDIO_SR,
    AUDIO_DURATION
)
from .weapon_stream import (
    WeaponThreatDetector,
    WeaponThreatEncoder,
    FIREARM_CLASSES,
    KNIFE_CLASSES,
    BLUNT_CLASSES,
    CONFUSION_CLASSES
)
from .multimodal_fusion import (
    GatedCrossModalFusion,
    PredictiveEarlyWarningModel
)
from .early_warning import (
    EscalationTrajectoryAnalyzer
)

__all__ = [
    "MobileNetV3SpatialEncoder",
    "TemporalAttentionModule",
    "SpatioTemporalViolencePredictor",
    "AudioAnomalyEncoder",
    "AcousticAttentionPooling",
    "extract_mel_from_audio",
    "AUDIO_N_MELS",
    "AUDIO_TIME_FRAMES",
    "AUDIO_SR",
    "AUDIO_DURATION",
    "WeaponThreatDetector",
    "WeaponThreatEncoder",
    "FIREARM_CLASSES",
    "KNIFE_CLASSES",
    "BLUNT_CLASSES",
    "CONFUSION_CLASSES",
    "GatedCrossModalFusion",
    "PredictiveEarlyWarningModel",
    "EscalationTrajectoryAnalyzer"
]
