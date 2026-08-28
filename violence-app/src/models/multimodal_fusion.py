"""
Module 4: Multi-Modal Gated Cross-Attention Fusion & Predictive Early Warning Engine
=====================================================================================
Integrates the Spatio-Temporal Video Dynamics (Module 1), Acoustic Anomaly Stream (Module 2),
and Parallel Weapon Threat Dynamics (Module 3) using a Gated Multimodal Unit (GMU) and
Cross-Attention to forecast violent escalation with a predictive lead-time metric.

Mathematical Formulation:
-------------------------
1. Modality Latent Projection (Dimension Alignment):
   z_v = LayerNorm(W_v * v_temporal) in R^{d_f}   (Visual Dynamics)
   z_a = LayerNorm(W_a * a_embed)    in R^{d_f}   (Acoustic Anomaly)
   z_w = LayerNorm(W_w * w_threat)   in R^{d_f}   (Weapon Threat)
   where d_f = 256

2. Gated Multimodal Unit (GMU) Modality Competition:
   g_v = sigma(W_{gv} * z_v + b_{gv})
   g_a = sigma(W_{ga} * z_a + b_{ga})
   g_w = sigma(W_{gw} * z_w + b_{gw})
   [g_v', g_a', g_w'] = softmax([g_v, g_a, g_w])

3. Cross-Modal Attention (Visual Guided by Audio & Weapon Cues):
   Q = W_Q * z_v
   K = W_K * [z_a; z_w],  V = W_V * [z_a; z_w]
   A_{cross} = softmax(Q * K^T / sqrt(d_k)) * V

4. Fused Feature Representation:
   F_fusion = LayerNorm(g_v' * z_v + g_a' * z_a + g_w' * z_w + A_{cross}) in R^{d_f}

5. Predictive Early Warning Probability Equation:
   P(V_{t+Delta t}) = sigma(W_{pred} * F_fusion + b_{pred})
   Logits = W_{cls} * F_fusion + b_{cls} in R^2
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_temporal import SpatioTemporalViolencePredictor
from .audio_stream import AudioAnomalyEncoder
from .weapon_stream import WeaponThreatEncoder


class GatedCrossModalFusion(nn.Module):
    """
    Fuses tri-modal features (Video, Audio, Weapon) via GMU gating and Cross-Modal Attention.
    """
    def __init__(
        self,
        video_dim: int = 512,
        audio_dim: int = 128,
        weapon_dim: int = 64,
        fused_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.fused_dim = fused_dim

        # Latent Projections
        self.proj_v = nn.Sequential(
            nn.Linear(video_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(audio_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.proj_w = nn.Sequential(
            nn.Linear(weapon_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Modality Gates (GMU)
        self.gate_v = nn.Linear(fused_dim, 1)
        self.gate_a = nn.Linear(fused_dim, 1)
        self.gate_w = nn.Linear(fused_dim, 1)

        # Cross-Attention: Visual (Query) attends to Audio + Weapon (Keys/Values)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Post-fusion normalization & refinement
        self.post_norm = nn.LayerNorm(fused_dim)
        self.refine = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim)
        )

    def forward(
        self,
        v_feat: torch.Tensor,
        a_feat: torch.Tensor,
        w_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            v_feat: (B, video_dim)
            a_feat: (B, audio_dim)
            w_feat: (B, weapon_dim)
        Returns:
            fused_feat: (B, fused_dim)
            aux_info: Dictionary containing gating weights and attention maps
        """
        z_v = self.proj_v(v_feat)  # (B, fused_dim)
        z_a = self.proj_a(a_feat)  # (B, fused_dim)
        z_w = self.proj_w(w_feat)  # (B, fused_dim)

        # Compute dynamic gating weights across modalities
        g_scores = torch.cat([
            self.gate_v(z_v),
            self.gate_a(z_a),
            self.gate_w(z_w)
        ], dim=1)  # (B, 3)
        g_weights = F.softmax(g_scores, dim=1)  # (B, 3)

        g_v = g_weights[:, 0:1]
        g_a = g_weights[:, 1:2]
        g_w = g_weights[:, 2:3]

        # Key-Value context from secondary modalities (Audio & Weapon)
        kv_context = torch.stack([z_a, z_w], dim=1)  # (B, 2, fused_dim)
        q_visual = z_v.unsqueeze(1)                   # (B, 1, fused_dim)

        attn_out, attn_weights = self.cross_attn(
            query=q_visual,
            key=kv_context,
            value=kv_context
        )  # (B, 1, fused_dim), (B, 1, 2)
        attn_out = attn_out.squeeze(1)

        # Gated fusion combination
        gated_blend = g_v * z_v + g_a * z_a + g_w * z_w
        fused = self.post_norm(gated_blend + attn_out)
        fused = fused + self.refine(fused)  # Residual connection

        aux = {
            "modality_gates": g_weights,
            "cross_attn_weights": attn_weights.squeeze(1),
            "z_v": z_v,
            "z_a": z_a,
            "z_w": z_w
        }
        return fused, aux


class PredictiveEarlyWarningModel(nn.Module):
    """
    End-to-End Multi-Modal Predictive Violence & Escalation Network.
    Integrates Modules 1, 2, 3 and 4 with the mathematical Early Warning Predictor:
    P(V_{t + Delta t}) = sigma(W_{pred} * F_{fusion} + b_{pred}).
    """
    def __init__(
        self,
        video_backbone_ckpt: Optional[str] = None,
        freeze_video: bool = False,
        fused_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # Module 1: Spatio-Temporal Video Predictor
        self.video_module = SpatioTemporalViolencePredictor(
            spatial_variant="small",
            spatial_dim=512,
            hidden_dim=256,
            dropout=0.2,
            pretrained=True
        )

        # Module 2: Audio Stream Anomaly Encoder
        self.audio_module = AudioAnomalyEncoder(
            n_mels=64,
            embed_dim=128,
            lstm_hidden=64,
            dropout=0.2
        )

        # Module 3: Threat Vector Neural Encoder (6D -> 64D)
        self.weapon_module = WeaponThreatEncoder(
            in_features=6,
            embed_dim=64
        )

        # Module 4: Gated Cross-Modal Fusion
        self.fusion = GatedCrossModalFusion(
            video_dim=512,
            audio_dim=128,
            weapon_dim=64,
            fused_dim=fused_dim,
            num_heads=4,
            dropout=dropout
        )

        # Predictive Early Warning Head: P(V_{t+Delta t}) = sigma(W * F + b)
        self.predictive_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Logits for P(V_{t+Delta t})
        )

        # Classification Head (Non-Violence vs. Violence)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        if video_backbone_ckpt:
            self.load_video_weights(video_backbone_ckpt, freeze=freeze_video)

    def load_video_weights(self, ckpt_path: str, freeze: bool = False):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model_state", ckpt)
            self.video_module.load_state_dict(state_dict, strict=False)
            if freeze:
                for p in self.video_module.parameters():
                    p.requires_grad = False
            print(f"[PredictiveEarlyWarningModel] Successfully loaded video weights from {ckpt_path}")
        except Exception as e:
            print(f"[PredictiveEarlyWarningModel] Note: could not load video checkpoint {ckpt_path}: {e}")

    def forward(
        self,
        clips: torch.Tensor,
        mel: torch.Tensor,
        has_audio: Optional[torch.Tensor] = None,
        threat_vector: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            clips: (B, C, T, H, W) or (B, T, C, H, W) video frames
            mel: (B, 1, 64, 130) audio Mel-spectrogram
            has_audio: (B,) boolean tensor
            threat_vector: Optional (B, 6) weapon threat vector
        Returns:
            Dictionary containing:
                - 'future_violence_prob': P(V_{t+Delta t}) in [0, 1]
                - 'logits': (B, 2) classification logits
                - 'probabilities': (B, 2) softmax probabilities
                - 'fused_features': (B, fused_dim)
                - 'modality_gates': (B, 3) attention/gating share
        """
        B = clips.size(0)
        device = clips.device

        # Module 1: Extract Video Behavioral Embedding
        temp_feats = self.video_module.extract_frame_features(clips)
        v_feat, v_attn, _ = self.video_module.forward_sequence(temp_feats)

        # Module 2: Extract Audio Anomaly Embedding
        a_feat, a_attn = self.audio_module(mel, has_audio)

        # Module 3: Extract Weapon Threat Embedding
        if threat_vector is None:
            threat_vector = torch.zeros(B, 6, device=device)
        elif threat_vector.device != device:
            threat_vector = threat_vector.to(device)
        w_feat = self.weapon_module(threat_vector)

        # Module 4: Gated Cross-Modal Fusion
        fused, aux = self.fusion(v_feat, a_feat, w_feat)

        # Compute Early Warning Predictive Probability: P(V_{t+Delta t}) = sigma(W * F + b)
        future_pred_logits = self.predictive_head(fused).squeeze(-1)  # (B,)
        future_violence_prob = torch.sigmoid(future_pred_logits)      # (B,)

        # Classification Logits
        logits = self.classifier(fused)  # (B, 2)
        probs = torch.softmax(logits, dim=-1)

        return {
            "future_violence_prob": future_violence_prob,
            "logits": logits,
            "probabilities": probs,
            "fused_features": fused,
            "modality_gates": aux["modality_gates"],
            "video_temporal_attn": v_attn,
            "audio_temporal_attn": a_attn
        }
