"""
Module 1: Lightweight Spatio-Temporal Violence Predictor
=========================================================
Implements real-time spatial feature extraction via lightweight 2D-CNN (MobileNetV3 / ResNet18)
combined with a Temporal Attention Module (TAM) and recurrent sequence modeling over a 16-frame window.

Mathematical Formulation:
-------------------------
1. Spatial Feature Extraction (per-frame):
   f_t = Backbone(I_t) in R^{d_s},  where I_t in R^{3 x H x W}, t = 1, ..., T

2. Recurrent Temporal Modeling:
   h_t = GRU(f_t, h_{t-1}) in R^{d_h}

3. Temporal Attention Mechanism (TAM):
   score_t = w_a^T * tanh(W_h * h_t + b_h)
   alpha_t = exp(score_t) / sum_{k=1}^T exp(score_k)
   v_temporal = sum_{t=1}^T alpha_t * h_t in R^{d_v}

4. Partial Trajectory Evaluation (Real-Time Streaming):
   v_{1:t} = TAM(h_{1:t}) for progressive prefix windows t in [1, T]
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
    from torchvision.models import (
        mobilenet_v3_small, MobileNet_V3_Small_Weights,
        mobilenet_v3_large, MobileNet_V3_Large_Weights,
        resnet18, ResNet18_Weights
    )
except ImportError:
    models = None


class TemporalAttentionModule(nn.Module):
    """
    Temporal Attention Module (TAM)
    Computes a normalized attention score over temporal frames to dynamically
    weight key moments of violent escalation or physical struggle.
    """
    def __init__(self, hidden_dim: int = 256, attention_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, sequence_features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_features: Tensor of shape (B, T, hidden_dim)
            mask: Optional boolean mask of shape (B, T) where True indicates valid frame
        Returns:
            context: (B, hidden_dim) attention-weighted summary vector
            attn_weights: (B, T) normalized temporal attention weights
        """
        # (B, T, attention_dim)
        projected = self.projection(sequence_features)
        # (B, T, 1) -> (B, T)
        scores = self.context_vector(projected).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (B, T)
        # (B, 1, T) @ (B, T, hidden_dim) -> (B, 1, hidden_dim) -> (B, hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), sequence_features).squeeze(1)
        return context, attn_weights


class MobileNetV3SpatialEncoder(nn.Module):
    """
    Lightweight 2D-CNN Spatial Encoder using MobileNetV3.
    Extracts high-level spatial representations per frame with minimal FLOPs,
    making it ideal for real-time edge and edge-cloud inference.
    """
    def __init__(self, variant: str = "small", pretrained: bool = True, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        if models is None:
            # Fallback mock backbone if torchvision is unavailable
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.proj = nn.Linear(32, out_dim)
            return

        if variant == "small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            mb = mobilenet_v3_small(weights=weights)
            in_features = mb.classifier[0].in_features
            mb.classifier = nn.Identity()
            self.backbone = mb
        elif variant == "large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            mb = mobilenet_v3_large(weights=weights)
            in_features = mb.classifier[0].in_features
            mb.classifier = nn.Identity()
            self.backbone = mb
        elif variant == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            r18 = resnet18(weights=weights)
            in_features = r18.fc.in_features
            r18.fc = nn.Identity()
            self.backbone = r18
        else:
            raise ValueError(f"Unknown backbone variant: {variant}")

        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, 3, H, W) or (B, 3, H, W)
        Returns:
            (B*T, out_dim) spatial feature vector
        """
        feats = self.backbone(x)
        if isinstance(feats, torch.Tensor) and feats.dim() > 2:
            feats = feats.flatten(1)
        return self.proj(feats)


class SpatioTemporalViolencePredictor(nn.Module):
    """
    Module 1 Complete Engine:
    Combines 2D CNN spatial extraction + Bi-GRU temporal sequence modeling + TAM.
    """
    def __init__(
        self,
        spatial_variant: str = "small",
        spatial_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pretrained: bool = True,
        num_classes: int = 2
    ):
        super().__init__()
        self.spatial_encoder = MobileNetV3SpatialEncoder(
            variant=spatial_variant,
            pretrained=pretrained,
            out_dim=spatial_dim
        )
        self.gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        gru_out_dim = hidden_dim * 2  # Bidirectional

        self.temporal_attention = TemporalAttentionModule(
            hidden_dim=gru_out_dim,
            attention_dim=128
        )
        self.behavior_proj = nn.Sequential(
            nn.Linear(gru_out_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(512, num_classes)

    def extract_frame_features(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clips: (B, C, T, H, W) or (B, T, C, H, W)
        Returns:
            (B, T, spatial_dim)
        """
        if clips.dim() == 5:
            # Check whether format is (B, C, T, H, W) or (B, T, C, H, W)
            if clips.size(1) in (1, 3) and clips.size(2) != 3:
                # (B, C, T, H, W) -> permute to (B, T, C, H, W)
                clips = clips.permute(0, 2, 1, 3, 4).contiguous()
            B, T, C, H, W = clips.shape
            frames = clips.view(B * T, C, H, W)
            feats = self.spatial_encoder(frames)
            return feats.view(B, T, -1)
        else:
            raise ValueError(f"Expected 5D tensor (B, C, T, H, W), got shape {clips.shape}")

    def forward_sequence(self, temporal_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            temporal_feats: (B, T, spatial_dim)
        Returns:
            behavior_embedding: (B, 512)
            attn_weights: (B, T)
            gru_outputs: (B, T, gru_out_dim)
        """
        gru_out, _ = self.gru(temporal_feats)  # (B, T, gru_out_dim)
        context, attn_weights = self.temporal_attention(gru_out)
        behavior_emb = self.behavior_proj(context)
        return behavior_emb, attn_weights, gru_out

    def forward(self, clips: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for standalone spatio-temporal detection.
        """
        temporal_feats = self.extract_frame_features(clips)
        behavior_emb, attn_weights, _ = self.forward_sequence(temporal_feats)
        logits = self.classifier(behavior_emb)
        return logits, attn_weights

    def compute_progressive_trajectories(self, clips: torch.Tensor) -> List[List[float]]:
        """
        Real-Time Prefix Inference:
        Evaluates the model over progressive prefixes t in [1, ..., T]
        to produce temporal violence probability trajectories.
        """
        self.eval()
        with torch.no_grad():
            temporal_feats = self.extract_frame_features(clips)  # (B, T, spatial_dim)
            B, T, _ = temporal_feats.shape
            trajectories = []
            for b in range(B):
                b_traj = []
                for t in range(1, T + 1):
                    prefix = temporal_feats[b:b+1, :t, :]
                    gru_out, _ = self.gru(prefix)
                    context, _ = self.temporal_attention(gru_out)
                    b_emb = self.behavior_proj(context)
                    logits = self.classifier(b_emb)
                    prob = torch.softmax(logits, dim=-1)[0, 1].item()
                    b_traj.append(prob)
                trajectories.append(b_traj)
        return trajectories
