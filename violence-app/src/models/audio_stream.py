"""
Module 2: Audio Stream Analysis & Acoustic Anomaly Encoder
===========================================================
Processes synchronized audio streams to extract Mel-spectrogram representations
and model aggressive acoustic sequences (screams, raised voices, impacts, glass breaks)
using a CNN-BiLSTM network with attention pooling and learned fallback embeddings.

Mathematical Formulation:
-------------------------
1. Log-Mel Spectrogram Extraction:
   y(t) -> STFT(y(t)) -> Mel Filterbank (N_mels=64) -> Log-Power (dB)
   S_mel in R^{1 x N_mels x T_audio}, where N_mels = 64, T_audio = 130

2. Convolutional Spatial-Frequency Reduction:
   z_audio = Conv2D_Block(S_mel) in R^{C' x (N_mels/4) x (T_audio/4)}
   Sequence representation: x_a in R^{T' x (C' * F')}

3. Recurrent Sequence Modeling (Bi-LSTM):
   h_t^a, (h_n^a, c_n^a) = BiLSTM(x_a) in R^{T' x 2*d_h}

4. Acoustic Attention Pooling & Anomaly Embedding:
   alpha_t^a = softmax(w_a^T * tanh(W_a * h_t^a + b_a))
   a_embed = sum_{t=1}^{T'} alpha_t^a * h_t^a in R^{d_a}

5. Missing Modality / Silence Gating:
   a_final = m_audio * a_embed + (1 - m_audio) * e_fallback
   where m_audio in {0, 1} indicates valid acoustic presence.
"""

from typing import Optional, Tuple, Union
import io
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import librosa
except ImportError:
    librosa = None


AUDIO_N_MELS = 64
AUDIO_TIME_FRAMES = 130
AUDIO_SR = 22050
AUDIO_DURATION = 3.0


class AcousticAttentionPooling(nn.Module):
    """
    Computes dynamic attention over audio time-frames to focus on anomalous transients.
    """
    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh()
        )
        self.context = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            pooled: (B, input_dim)
            weights: (B, T)
        """
        scores = self.context(self.projection(x)).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, input_dim)
        return pooled, weights


class AudioAnomalyEncoder(nn.Module):
    """
    CNN + Bi-LSTM Audio Encoder with Acoustic Attention.
    """
    def __init__(
        self,
        n_mels: int = AUDIO_N_MELS,
        embed_dim: int = 128,
        lstm_hidden: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_mels = n_mels
        self.embed_dim = embed_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 16, n_mels/2, T/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 32, n_mels/4, T/4)
        )

        reduced_freq = n_mels // 4
        in_lstm_dim = 32 * reduced_freq

        self.lstm = nn.LSTM(
            input_size=in_lstm_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        bi_lstm_dim = lstm_hidden * 2
        self.attn_pool = AcousticAttentionPooling(bi_lstm_dim, attention_dim=64)

        self.proj = nn.Sequential(
            nn.Linear(bi_lstm_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Learnable fallback vector when audio is absent or silent
        self.missing_audio_embed = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.missing_audio_embed, mean=0.0, std=0.02)

    def forward(
        self,
        mel: torch.Tensor,
        has_audio: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: (B, 1, n_mels, T)
            has_audio: (B,) boolean tensor indicating audio availability
        Returns:
            audio_embed: (B, embed_dim)
            attn_weights: (B, T')
        """
        x = self.conv(mel)  # (B, 32, F', T')
        B, C, F_dim, T_dim = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_dim, C * F_dim)  # (B, T', C*F')

        lstm_out, _ = self.lstm(x)  # (B, T', bi_lstm_dim)
        pooled, attn_weights = self.attn_pool(lstm_out)  # (B, bi_lstm_dim), (B, T')
        audio_embed = self.proj(pooled)  # (B, embed_dim)

        if has_audio is not None:
            fallback = self.missing_audio_embed.unsqueeze(0).expand(B, -1)
            mask = has_audio.view(B, 1).float()
            audio_embed = mask * audio_embed + (1.0 - mask) * fallback

        return audio_embed, attn_weights


def extract_mel_from_audio(
    audio_source: Union[str, Path, bytes, np.ndarray],
    sr: int = AUDIO_SR,
    duration: float = AUDIO_DURATION,
    n_mels: int = AUDIO_N_MELS,
    time_frames: int = AUDIO_TIME_FRAMES
) -> Tuple[np.ndarray, bool]:
    """
    Extracts normalized Log-Mel spectrogram from a file path, bytes, or numpy audio array.
    Returns:
        mel_spectrogram: np.ndarray of shape (n_mels, time_frames)
        has_audio: bool (True if valid non-silent audio was extracted)
    """
    zero_mel = np.zeros((n_mels, time_frames), dtype=np.float32)
    if librosa is None:
        return zero_mel, False

    try:
        if isinstance(audio_source, (str, Path)):
            y, _ = librosa.load(str(audio_source), sr=sr, duration=duration, mono=True)
        elif isinstance(audio_source, (bytes, bytearray)):
            y, _ = librosa.load(io.BytesIO(audio_source), sr=sr, duration=duration, mono=True)
        elif isinstance(audio_source, np.ndarray):
            y = audio_source
            if len(y) > int(sr * duration):
                y = y[:int(sr * duration)]
        else:
            return zero_mel, False

        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Check silence threshold
        energy = np.sum(np.abs(y))
        if energy < 1e-5:
            return zero_mel, False

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        # Pad or crop time frames
        t = mel_db.shape[1]
        if t < time_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, time_frames - t)))
        else:
            mel_db = mel_db[:, :time_frames]

        return mel_db, True
    except Exception:
        return zero_mel, False
