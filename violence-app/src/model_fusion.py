import torch
import torch.nn as nn
from src.model import ViolenceDetector


class AudioEncoder(nn.Module):
    def __init__(self, n_mels=64, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        reduced_freq = n_mels // 4
        self.lstm = nn.LSTM(input_size=32 * reduced_freq, hidden_size=embed_dim,
                             batch_first=True, bidirectional=False)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.conv(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class MultimodalViolenceDetector(nn.Module):
    def __init__(self, video_backbone_ckpt=None, freeze_video_backbone=True,
                 audio_embed_dim=128, num_classes=2, pretrained=False):
        super().__init__()

        video_model = ViolenceDetector(num_classes=num_classes, freeze_until_layer=3,
                                        pretrained=pretrained)
        if video_backbone_ckpt is not None:
            ckpt = torch.load(video_backbone_ckpt, map_location="cpu")
            video_model.load_state_dict(ckpt["model_state"])

        self.stem, self.layer1, self.layer2 = video_model.stem, video_model.layer1, video_model.layer2
        self.layer3, self.layer4 = video_model.layer3, video_model.layer4
        self.attention = video_model.attention
        self.pool = video_model.pool

        if freeze_video_backbone:
            for p in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4, self.attention]:
                for param in p.parameters():
                    param.requires_grad = False

        self.audio_encoder = AudioEncoder(embed_dim=audio_embed_dim)
        self.missing_audio_embed = nn.Parameter(torch.zeros(audio_embed_dim))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 + audio_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, clips, mel, has_audio):
        x = self.stem(clips)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.attention(x)
        video_embed = self.pool(x).flatten(1)

        audio_embed = self.audio_encoder(mel)
        fallback = self.missing_audio_embed.unsqueeze(0).expand(audio_embed.size(0), -1)
        has_audio = has_audio.unsqueeze(1)
        audio_embed = torch.where(has_audio, audio_embed, fallback)

        fused = torch.cat([video_embed, audio_embed], dim=1)
        return self.classifier(fused)
