import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class SpatioTemporalAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.query = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.key = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, T, H, W = x.shape
        N = T * H * W
        q = self.query(x).view(B, -1, N).permute(0, 2, 1)
        k = self.key(x).view(B, -1, N)
        v = self.value(x).view(B, C, N)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, T, H, W)
        return x + self.gamma * out


class ViolenceDetector(nn.Module):
    def __init__(self, num_classes=2, freeze_until_layer=3, pretrained=False):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = r2plus1d_18(weights=weights)

        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        layers = [self.stem, self.layer1, self.layer2, self.layer3]
        for i, layer in enumerate(layers):
            if i < freeze_until_layer:
                for p in layer.parameters():
                    p.requires_grad = False

        self.attention = SpatioTemporalAttention(channels=512)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
