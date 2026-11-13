"""MobileNetV3 frame encoder with a small bidirectional temporal classifier."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetTemporal(nn.Module):
    def __init__(self, num_classes=50, hidden_size=256, dropout=0.35, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 576
        self.gru = nn.GRU(self.feature_dim, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def encode_frames(self, video):
        """Return unnormalized temporal embeddings for input [B, T, C, H, W]."""
        b, t, c, h, w = video.shape
        x = ((video - self.mean) / self.std).reshape(b * t, c, h, w)
        return self.pool(self.cnn(x)).flatten(1).reshape(b, t, self.feature_dim)

    def forward_features(self, features):
        sequence, _ = self.gru(features)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1).unsqueeze(-1)
        return self.classifier(self.dropout((sequence * weights).sum(dim=1)))

    def forward(self, video):
        return self.forward_features(self.encode_frames(video))
