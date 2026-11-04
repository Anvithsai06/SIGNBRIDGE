import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


class MobileNetV3Baseline(nn.Module):

    def __init__(self, num_classes=50, pretrained=True):

        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained else None
        )

        mobilenet = mobilenet_v3_small(weights=weights)

        # Spatial feature extractor
        self.cnn = mobilenet.features

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = 576

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        )

        # Freeze early layers
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        # Fine-tune last two CNN blocks
        for block in list(self.cnn.children())[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            self.feature_dim,
            num_classes
        )

    def train(self, mode=True):
        super().train(mode)

        # Keep frozen CNN blocks in evaluation mode.
        for block in list(self.cnn.children())[:-2]:
            block.eval()

        return self

    def forward(self, x):

        # Input: (B, T, C, H, W)

        batch_size, num_frames, channels, height, width = x.shape

        # Normalize frames
        x = (x - self.mean) / self.std

        # Merge batch and temporal dimensions
        x = x.reshape(
            batch_size * num_frames,
            channels,
            height,
            width
        )

        # CNN feature extraction
        features = self.cnn(x)

        features = self.avgpool(features)

        features = torch.flatten(features, 1)

        # Restore temporal dimension
        features = features.reshape(
            batch_size,
            num_frames,
            self.feature_dim
        )

        # Temporal average pooling
        features = features.mean(dim=1)

        # Classification
        features = self.dropout(features)

        output = self.classifier(features)

        return output


if __name__ == "__main__":

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = MobileNetV3Baseline(
        num_classes=50,
        pretrained=True
    ).to(device)

    dummy_input = torch.randn(
        1, 4, 3, 224, 224
    ).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print("\nBASELINE MODEL TEST SUCCESSFUL")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Number of classes:", output.shape[1])
    