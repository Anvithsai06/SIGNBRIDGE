import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


class MobileNetV3LSTM(nn.Module):

    def __init__(
        self,
        num_classes=50,
        hidden_size=256,
        num_layers=1,
        dropout=0.3,
        pretrained=True
    ):

        super().__init__()

        # Load pretrained MobileNetV3
        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained else None
        )

        mobilenet = mobilenet_v3_small(weights=weights)

        # Extract spatial features
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

        # Freeze early CNN layers
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        # Fine-tune the last two CNN blocks
        for block in list(self.cnn.children())[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        # Temporal feature learning
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
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

        # Normalize using ImageNet statistics
        x = (x - self.mean) / self.std

        # Combine batch and temporal dimensions
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

        # LSTM temporal learning
        lstm_output, _ = self.lstm(features)

        # Use the final time-step representation
        final_features = lstm_output[:, -1, :]

        # Classification
        output = self.dropout(final_features)

        output = self.classifier(output)

        return output


if __name__ == "__main__":

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = MobileNetV3LSTM(
        num_classes=50,
        pretrained=True
    )

    model = model.to(device)

    # Test using a small dummy batch
    dummy_input = torch.randn(
        1, 4, 3, 224, 224
    ).to(device)

    with torch.no_grad():

        output = model(dummy_input)

    print("\nMODEL TEST SUCCESSFUL")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Number of classes:", output.shape[1])

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Trainable parameters:", trainable_parameters)
    print("Total parameters:", total_parameters)