from pathlib import Path

import torch

from models.mobilenet_baseline import MobileNetV3Baseline


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Trained checkpoint
CHECKPOINT_PATH = (
    BASE_DIR
    / "checkpoints"
    / "mobilenet_baseline_letterbox_best.pth"
)


class ModelLoader:

    def __init__(self):

        self.device = self._get_device()

        self.model = None
        self.classes = None

    def _get_device(self):

        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def load_model(self):

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}"
            )

        print(f"Loading model on {self.device}...")

        # Load checkpoint
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False
        )

        # Retrieve class mapping
        self.classes = checkpoint.get("classes")

        if not self.classes:
            raise ValueError(
                "Class mapping missing from checkpoint."
            )

        if len(self.classes) != 50:
            raise ValueError(
                f"Expected 50 classes, got {len(self.classes)}"
            )

        # Initialize architecture without downloading weights
        self.model = MobileNetV3Baseline(
            num_classes=len(self.classes),
            pretrained=False
        )

        # Load trained parameters
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Move model to selected device
        self.model.to(self.device)

        # Set inference mode
        self.model.eval()

        print("Model loaded successfully!")
        print("Device:", self.device)
        print("Classes:", len(self.classes))
        print("Checkpoint epoch:", checkpoint.get("epoch"))

        return self.model

    def get_model(self):

        if self.model is None:
            self.load_model()

        return self.model


# Shared model instance
model_loader = ModelLoader()


if __name__ == "__main__":

    model = model_loader.load_model()

    # Test with 16 frames
    dummy_input = torch.rand(
        1, 16, 3, 224, 224
    ).to(model_loader.device)

    with torch.no_grad():

        output = model(dummy_input)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_index = probabilities.max(dim=1)

    print("\nINFERENCE TEST SUCCESSFUL")

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print(
        "Predicted sign:",
        model_loader.classes[predicted_index.item()]
    )

    print(
        "Confidence:",
        round(confidence.item() * 100, 2),
        "%"
    )