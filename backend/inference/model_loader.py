from pathlib import Path

import torch

from models.mobilenet_baseline import MobileNetV3Baseline
from models.mobilenet_temporal import MobileNetTemporal


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Trained checkpoint
FINAL_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "signbridge_final.pth"
LEGACY_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "mobilenet_baseline_letterbox_best.pth"


class ModelLoader:

    def __init__(self):

        self.device = self._get_device()

        self.model = None
        self.classes = None
        self.preprocessing = None

    def _get_device(self):

        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def load_model(self):

        checkpoint_path = FINAL_CHECKPOINT_PATH if FINAL_CHECKPOINT_PATH.exists() else LEGACY_CHECKPOINT_PATH
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        print(f"Loading model on {self.device}...")

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
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

        architecture = checkpoint.get("architecture", "MobileNetV3Baseline")
        if architecture == "MobileNetTemporal":
            self.model = MobileNetTemporal(num_classes=len(self.classes), pretrained=False)
        elif architecture == "MobileNetV3Baseline":
            self.model = MobileNetV3Baseline(num_classes=len(self.classes), pretrained=False)
        else:
            raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
        self.preprocessing = {
            "num_frames": checkpoint.get("num_frames", 16),
            "image_size": checkpoint.get("image_size", 224),
            **checkpoint.get("preprocessing", {}),
        }
        self.preprocessing["num_frames"] = int(self.preprocessing["num_frames"])
        self.preprocessing["image_size"] = int(self.preprocessing["image_size"])

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
        print("Checkpoint:", checkpoint_path.name)
        print("Checkpoint epoch:", checkpoint.get("epoch"))

        return self.model

    def get_model(self):

        if self.model is None:
            self.load_model()

        return self.model

    def get_inference_config(self):
        """Return the preprocessing contract stored with the loaded checkpoint."""
        self.get_model()
        return {
            "frame_count": self.preprocessing["num_frames"],
            "num_classes": len(self.classes),
            "input_size": self.preprocessing["image_size"],
            "architecture": self.model.__class__.__name__,
        }


# Shared model instance
model_loader = ModelLoader()


if __name__ == "__main__":

    model = model_loader.load_model()

    # Test with the frame count and input size saved in the checkpoint.
    dummy_input = torch.rand(
        1,
        model_loader.preprocessing["num_frames"],
        3,
        model_loader.preprocessing["image_size"],
        model_loader.preprocessing["image_size"],
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
