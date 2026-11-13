import numpy as np
import torch
import cv2

from backend.inference.model_loader import model_loader


class InferenceService:

    def __init__(self):

        self.model_loader = model_loader
        self.model = self.model_loader.get_model()
        self.device = self.model_loader.device
        self.classes = self.model_loader.classes

    def preprocess_frames(self, frames):

        if len(frames) != 16:
            raise ValueError(
                f"Expected 16 frames, received {len(frames)}"
            )

        processed_frames = []

        for frame in frames:

            # Convert input to NumPy array
            frame = np.asarray(frame)

            if frame.dtype != np.uint8:
                raise ValueError("Frames must contain uint8 pixel values.")

            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    "Each frame must have shape (H, W, 3)."
                )

            # Browser frames are expected in RGB format.
            height, width = frame.shape[:2]

            # Preserve aspect ratio
            scale = min(
                224 / width,
                224 / height
            )

            new_width = int(round(width * scale))
            new_height = int(round(height * scale))

            resized = cv2.resize(
                frame,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

            # Letterbox padding
            canvas = np.full(
                (224, 224, 3),
                114,
                dtype=np.uint8
            )

            x_offset = (224 - new_width) // 2
            y_offset = (224 - new_height) // 2

            canvas[
                y_offset:y_offset + new_height,
                x_offset:x_offset + new_width
            ] = resized

            # Convert to float32 [0, 1]
            canvas = canvas.astype(np.float32) / 255.0

            processed_frames.append(canvas)

        # (T, H, W, C)
        frames_array = np.stack(processed_frames)

        # (T, C, H, W)
        tensor = torch.from_numpy(
            frames_array
        ).permute(0, 3, 1, 2)

        # (1, T, C, H, W)
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, frames):

        # Preprocess webcam frames
        input_tensor = self.preprocess_frames(frames)

        # Model inference
        with torch.inference_mode():

            outputs = self.model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=5,
                dim=1
            )

        top_probabilities = top_probabilities[0].cpu().tolist()
        top_indices = top_indices[0].cpu().tolist()

        predictions = []

        for probability, index in zip(
            top_probabilities,
            top_indices
        ):

            predictions.append({
                "sign": self.classes[index],
                "confidence": round(
                    probability * 100,
                    2
                )
            })

        return {
            "predicted_sign": predictions[0]["sign"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions
        }


if __name__ == "__main__":

    service = InferenceService()

    # Create 16 dummy RGB frames
    dummy_frames = [
        np.random.randint(
            0,
            256,
            (480, 640, 3),
            dtype=np.uint8
        )
        for _ in range(16)
    ]

    result = service.predict(dummy_frames)

    print("\nINFERENCE SERVICE TEST SUCCESSFUL")
    print("\nPrediction:")
    print(result)