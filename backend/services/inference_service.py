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
        self.num_frames = self.model_loader.preprocessing["num_frames"]
        self.image_size = self.model_loader.preprocessing["image_size"]

    def preprocess_frames(self, frames):

        if len(frames) != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, received {len(frames)}"
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
                self.image_size / width,
                self.image_size / height
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
                (self.image_size, self.image_size, 3),
                114,
                dtype=np.uint8
            )

            x_offset = (self.image_size - new_width) // 2
            y_offset = (self.image_size - new_height) // 2

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

    def decode_video(self, video_path, max_duration_seconds=30):
        """Decode a complete recording and sample it as training did: linspace over time."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Unable to read the uploaded video. Try MP4 or a browser-supported WebM codec.")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if frame_count < self.num_frames:
                raise ValueError(f"Video must contain at least {self.num_frames} frames.")
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Video frame rate could not be determined.")
            duration = frame_count / fps
            if duration <= 0 or duration > max_duration_seconds:
                raise ValueError(f"Video duration must be between 0 and {max_duration_seconds} seconds.")

            # This matches VideoPreprocessor.load_video used for training/evaluation.
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError("Unable to decode all sampled video frames.")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return frames, {"frame_count": frame_count, "fps": fps, "duration_seconds": duration}
        finally:
            capture.release()

    def predict_video(self, video_path):
        frames, metadata = self.decode_video(video_path)
        return self.predict(frames), metadata


if __name__ == "__main__":

    service = InferenceService()

    # Use the frame count supplied by the loaded checkpoint.
    dummy_frames = [
        np.random.randint(
            0,
            256,
            (480, 640, 3),
            dtype=np.uint8
        )
        for _ in range(service.num_frames)
    ]

    result = service.predict(dummy_frames)

    print("\nINFERENCE SERVICE TEST SUCCESSFUL")
    print("\nPrediction:")
    print(result)
