import cv2
import numpy as np


class VideoPreprocessor:

    def __init__(self, num_frames=16, image_size=224):
        self.num_frames = num_frames
        self.image_size = image_size

    def resize_with_padding(self, frame):

        height, width = frame.shape[:2]

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

        return canvas

    def load_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"No frames found: {video_path}")

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=int
        )

        frames = []

        for frame_index in frame_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

            success, frame = cap.read()

            if not success:
                cap.release()
                raise ValueError(
                    f"Failed to read frame {frame_index}: {video_path}"
                )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = self.resize_with_padding(frame)

            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

        cap.release()

        return np.stack(frames)


if __name__ == "__main__":

    processor = VideoPreprocessor()

    video_path = (
        "datasets/raw/include50/"
        "Adjectives/78. long/MVI_5107.MOV"
    )

    frames = processor.load_video(video_path)

    print("Preprocessing successful!")
    print("Output shape:", frames.shape)
    print("Data type:", frames.dtype)
    print("Minimum pixel value:", frames.min())
    print("Maximum pixel value:", frames.max())