from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.video_preprocessor import VideoPreprocessor


class INCLUDE50Dataset(Dataset):

    def __init__(
        self,
        annotation_file,
        dataset_root="datasets/raw/include50",
        split="train",
        num_frames=16,
        image_size=224
    ):

        self.dataset_root = Path(dataset_root)

        self.annotations = pd.read_csv(annotation_file)

        # Keep only the requested dataset split
        self.annotations = self.annotations[
            self.annotations["dataset_split"] == split
        ].reset_index(drop=True)

        # Create consistent class-to-index mapping
        self.classes = sorted(
            self.annotations["label"].unique().tolist()
        )

        self.class_to_idx = {
            label: index
            for index, label in enumerate(self.classes)
        }

        self.processor = VideoPreprocessor(
            num_frames=num_frames,
            image_size=image_size
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        row = self.annotations.iloc[index]

        video_path = self.dataset_root / row["video_path"]

        frames = self.processor.load_video(str(video_path))

        # (T, H, W, C) -> (T, C, H, W)
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

        label = self.class_to_idx[row["label"]]

        return video_tensor, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":

    annotation_file = (
        "datasets/annotations/include50_valid_candidates.csv"
    )

    dataset = INCLUDE50Dataset(
        annotation_file=annotation_file,
        split="train"
    )

    video, label = dataset[0]

    print("Dataset loaded successfully!")
    print("Total training videos:", len(dataset))
    print("Number of classes:", len(dataset.classes))
    print("Video tensor shape:", video.shape)
    print("Label:", label.item())
    print("Sign:", dataset.classes[label.item()])