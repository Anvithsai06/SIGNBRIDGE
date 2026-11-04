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

        # Load complete annotations
        all_annotations = pd.read_csv(annotation_file)

        # Create one shared mapping using all 50 classes
        self.classes = sorted(
            all_annotations["label"].unique().tolist()
        )

        self.class_to_idx = {
            label: index
            for index, label in enumerate(self.classes)
        }

        self.idx_to_class = {
            index: label
            for label, index in self.class_to_idx.items()
        }

        # Validate requested split
        valid_splits = {"train", "val", "test"}

        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Choose from {valid_splits}"
            )

        # Filter annotations
        self.annotations = all_annotations[
            all_annotations["dataset_split"] == split
        ].reset_index(drop=True)

        if len(self.annotations) == 0:
            raise ValueError(f"No videos found for split: {split}")

        self.split = split

        # Video preprocessing
        self.processor = VideoPreprocessor(
            num_frames=num_frames,
            image_size=image_size
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        row = self.annotations.iloc[index]

        video_path = self.dataset_root / row["video_path"]

        # Process video
        frames = self.processor.load_video(str(video_path))

        # Convert (T, H, W, C) to (T, C, H, W)
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

        # Get label using shared mapping
        label = self.class_to_idx[row["label"]]

        return video_tensor, torch.tensor(
            label,
            dtype=torch.long
        )


if __name__ == "__main__":

    annotation_file = (
        "datasets/annotations/include50_valid_candidates.csv"
    )

    for split in ["train", "val", "test"]:

        dataset = INCLUDE50Dataset(
            annotation_file=annotation_file,
            split=split
        )

        print(f"\n{split.upper()} DATASET")
        print("-" * 40)
        print("Videos:", len(dataset))
        print("Classes:", len(dataset.classes))
        print("First class:", dataset.classes[0])
        print("Last class:", dataset.classes[-1])

    print("\nShared class mapping verified!")