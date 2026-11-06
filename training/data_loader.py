import random
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from preprocessing.include50_dataset import INCLUDE50Dataset


# ==========================================
# CONFIGURATION
# ==========================================

ANNOTATION_FILE = (
    "datasets/annotations/include50_valid_candidates.csv"
)

BATCH_SIZE = 2
NUM_FRAMES = 16
IMAGE_SIZE = 224


# ==========================================
# VIDEO AUGMENTATION
# ==========================================

class VideoAugmentation:

    def __init__(self):

        self.brightness = 0.15
        self.contrast = 0.15

        self.rotation = 8

        self.translation = 0.05

        self.scale_min = 0.95
        self.scale_max = 1.05

    def __call__(self, video):

        # Expected shape: [T, C, H, W]

        _, _, height, width = video.shape

        # Generate shared parameters for all frames
        angle = random.uniform(
            -self.rotation,
            self.rotation
        )

        translate_x = int(
            random.uniform(
                -self.translation,
                self.translation
            ) * width
        )

        translate_y = int(
            random.uniform(
                -self.translation,
                self.translation
            ) * height
        )

        scale = random.uniform(
            self.scale_min,
            self.scale_max
        )

        brightness_factor = random.uniform(
            1 - self.brightness,
            1 + self.brightness
        )

        contrast_factor = random.uniform(
            1 - self.contrast,
            1 + self.contrast
        )

        # Apply the same geometric transformation
        # to every frame in the video.
        video = TF.affine(
            video,
            angle=angle,
            translate=[translate_x, translate_y],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=[0.447, 0.447, 0.447]
        )

        # Apply consistent brightness and contrast
        video = TF.adjust_brightness(
            video,
            brightness_factor
        )

        video = TF.adjust_contrast(
            video,
            contrast_factor
        )

        return video.clamp(0.0, 1.0)


# ==========================================
# AUGMENTED DATASET WRAPPER
# ==========================================

class AugmentedVideoDataset(Dataset):

    def __init__(self, dataset, augmentation):

        self.dataset = dataset
        self.augmentation = augmentation

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        video, label = self.dataset[index]

        video = self.augmentation(video)

        return video, label


# ==========================================
# DATALOADER CREATION
# ==========================================

def create_dataloaders():

    print("\nLoading INCLUDE-50 datasets...")

    # Original training dataset
    train_dataset = INCLUDE50Dataset(
        annotation_file=ANNOTATION_FILE,
        split="train",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE
    )

    # Apply augmentation only to training data
    train_dataset = AugmentedVideoDataset(
        train_dataset,
        VideoAugmentation()
    )

    # Validation dataset - no augmentation
    val_dataset = INCLUDE50Dataset(
        annotation_file=ANNOTATION_FILE,
        split="val",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE
    )

    # Test dataset - no augmentation
    test_dataset = INCLUDE50Dataset(
        annotation_file=ANNOTATION_FILE,
        split="test",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    print("\nDATALOADERS CREATED SUCCESSFULLY")
    print("=" * 45)

    print("Training videos:", len(train_loader.dataset))
    print("Validation videos:", len(val_loader.dataset))
    print("Testing videos:", len(test_loader.dataset))

    videos, labels = next(iter(train_loader))

    print("\nFIRST TRAINING BATCH")
    print("-" * 45)

    print("Video batch shape:", videos.shape)
    print("Label batch shape:", labels.shape)

    print("Video data type:", videos.dtype)

    print("Minimum pixel value:", videos.min().item())
    print("Maximum pixel value:", videos.max().item())

    print("Labels:", labels.tolist())

    print("\nDataLoader verification complete!")