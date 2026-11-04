from torch.utils.data import DataLoader

from preprocessing.include50_dataset import INCLUDE50Dataset


ANNOTATION_FILE = (
    "datasets/annotations/include50_valid_candidates.csv"
)

BATCH_SIZE = 2
NUM_FRAMES = 16
IMAGE_SIZE = 224


def create_dataloaders():

    train_dataset = INCLUDE50Dataset(
        annotation_file=ANNOTATION_FILE,
        split="train",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE
    )

    val_dataset = INCLUDE50Dataset(
        annotation_file=ANNOTATION_FILE,
        split="val",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE
    )

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


if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders()

    print("\nDATALOADERS CREATED SUCCESSFULLY")
    print("=" * 45)

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))

    videos, labels = next(iter(train_loader))

    print("\nFIRST TRAINING BATCH")
    print("-" * 45)

    print("Video batch shape:", videos.shape)
    print("Label batch shape:", labels.shape)
    print("Video data type:", videos.dtype)
    print("Labels:", labels.tolist())

    print("\nDataLoader verification complete!")