import os
import csv
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.data_loader import create_dataloaders
from models.mobilenet_baseline import MobileNetV3Baseline

# ==========================================
# CONFIGURATION
# ==========================================

EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 4

NUM_CLASSES = 50

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# DEVICE CONFIGURATION
# ==========================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training")

    for videos, labels in progress:

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(videos)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# VALIDATION FUNCTION
# ==========================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        progress = tqdm(loader, desc="Validation")

        for videos, labels in progress:

            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(videos)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================

def main():

    print("\nLoading INCLUDE-50 datasets...")

    train_loader, val_loader, test_loader = create_dataloaders()

    print("Training videos:", len(train_loader.dataset))
    print("Validation videos:", len(val_loader.dataset))
    print("Testing videos:", len(test_loader.dataset))

    # Initialize CNN-LSTM
    model = MobileNetV3Baseline(
    num_classes=NUM_CLASSES,
    pretrained=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_val_loss = float("inf")
    early_stop_counter = 0

    history_path = os.path.join(
        RESULTS_DIR,
        "mobilenet_baseline_letterbox_history.csv"
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "mobilenet_baseline_letterbox_best.pth"
    )

    history = []

    print("\nStarting CNN-LSTM training...")
    print("=" * 60)

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print("\nEpoch Results")
        print("-" * 45)

        print(f"Training Loss:   {train_loss:.4f}")
        print(f"Training Acc:    {train_accuracy * 100:.2f}%")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Acc:  {val_accuracy * 100:.2f}%")

        print(f"Learning Rate:   {current_lr:.6f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr
        })

        # Save best checkpoint
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            early_stop_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "classes": train_loader.dataset.classes
                },
                checkpoint_path
            )

            print("\nBest model checkpoint saved!")

        else:

            early_stop_counter += 1

            print(
                f"\nNo improvement: "
                f"{early_stop_counter}/{PATIENCE}"
            )

        # Save training history
        with open(
            history_path,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=history[0].keys()
            )

            writer.writeheader()
            writer.writerows(history)

        # Early stopping
        if early_stop_counter >= PATIENCE:

            print("\nEarly stopping triggered!")
            break

    print("\n" + "=" * 60)
    print("CNN-LSTM TRAINING FINISHED")

    print("Best validation loss:", round(best_val_loss, 4))
    print("Checkpoint:", checkpoint_path)
    print("Training history:", history_path)


if __name__ == "__main__":
    main()