import os
import torch
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from training.data_loader import create_dataloaders
from models.cnn_lstm import MobileNetV3LSTM
from models.mobilenet_baseline import MobileNetV3Baseline


# ==========================================
# CONFIGURATION
# ==========================================

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    "CNN_LSTM": {
        "class": MobileNetV3LSTM,
        "checkpoint": "cnn_lstm_best.pth"
    },

    "MobileNetV3": {
        "class": MobileNetV3Baseline,
        "checkpoint": "mobilenet_baseline_best.pth"
    }
}


# ==========================================
# DEVICE
# ==========================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(model_name, model_class, checkpoint_name, test_loader):

    print("\n" + "=" * 60)
    print("EVALUATING:", model_name)
    print("=" * 60)

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        checkpoint_name
    )

    # Load model without downloading pretrained weights again
    model = model_class(
        num_classes=50,
        pretrained=False
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    all_predictions = []
    all_targets = []
    all_confidences = []

    # ======================================
    # TEST LOOP
    # ======================================

    with torch.no_grad():

        for videos, labels in tqdm(test_loader, desc=model_name):

            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(videos)

            probabilities = torch.softmax(outputs, dim=1)

            confidences, predictions = probabilities.max(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_targets.extend(
                labels.cpu().tolist()
            )

            all_confidences.extend(
                confidences.cpu().tolist()
            )

    # ======================================
    # METRICS
    # ======================================

    classes = test_loader.dataset.classes

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    print("\nTEST RESULTS")
    print("-" * 45)

    print(f"Accuracy:           {accuracy * 100:.2f}%")
    print(f"Macro Precision:    {precision:.4f}")
    print(f"Macro Recall:       {recall:.4f}")
    print(f"Macro F1-score:     {f1:.4f}")

    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1-score:  {weighted_f1:.4f}")

    # ======================================
    # CLASSIFICATION REPORT
    # ======================================

    report = classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_classification_report.csv"
    )

    report_df.to_csv(report_path)

    # ======================================
    # CONFUSION MATRIX
    # ======================================

    matrix = confusion_matrix(
        all_targets,
        all_predictions,
        labels=list(range(len(classes)))
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=classes,
        columns=classes
    )

    matrix_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_confusion_matrix.csv"
    )

    matrix_df.to_csv(matrix_path)

    # ======================================
    # SAVE PREDICTIONS
    # ======================================

    test_annotations = test_loader.dataset.annotations

    predictions_df = pd.DataFrame({
        "video_path": test_annotations["video_path"],
        "actual_sign": [
            classes[index] for index in all_targets
        ],
        "predicted_sign": [
            classes[index] for index in all_predictions
        ],
        "confidence": all_confidences
    })

    predictions_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print("\nSaved:")
    print(report_path)
    print(matrix_path)
    print(predictions_path)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1
    }


# ==========================================
# MAIN
# ==========================================

def main():

    print("SIGNBRIDGE MODEL EVALUATION")
    print("Device:", DEVICE)

    _, _, test_loader = create_dataloaders()

    print("\nTest videos:", len(test_loader.dataset))

    results = []

    for model_name, config in MODELS.items():

        result = evaluate_model(
            model_name,
            config["class"],
            config["checkpoint"],
            test_loader
        )

        results.append(result)

    # Save comparison
    comparison_df = pd.DataFrame(results)

    comparison_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(comparison_df.to_string(index=False))

    print("\nComparison saved:", comparison_path)


if __name__ == "__main__":
    main()