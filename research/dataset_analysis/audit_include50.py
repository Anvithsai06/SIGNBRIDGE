"""Reproducible integrity audit for the locally curated INCLUDE-50 subset.

The audit is deliberately read-only: it never moves, rewrites, or excludes data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / "datasets/annotations/include50_valid_candidates.csv"
DATA_ROOT = ROOT / "datasets/raw/include50"
OUT_DIR = ROOT / "research/dataset_analysis"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def video_info(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    frame_ok, frame = cap.read() if opened else (False, None)
    info = {
        "opened": opened,
        "first_frame_decoded": bool(frame_ok and frame is not None and frame.size),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0,
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0,
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0,
    }
    cap.release()
    return info


def main() -> None:
    annotations = pd.read_csv(ANNOTATIONS)
    required = {"label", "video_path", "dataset_split"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Missing required columns: {required - set(annotations.columns)}")

    rows: list[dict[str, object]] = []
    for row in annotations.itertuples(index=False):
        relative = Path(row.video_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe video path: {row.video_path}")
        path = DATA_ROOT / relative
        record = {
            "video_path": row.video_path,
            "label": row.label,
            "split": row.dataset_split,
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": digest(path) if path.is_file() else "",
        }
        record.update(video_info(path) if path.is_file() else {})
        rows.append(record)

    details = pd.DataFrame(rows)
    details.to_csv(OUT_DIR / "include50_local_audit.csv", index=False)

    split_class = pd.crosstab(annotations["label"], annotations["dataset_split"])
    split_class.to_csv(OUT_DIR / "include50_split_class_counts.csv")

    hashes = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            hashes[row["sha256"]].append(row["video_path"])
    exact_duplicates = {key: value for key, value in hashes.items() if len(value) > 1}

    labels_by_path = annotations.groupby("video_path")["label"].nunique()
    splits_by_path = annotations.groupby("video_path")["dataset_split"].nunique()
    class_split_counts = annotations.groupby(["label", "dataset_split"]).size().unstack(fill_value=0)
    missing_test_classes = [
        label for label in sorted(annotations.label.unique())
        if class_split_counts.loc[label].get("test", 0) == 0
    ]

    summary = {
        "records": len(annotations),
        "classes": int(annotations.label.nunique()),
        "split_counts": {key: int(value) for key, value in annotations.dataset_split.value_counts().items()},
        "missing_files": int((~details.exists).sum()),
        "unopenable_or_undecodable_first_frame": int(
            (~details.opened | ~details.first_frame_decoded).sum()
        ),
        "duplicate_annotation_paths": int(annotations.video_path.duplicated().sum()),
        "conflicting_labels_for_path": int((labels_by_path > 1).sum()),
        "cross_split_path_leakage": int((splits_by_path > 1).sum()),
        "exact_duplicate_file_groups": len(exact_duplicates),
        "missing_test_classes": missing_test_classes,
        "samples_per_class": {
            key: int(value) for key, value in annotations.label.value_counts().sort_index().items()
        },
    }
    (OUT_DIR / "include50_local_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
