import csv
import cv2
import time

from pathlib import Path
from collections import Counter


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets" / "raw" / "include50"

MAPPING_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "annotations"
    / "include50_archive_mapping.csv"
)

REPORT_DIR = PROJECT_ROOT / "datasets" / "annotations"

AUDIT_REPORT = REPORT_DIR / "downloaded_dataset_audit.csv"

CLASS_REPORT = REPORT_DIR / "class_distribution.csv"


# =====================================================
# AUDIT ONE VIDEO
# =====================================================

def audit_video(video_path):

    result = {
        "filename": str(video_path.relative_to(DATASET_DIR)),
        "exists": video_path.exists(),
        "file_size_mb": 0,
        "opened": False,
        "decoded": False,
        "frames": 0,
        "fps": 0,
        "width": 0,
        "height": 0,
        "duration_seconds": 0,
        "status": "FAILED",
        "error": ""
    }

    if not video_path.exists():
        result["error"] = "File missing"
        return result

    result["file_size_mb"] = round(
        video_path.stat().st_size / (1024 * 1024), 2
    )

    cap = cv2.VideoCapture(str(video_path))

    try:

        if not cap.isOpened():
            result["error"] = "OpenCV could not open video"
            return result

        result["opened"] = True

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result["fps"] = round(fps, 2)
        result["width"] = width
        result["height"] = height

        frame_count = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame is None or frame.size == 0:
                result["error"] = "Invalid frame encountered"
                break

            frame_count += 1

        result["frames"] = frame_count

        if fps > 0:
            result["duration_seconds"] = round(
                frame_count / fps, 2
            )

        result["decoded"] = frame_count > 0

        if frame_count > 0 and not result["error"]:
            result["status"] = "VALID"
        else:
            result["error"] = result["error"] or "No frames decoded"

    except Exception as error:

        result["error"] = str(error)

    finally:

        cap.release()

    return result


# =====================================================
# MAIN AUDIT
# =====================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with MAPPING_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        expected_entries = list(csv.DictReader(file))

    total_expected = len(expected_entries)

    print("\n========================================")
    print("       SIGNBRIDGE DATASET AUDIT")
    print("========================================")

    print("Expected videos:", total_expected)

    results = []

    class_counts = Counter()

    valid_count = 0
    failed_count = 0
    missing_count = 0

    start_time = time.time()

    for index, entry in enumerate(expected_entries, start=1):

        relative_path = Path(entry["filename"])

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Unsafe path in mapping: {entry['filename']}"
            )

        video_path = DATASET_DIR / relative_path

        class_name = relative_path.parent.name

        print(
            f"\n[{index}/{total_expected}] "
            f"{entry['filename']}"
        )

        result = audit_video(video_path)

        result["class"] = class_name

        results.append(result)

        if not result["exists"]:
            missing_count += 1

        if result["status"] == "VALID":

            valid_count += 1
            class_counts[class_name] += 1

            print(
                f"VALID | Frames: {result['frames']} | "
                f"FPS: {result['fps']} | "
                f"Resolution: {result['width']}x{result['height']}"
            )

        else:

            failed_count += 1

            print("FAILED:", result["error"])

    # =================================================
    # SAVE VIDEO AUDIT REPORT
    # =================================================

    fields = [
        "filename",
        "class",
        "exists",
        "file_size_mb",
        "opened",
        "decoded",
        "frames",
        "fps",
        "width",
        "height",
        "duration_seconds",
        "status",
        "error"
    ]

    with AUDIT_REPORT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fields)

        writer.writeheader()
        writer.writerows(results)

    # =================================================
    # SAVE CLASS DISTRIBUTION
    # =================================================

    with CLASS_REPORT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["class", "valid_video_count"])

        for class_name, count in sorted(class_counts.items()):
            writer.writerow([class_name, count])

    # =================================================
    # FINAL SUMMARY
    # =================================================

    elapsed = time.time() - start_time

    print("\n========================================")
    print("           AUDIT SUMMARY")
    print("========================================")

    print("Expected videos:", total_expected)
    print("Valid videos:", valid_count)
    print("Failed videos:", failed_count)
    print("Missing videos:", missing_count)

    print("Classes found:", len(class_counts))

    print(
        "Total audit time:",
        round(elapsed / 60, 2),
        "minutes"
    )

    print("\nReports generated:")

    print(AUDIT_REPORT)
    print(CLASS_REPORT)

    print("\nDataset audit completed.")


if __name__ == "__main__":
    main()