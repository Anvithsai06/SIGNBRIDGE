import csv
from collections import Counter
from pathlib import PurePosixPath

CSV_FILE = "datasets/annotations/include50_metadata.csv"


def analyze_paths():

    print("Reading INCLUDE-50 metadata...")

    with open(CSV_FILE, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    categories = Counter()
    video_extensions = Counter()
    video_paths = set()
    duplicate_paths = []

    print("\n========== VIDEO PATH ANALYSIS ==========")

    for row in rows:

        path = row["video_path"]

        parts = PurePosixPath(path).parts

        if len(parts) < 3:
            print("Unexpected path:", path)
            continue

        category = parts[0]
        extension = PurePosixPath(path).suffix.lower()

        categories[category] += 1
        video_extensions[extension] += 1

        if path in video_paths:
            duplicate_paths.append(path)

        video_paths.add(path)

    print("\nTotal metadata records:", len(rows))
    print("Unique video paths:", len(video_paths))
    print("Duplicate paths:", len(duplicate_paths))

    print("\n========== CATEGORY DISTRIBUTION ==========")

    for category, count in categories.most_common():
        print(f"{category}: {count} videos")

    print("\n========== VIDEO FORMATS ==========")

    for extension, count in video_extensions.items():
        print(f"{extension}: {count}")

    print("\n========== SAMPLE VIDEO PATHS ==========")

    for row in rows[:15]:
        print(row["video_path"])

    print("\n========== REQUIRED CATEGORIES ==========")

    print("Total categories:", len(categories))

    print("\nAnalysis completed.")


if __name__ == "__main__":
    analyze_paths()