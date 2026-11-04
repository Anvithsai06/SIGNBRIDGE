import json
import ssl
import certifi
import csv
from urllib.request import urlopen
from urllib.parse import urlencode
from collections import Counter

DATASET = "ai4bharat/INCLUDE"

BASE_URL = "https://datasets-server.huggingface.co"

SSL_CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)


def fetch_json(url):
    """Fetch JSON data securely from Hugging Face."""

    with urlopen(
        url,
        context=SSL_CONTEXT,
        timeout=60
    ) as response:

        return json.loads(response.read().decode("utf-8"))


def get_splits():
    """Retrieve available dataset splits."""

    url = f"{BASE_URL}/splits?{urlencode({'dataset': DATASET})}"

    data = fetch_json(url)

    return data["splits"]


def get_split_rows(config, split):
    """Retrieve all metadata rows for a split."""

    all_rows = []
    offset = 0
    limit = 100

    while True:

        params = {
            "dataset": DATASET,
            "config": config,
            "split": split,
            "offset": offset,
            "length": limit
        }

        url = f"{BASE_URL}/rows?{urlencode(params)}"

        data = fetch_json(url)

        rows = data.get("rows", [])

        if not rows:
            break

        for item in rows:
            all_rows.append(item["row"])

        offset += len(rows)

        total = data.get("num_rows_total", 0)

        print(
            f"  Retrieved {len(all_rows)} / {total} rows"
        )

        if offset >= total:
            break

    return all_rows


def is_true(value):
    """Handle Boolean values safely."""

    return value is True or str(value).lower() == "true"


def inspect_metadata():

    print("\nConnecting to Hugging Face...")
    splits = get_splits()

    print("\nAvailable splits:")

    for item in splits:
        print(item)

    include50_records = []

    print("\n========== DOWNLOADING METADATA ==========")

    for item in splits:

        config = item["config"]
        split = item["split"]

        print(f"\nInspecting: {split}")

        rows = get_split_rows(config, split)

        for row in rows:

            if is_true(row.get("include_50")):

                row["dataset_split"] = split
                include50_records.append(row)

    if not include50_records:

        print("\nNo INCLUDE-50 records found.")
        print("Check the dataset configuration and metadata.")

        return

    print("\n========== INCLUDE-50 SUMMARY ==========")

    print("Total INCLUDE-50 records:", len(include50_records))

    labels = Counter(
        row.get("label", "Unknown")
        for row in include50_records
    )

    parent_labels = Counter(
        row.get("parent_label", "Unknown")
        for row in include50_records
    )

    split_counts = Counter(
        row.get("dataset_split", "Unknown")
        for row in include50_records
    )

    print("Unique sign labels:", len(labels))
    print("Parent categories:", len(parent_labels))

    print("\nSplit distribution:")

    for split, count in split_counts.items():
        print(f"{split}: {count}")

    print("\nParent category distribution:")

    for category, count in parent_labels.items():
        print(f"{category}: {count}")

    print("\nSign label distribution:")

    for label, count in labels.most_common():
        print(f"{label}: {count}")

    output_file = "datasets/annotations/include50_metadata.csv"

    fieldnames = list(include50_records[0].keys())

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(include50_records)

    print("\nMetadata saved to:", output_file)
    print("\nInspection completed successfully!")


if __name__ == "__main__":
    inspect_metadata()