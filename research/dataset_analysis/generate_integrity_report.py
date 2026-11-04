import csv
from collections import defaultdict

INPUT_FILE = "datasets/annotations/include50_metadata.csv"

REPORT_FILE = "datasets/annotations/include50_integrity_report.csv"

VALID_FILE = "datasets/annotations/include50_valid_candidates.csv"

QUARANTINE_FILE = "datasets/annotations/include50_quarantine.csv"


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["video_path"]].append(row)

    report = []
    valid = []
    quarantine = []

    for path, records in grouped.items():

        labels = set(row["label"] for row in records)

        categories = set(
            row["parent_label"] for row in records
        )

        path_category = path.split("/")[0]

        if len(labels) > 1:

            status = "CONFLICTING_LABELS"

        elif len(categories) > 1:

            status = "CONFLICTING_CATEGORIES"

        elif path_category not in categories:

            status = "PATH_CATEGORY_MISMATCH"

        elif len(records) > 1:

            status = "EXACT_DUPLICATE"

        else:

            status = "VALID_CANDIDATE"

        report.append({
            "video_path": path,
            "labels": " | ".join(sorted(labels)),
            "categories": " | ".join(sorted(categories)),
            "record_count": len(records),
            "status": status
        })

        if status == "VALID_CANDIDATE":
            valid.append(records[0])

        elif status == "EXACT_DUPLICATE":
            valid.append(records[0])
            quarantine.extend(records[1:])

        else:
            quarantine.extend(records)

    def save_csv(filename, data):

        if not data:
            return

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=data[0].keys()
            )

            writer.writeheader()
            writer.writerows(data)

    save_csv(REPORT_FILE, report)
    save_csv(VALID_FILE, valid)
    save_csv(QUARANTINE_FILE, quarantine)

    print("\n========== INTEGRITY REPORT ==========")

    print("Original records:", len(rows))
    print("Unique paths:", len(grouped))

    print("\nStatus distribution:")

    for status in sorted(set(r["status"] for r in report)):

        count = sum(
            r["status"] == status for r in report
        )

        print(status, ":", count)

    print("\nValid candidates:", len(valid))
    print("Quarantined records:", len(quarantine))

    print("\nGenerated files:")
    print(REPORT_FILE)
    print(VALID_FILE)
    print(QUARANTINE_FILE)


if __name__ == "__main__":
    main()