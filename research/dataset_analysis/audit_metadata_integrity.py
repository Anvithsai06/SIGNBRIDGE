import csv
from collections import defaultdict

CSV_FILE = "datasets/annotations/include50_metadata.csv"

def audit_metadata():

    with open(CSV_FILE, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    path_records = defaultdict(list)

    for row in rows:
        path_records[row["video_path"]].append(row)

    duplicates = {
        path: records
        for path, records in path_records.items()
        if len(records) > 1
    }

    print("\n========== DATA INTEGRITY AUDIT ==========")

    print("Total records:", len(rows))
    print("Unique video paths:", len(path_records))
    print("Repeated paths:", len(duplicates))

    print("\n========== DUPLICATE VIDEO INVESTIGATION ==========")

    for path, records in duplicates.items():

        print("\nVideo:", path)

        for record in records:
            print(
                "Label:", record["label"],
                "| Category:", record["parent_label"],
                "| Split:", record["dataset_split"]
            )

    print("\n========== PATH-CATEGORY MISMATCHES ==========")

    mismatches = []

    for row in rows:

        folder = row["video_path"].split("/")[0]

        if folder != row["parent_label"]:
            mismatches.append(row)

    print("Category mismatches:", len(mismatches))

    for row in mismatches:
        print(
            "\nPath:", row["video_path"],
            "| Expected category:", row["parent_label"],
            "| Label:", row["label"]
        )

    print("\n========== SPLIT LEAKAGE CHECK ==========")

    leakage = {}

    for path, records in path_records.items():

        splits = set(
            record["dataset_split"]
            for record in records
        )

        if len(splits) > 1:
            leakage[path] = splits

    print("Paths appearing in multiple splits:", len(leakage))

    for path, splits in leakage.items():
        print(path, "->", sorted(splits))

    print("\n========== AUDIT SUMMARY ==========")

    print("Duplicate paths:", len(duplicates))
    print("Category mismatches:", len(mismatches))
    print("Cross-split paths:", len(leakage))

    print("\nAudit completed.")


if __name__ == "__main__":
    audit_metadata()