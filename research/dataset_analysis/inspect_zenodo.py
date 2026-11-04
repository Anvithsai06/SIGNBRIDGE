import json
import ssl
import certifi
from urllib.request import urlopen

API_URL = "https://zenodo.org/api/records/4010759"


def format_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def inspect_dataset():

    print("Connecting to Zenodo...")

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    with urlopen(API_URL, context=ssl_context, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    print("\n========== DATASET INFORMATION ==========")

    print("Title:", data["metadata"]["title"])
    print("Record ID:", data["id"])

    files = data.get("files", [])

    print("\n========== AVAILABLE FILES ==========")

    total_size = 0

    for index, file in enumerate(files, start=1):

        name = file["key"]
        size = file["size"]

        total_size += size

        print(f"\nFile {index}")
        print("Name:", name)
        print("Size:", format_size(size))

    print("\n========== STORAGE SUMMARY ==========")

    print("Total files:", len(files))
    print("Total download size:", format_size(total_size))

    print("\nInspection completed.")


if __name__ == "__main__":
    inspect_dataset()