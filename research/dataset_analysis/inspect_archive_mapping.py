import json
import ssl
import certifi
from urllib.request import urlopen
from collections import defaultdict

API_URL = "https://zenodo.org/api/records/4010759"

VALID_FILE = "datasets/annotations/include50_valid_candidates.csv"

SSL_CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)


def main():

    print("Connecting to Zenodo...")

    with urlopen(
        API_URL,
        context=SSL_CONTEXT,
        timeout=60
    ) as response:

        data = json.loads(response.read().decode("utf-8"))

    archives = defaultdict(list)

    for file in data["files"]:

        name = file["key"]

        if name.endswith(".zip"):

            category = name.rsplit("_", 1)[0]

            archives[category].append({
                "name": name,
                "size": file["size"]
            })

    print("\n========== ZENODO ARCHIVE INVENTORY ==========")

    total_size = 0

    for category, files in sorted(archives.items()):

        category_size = sum(
            item["size"] for item in files
        )

        total_size += category_size

        print(f"\n{category}")

        for item in files:
            size_gb = item["size"] / (1024 ** 3)

            print(
                f"  {item['name']} - {size_gb:.2f} GB"
            )

        print(
            f"  Category total: "
            f"{category_size / (1024 ** 3):.2f} GB"
        )

    print("\nTotal archive categories:", len(archives))
    print("Total ZIP size:", round(total_size / (1024 ** 3), 2), "GB")


if __name__ == "__main__":
    main()