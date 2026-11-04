import csv
import json
import ssl
import certifi
import struct

from urllib.request import Request, urlopen
from collections import defaultdict


API_URL = "https://zenodo.org/api/records/4010759"

CSV_FILE = "datasets/annotations/include50_valid_candidates.csv"

TAIL_SIZE = 1024 * 1024

CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)


def fetch_json(url):

    with urlopen(
        url,
        context=CONTEXT,
        timeout=60
    ) as response:

        return json.loads(response.read().decode())


def fetch_range(url, start, end):

    request = Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}"
        }
    )

    with urlopen(
        request,
        context=CONTEXT,
        timeout=60
    ) as response:

        if response.status != 206:
            raise RuntimeError(
                "Partial content request failed."
            )

        return response.read()


def get_candidates():

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        rows = list(csv.DictReader(file))

    return {
        row["video_path"]: row
        for row in rows
    }


def inspect_archive(file):

    url = file["links"]["self"]
    size = file["size"]

    start = max(0, size - TAIL_SIZE)

    tail = fetch_range(
        url,
        start,
        size - 1
    )

    signature = b"PK\x05\x06"

    position = tail.rfind(signature)

    if position == -1:
        raise RuntimeError(
            f"ZIP directory not found: {file['key']}"
        )

    eocd = tail[position:position + 22]

    if len(eocd) < 22:
        raise RuntimeError("Incomplete ZIP directory.")

    (
        sig,
        disk,
        directory_disk,
        disk_entries,
        total_entries,
        directory_size,
        directory_offset,
        comment_length
    ) = struct.unpack("<4s4H2IH", eocd)

    if (
        disk != 0
        or directory_disk != 0
        or disk_entries != total_entries
    ):
        raise RuntimeError("Multi-disk ZIP unsupported.")

    # ZIP64 requires a separate parser.
    if (
        total_entries == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    ):
        raise RuntimeError("ZIP64 archive detected.")

    directory = fetch_range(
        url,
        directory_offset,
        directory_offset + directory_size - 1
    )

    entries = []

    offset = 0

    for _ in range(total_entries):

        if directory[offset:offset + 4] != b"PK\x01\x02":
            raise RuntimeError("Invalid directory entry.")

        fields = struct.unpack_from(
            "<4s6H3I5H2I",
            directory,
            offset
        )

        compression_method = fields[4]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        crc32 = fields[7]

        filename_length = fields[10]
        extra_length = fields[11]
        entry_comment_length = fields[12]

        local_header_offset = fields[16]

        filename_start = offset + 46

        filename = directory[
            filename_start:
            filename_start + filename_length
        ].decode("utf-8", errors="replace")

        entries.append({
            "archive": file["key"],
            "filename": filename,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "crc32": crc32,
            "flags": fields[3],
            "compression_method": compression_method,
            "local_header_offset": local_header_offset,
            "archive_size": size,
            "url": url
        })

        offset += (
            46
            + filename_length
            + extra_length
            + entry_comment_length
        )

    return entries


def main():

    print("Loading INCLUDE-50 candidates...")

    candidates = get_candidates()

    print("Valid candidate paths:", len(candidates))

    print("\nConnecting to Zenodo...")

    record = fetch_json(API_URL)

    archives = [
        file for file in record["files"]
        if file["key"].endswith(".zip")
    ]

    matched = {}
    archive_sizes = {}

    print("\n========== INSPECTING ARCHIVES ==========")

    for index, archive in enumerate(archives, start=1):

        name = archive["key"]

        print(
            f"\n[{index}/{len(archives)}] Inspecting {name}"
        )

        try:

            entries = inspect_archive(archive)

        except Exception as error:

            print("ERROR:", error)
            continue

        archive_sizes[name] = archive["size"]

        for entry in entries:

            path = entry["filename"]

            if path in candidates:

                matched[path] = entry

                print("MATCH:", path)

    print("\n========== MAPPING SUMMARY ==========")

    print("Candidate videos:", len(candidates))
    print("Matched videos:", len(matched))
    print("Unmatched videos:", len(candidates) - len(matched))

    required_archives = defaultdict(list)

    for path, entry in matched.items():

        required_archives[entry["archive"]].append(entry)

    print("\n========== REQUIRED ARCHIVES ==========")

    total_video_size = 0
    total_archive_size = 0

    for archive, entries in sorted(required_archives.items()):

        compressed = sum(
            entry["compressed_size"]
            for entry in entries
        )

        total_video_size += compressed
        total_archive_size += archive_sizes[archive]

        print(
            f"\n{archive}: "
            f"{len(entries)} videos"
        )

        print(
            "Compressed video size:",
            round(compressed / (1024 ** 2), 2),
            "MB"
        )

    print("\n========== STORAGE ESTIMATE ==========")

    print(
        "Required archive download:",
        round(total_archive_size / (1024 ** 3), 2),
        "GB"
    )

    print(
        "Matched videos compressed size:",
        round(total_video_size / (1024 ** 3), 2),
        "GB"
    )

    output_file = "datasets/annotations/include50_archive_mapping.csv"

    if matched:

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(next(iter(matched.values())).keys())
            )

            writer.writeheader()
            writer.writerows(matched.values())

        print("\nMapping saved to:", output_file)

    print("\nInspection completed.")


if __name__ == "__main__":
    main()