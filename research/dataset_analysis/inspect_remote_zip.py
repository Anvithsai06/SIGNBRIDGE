import json
import ssl
import certifi
import struct
import re

from urllib.request import Request, urlopen


API_URL = "https://zenodo.org/api/records/4010759"

TARGET_ARCHIVE = "Seasons_1of1.zip"

TAIL_SIZE = 1024 * 1024

CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)


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
                "Server did not return partial content."
            )

        return response.read()


def get_archive():

    with urlopen(
        API_URL,
        context=CONTEXT,
        timeout=60
    ) as response:

        record = json.loads(response.read().decode())

    for file in record["files"]:

        if file["key"] == TARGET_ARCHIVE:

            return file["links"]["self"], file["size"]

    raise RuntimeError("Archive not found.")


def inspect_zip():

    print("Retrieving Zenodo archive information...")

    url, file_size = get_archive()

    print("Archive:", TARGET_ARCHIVE)
    print("Archive size:", file_size)

    print("\nRetrieving final archive section...")

    start = max(0, file_size - TAIL_SIZE)

    tail = fetch_range(
        url,
        start,
        file_size - 1
    )

    signature = b"PK\x05\x06"

    position = tail.rfind(signature)

    if position == -1:
        raise RuntimeError(
            "ZIP End of Central Directory not found."
        )

    eocd = tail[position:position + 22]

    if len(eocd) < 22:
        raise RuntimeError("Incomplete ZIP directory record.")

    (
        signature,
        disk_number,
        directory_disk,
        entries_on_disk,
        total_entries,
        directory_size,
        directory_offset,
        comment_length
    ) = struct.unpack("<4s4H2IH", eocd)

    if (
        disk_number != 0
        or directory_disk != 0
        or entries_on_disk != total_entries
    ):
        raise RuntimeError(
            "Multi-disk ZIP archives are not supported."
        )

    print("\n========== ZIP DIRECTORY ==========")

    print("Total entries:", total_entries)
    print("Directory size:", directory_size)

    directory = fetch_range(
        url,
        directory_offset,
        directory_offset + directory_size - 1
    )

    offset = 0

    for index in range(total_entries):

        if directory[offset:offset + 4] != b"PK\x01\x02":
            raise RuntimeError(
                "Invalid central directory entry."
            )

        fields = struct.unpack_from(
            "<4s6H3I5H2I",
            directory,
            offset
        )

        compressed_size = fields[8]
        uncompressed_size = fields[9]

        filename_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]

        filename_start = offset + 46

        filename = directory[
            filename_start:
            filename_start + filename_length
        ]

        filename = filename.decode(
            "utf-8",
            errors="replace"
        )

        if filename.lower().endswith((".mov", ".mp4")):

            print(
                f"{index + 1}. {filename} "
                f"| {uncompressed_size / (1024 ** 2):.2f} MB"
            )

        offset += (
            46
            + filename_length
            + extra_length
            + comment_length
        )

    print("\nRemote ZIP inspection completed.")

    print("Archive bytes retrieved:", len(tail) + len(directory))


if __name__ == "__main__":
    inspect_zip()