import csv
import ssl
import certifi
import struct
import zlib
import time
import urllib.request
import urllib.error

from pathlib import Path
from urllib.parse import unquote


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAPPING_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "annotations"
    / "include50_archive_mapping.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "include50"
)

FAILURE_LOG = (
    PROJECT_ROOT
    / "datasets"
    / "annotations"
    / "download_failures.csv"
)

# SSL certificate configuration
SSL_CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)

# Number of retries per video
MAX_RETRIES = 3

# None = download all videos.
# Set to 5 if you want to test with only five videos first.
MAX_VIDEOS = None


# ============================================================
# HTTP RANGE REQUEST
# ============================================================

def fetch_range(url, start, end):

    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "SIGNBRIDGE-Dataset-Downloader/1.0"
        }
    )

    with urllib.request.urlopen(
        request,
        context=SSL_CONTEXT,
        timeout=90
    ) as response:

        if response.status != 206:
            raise RuntimeError(
                f"Range request failed: HTTP {response.status}"
            )

        expected_range = f"bytes {start}-{end}/"

        content_range = response.headers.get(
            "Content-Range", ""
        )

        if not content_range.startswith(expected_range):
            raise RuntimeError(
                f"Unexpected Content-Range: {content_range}"
            )

        data = response.read()

    expected_length = end - start + 1

    if len(data) != expected_length:
        raise RuntimeError(
            f"Incomplete response: expected {expected_length}, "
            f"received {len(data)} bytes"
        )

    return data


# ============================================================
# VERIFY EXISTING FILE
# ============================================================

def verify_existing_file(path, expected_size, expected_crc):

    if not path.exists():
        return False

    if path.stat().st_size != expected_size:
        return False

    crc = 0

    with path.open("rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            crc = zlib.crc32(chunk, crc)

    crc &= 0xFFFFFFFF

    return crc == expected_crc


# ============================================================
# EXTRACT ONE VIDEO FROM REMOTE ZIP
# ============================================================

def extract_video(entry):

    url = entry["url"]

    offset = int(entry["local_header_offset"])

    compressed_size = int(entry["compressed_size"])

    uncompressed_size = int(entry["uncompressed_size"])

    expected_crc = int(entry["crc32"])

    expected_flags = int(entry["flags"])

    expected_method = int(entry["compression_method"])

    archive_size = int(entry["archive_size"])

    # Reject unsupported ZIP64 entries.
    if (
        compressed_size == 0xFFFFFFFF
        or uncompressed_size == 0xFFFFFFFF
        or offset == 0xFFFFFFFF
    ):
        raise RuntimeError("ZIP64 entry is not supported.")

    # Reject encrypted entries.
    if expected_flags & 0x41:
        raise RuntimeError("Encrypted ZIP entry.")

    # Reject multi-disk or unsupported compression.
    if expected_method not in (0, 8):
        raise RuntimeError(
            f"Unsupported compression method: {expected_method}"
        )

    # --------------------------------------------------------
    # STEP 1: Read local ZIP header
    # --------------------------------------------------------

    header = fetch_range(
        url,
        offset,
        offset + 29
    )

    fields = struct.unpack(
        "<IHHHHHIIIHH",
        header
    )

    signature = fields[0]
    local_flags = fields[2]
    local_method = fields[3]
    local_crc = fields[6]
    local_compressed_size = fields[7]
    local_uncompressed_size = fields[8]
    filename_length = fields[9]
    extra_length = fields[10]

    if signature != 0x04034B50:
        raise RuntimeError("Invalid ZIP local header.")

    if local_flags != expected_flags:
        raise RuntimeError("ZIP flags mismatch.")

    if local_method != expected_method:
        raise RuntimeError("Compression method mismatch.")

    # If data descriptor is not present, verify local metadata.
    if not (expected_flags & 0x08):

        if (
            local_crc != expected_crc
            or local_compressed_size != compressed_size
            or local_uncompressed_size != uncompressed_size
        ):
            raise RuntimeError("ZIP local metadata mismatch.")

    # --------------------------------------------------------
    # STEP 2: Verify filename
    # --------------------------------------------------------

    name_start = offset + 30
    name_end = name_start + filename_length - 1

    if filename_length <= 0:
        raise RuntimeError("Invalid ZIP filename length.")

    local_name = fetch_range(
        url,
        name_start,
        name_end
    )

    encoding = (
        "utf-8"
        if expected_flags & 0x800
        else "cp437"
    )

    decoded_name = local_name.decode(encoding)

    if decoded_name != entry["filename"]:
        raise RuntimeError("ZIP filename mismatch.")

    # --------------------------------------------------------
    # STEP 3: Calculate video payload position
    # --------------------------------------------------------

    data_start = (
        offset
        + 30
        + filename_length
        + extra_length
    )

    data_end = data_start + compressed_size - 1

    if data_start < 0 or data_end >= archive_size:
        raise RuntimeError("Invalid video data range.")

    # --------------------------------------------------------
    # STEP 4: Download ONLY video payload
    # --------------------------------------------------------

    compressed_data = fetch_range(
        url,
        data_start,
        data_end
    )

    # --------------------------------------------------------
    # STEP 5: Decompress
    # --------------------------------------------------------

    if expected_method == 0:

        video_data = compressed_data

    elif expected_method == 8:

        video_data = zlib.decompress(
            compressed_data,
            -15
        )

    # --------------------------------------------------------
    # STEP 6: Verify extracted size
    # --------------------------------------------------------

    if len(video_data) != uncompressed_size:
        raise RuntimeError("Extracted size mismatch.")

    # --------------------------------------------------------
    # STEP 7: Verify CRC32
    # --------------------------------------------------------

    actual_crc = zlib.crc32(video_data) & 0xFFFFFFFF

    if actual_crc != expected_crc:
        raise RuntimeError("CRC32 verification failed.")

    # --------------------------------------------------------
    # STEP 8: Safely construct output path
    # --------------------------------------------------------

    relative_path = Path(decoded_name)

    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise RuntimeError("Unsafe output path.")

    output_path = OUTPUT_DIR / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # STEP 9: Save atomically
    # --------------------------------------------------------

    temporary_path = output_path.with_name(
        output_path.name + ".part"
    )

    try:

        with temporary_path.open("wb") as file:
            file.write(video_data)

        temporary_path.replace(output_path)

    except Exception:

        temporary_path.unlink(missing_ok=True)
        raise

    return output_path


# ============================================================
# DOWNLOAD WITH RETRIES
# ============================================================

def download_with_retry(entry):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            return extract_video(entry)

        except Exception as error:

            print(
                f"\nRetry {attempt}/{MAX_RETRIES} failed: {error}"
            )

            if attempt == MAX_RETRIES:
                raise

            time.sleep(3 * attempt)


# ============================================================
# FAILURE REPORT
# ============================================================

def save_failure_report(failures):

    with FAILURE_LOG.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "archive",
                "filename",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(failures)


# ============================================================
# MAIN BATCH DOWNLOADER
# ============================================================

def main():

    with MAPPING_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        entries = list(csv.DictReader(file))

    if MAX_VIDEOS is not None:
        entries = entries[:MAX_VIDEOS]

    total = len(entries)

    downloaded = 0
    skipped = 0
    failed = 0

    failures = []

    total_downloaded_bytes = 0

    total_expected_bytes = sum(
        int(entry["uncompressed_size"])
        for entry in entries
    )

    print("\n======================================")
    print("       SIGNBRIDGE DATASET DOWNLOADER")
    print("======================================")

    print("Videos to process:", total)

    print(
        f"Expected extracted size: "
        f"{total_expected_bytes / (1024 ** 3):.2f} GB"
    )

    print("Output:", OUTPUT_DIR)

    print("\nStarting download...\n")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, entry in enumerate(entries, start=1):

        relative_path = Path(entry["filename"])

        output_path = OUTPUT_DIR / relative_path

        expected_size = int(entry["uncompressed_size"])

        expected_crc = int(entry["crc32"])

        print(
            f"\n[{index}/{total}] "
            f"{entry['filename']}"
        )

        # ----------------------------------------------------
        # Skip already verified files
        # ----------------------------------------------------

        if verify_existing_file(
            output_path,
            expected_size,
            expected_crc
        ):

            print("Already verified. Skipping.")

            skipped += 1

            continue

        # ----------------------------------------------------
        # Download video
        # ----------------------------------------------------

        try:

            saved_path = download_with_retry(entry)

            downloaded += 1

            total_downloaded_bytes += saved_path.stat().st_size

            print("Download successful.")
            print("CRC32 verified.")

        except Exception as error:

            failed += 1

            print("DOWNLOAD FAILED:", error)

            failures.append({
                "archive": entry["archive"],
                "filename": entry["filename"],
                "error": str(error)
            })

            save_failure_report(failures)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        completed = downloaded + skipped + failed

        print("\n---------- PROGRESS ----------")

        print(
            f"Completed: {completed}/{total}"
        )

        print("Downloaded:", downloaded)
        print("Skipped:", skipped)
        print("Failed:", failed)

        print(
            f"New data downloaded: "
            f"{total_downloaded_bytes / (1024 ** 3):.2f} GB"
        )

        print("------------------------------")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n======================================")
    print("          DOWNLOAD SUMMARY")
    print("======================================")

    print("Total videos:", total)
    print("Downloaded:", downloaded)
    print("Already verified:", skipped)
    print("Failed:", failed)

    print(
        f"New data downloaded: "
        f"{total_downloaded_bytes / (1024 ** 3):.2f} GB"
    )

    if failures:

        save_failure_report(failures)

        print("\nFailure report:")
        print(FAILURE_LOG)

    else:

        print("\nAll processed videos were successful.")

    print("\nSIGNBRIDGE dataset download finished.")


if __name__ == "__main__":
    main()