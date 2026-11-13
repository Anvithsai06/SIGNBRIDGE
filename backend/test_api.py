import cv2
import requests
import numpy as np
import sys


API_URL = "http://127.0.0.1:8000/api/predict"

NUM_FRAMES = 16


def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError("Video contains no frames.")

    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES,
        dtype=int
    )

    frames = []

    for index in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

        success, frame = cap.read()

        if not success:
            cap.release()
            raise ValueError(f"Failed to read frame {index}")

        # Encode as JPEG.
        success, encoded = cv2.imencode(".jpg", frame)

        if not success:
            cap.release()
            raise ValueError("JPEG encoding failed.")

        frames.append(encoded.tobytes())

    cap.release()

    return frames


def test_prediction(video_path):

    print("Extracting video frames...")

    frames = extract_frames(video_path)

    print("Extracted frames:", len(frames))

    files = [
        (
            "frames",
            (f"frame_{i}.jpg", frame, "image/jpeg")
        )
        for i, frame in enumerate(frames)
    ]

    print("\nSending request to SIGNBRIDGE API...")

    response = requests.post(
        API_URL,
        files=files,
        timeout=120
    )

    print("\nHTTP Status:", response.status_code)

    response.raise_for_status()

    result = response.json()

    print("\nPREDICTION RESULT")
    print("=" * 40)

    print("Predicted sign:", result["result"]["predicted_sign"])

    print("Confidence:", result["result"]["confidence"], "%")

    print("\nTop 5 predictions:")

    for prediction in result["result"]["top_predictions"]:

        print(
            prediction["sign"],
            "->",
            prediction["confidence"],
            "%"
        )

    print("\nREAL VIDEO API TEST SUCCESSFUL!")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python -m backend.test_api <video_path>")
        sys.exit(1)

    test_prediction(sys.argv[1])