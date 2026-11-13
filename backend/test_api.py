import cv2
import requests
import numpy as np
import sys


API_URL = "http://127.0.0.1:8000/api/predict"
CONFIG_URL = "http://127.0.0.1:8000/api/config"

def extract_frames(video_path, num_frames):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError("Video contains no frames.")

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
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

    config_response = requests.get(CONFIG_URL, timeout=15)
    config_response.raise_for_status()
    frame_count = config_response.json()["frame_count"]

    print("Extracting video frames...")

    frames = extract_frames(video_path, frame_count)

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
