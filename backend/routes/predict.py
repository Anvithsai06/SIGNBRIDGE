import cv2
import numpy as np
from pathlib import Path
import tempfile
import os

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.inference_service import InferenceService


router = APIRouter()
MAX_VIDEO_BYTES = 50 * 1024 * 1024
VIDEO_SUFFIXES = {".webm", ".mp4", ".mov"}

# Shared inference service
inference_service = None


def initialize_inference():
    global inference_service

    if inference_service is None:
        inference_service = InferenceService()


def get_inference_service():
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service is not ready.")
    return inference_service


@router.get("/config")
def inference_config():
    """Expose the contract embedded in the currently loaded checkpoint."""
    return get_inference_service().model_loader.get_inference_config()


@router.post("/predict")
async def predict_sign(
    frames: list[UploadFile] = File(...)
):
    service = get_inference_service()

    if len(frames) != service.num_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly {service.num_frames} frames are required."
        )

    decoded_frames = []

    try:

        for uploaded_file in frames:

            if uploaded_file.content_type not in (
                "image/jpeg",
                "image/png",
                "image/webp"
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Only JPEG, PNG and WebP frames are supported."
                )

            image_bytes = await uploaded_file.read()

            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="An uploaded frame is empty."
                )

            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to decode an uploaded frame."
                )

            # OpenCV BGR -> RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            decoded_frames.append(frame)

    finally:

        for uploaded_file in frames:
            await uploaded_file.close()

    try:

        result = service.predict(decoded_frames)

        return {
            "success": True,
            "result": result
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Inference error:", error)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )


@router.post("/predict-video")
async def predict_video(video: UploadFile = File(...)):
    """Predict one sign from a short browser-recorded video without retaining it."""
    service = get_inference_service()
    suffix = Path(video.filename or "recording.webm").suffix.lower()
    content_type = (video.content_type or "").split(";", 1)[0].lower()
    if suffix not in VIDEO_SUFFIXES or content_type not in {"video/webm", "video/mp4", "video/quicktime"}:
        raise HTTPException(status_code=400, detail="Upload a WebM, MP4, or MOV video recording.")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_path = temporary_file.name
            total_bytes = 0
            while chunk := await video.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="Video must be 50 MB or smaller.")
                temporary_file.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty.")
        result, metadata = service.predict_video(temporary_path)
        return {"success": True, "result": result, "video": metadata}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except HTTPException:
        raise
    except Exception as error:
        print("Video inference error:", error)
        raise HTTPException(status_code=500, detail="Video prediction failed.")
    finally:
        await video.close()
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
