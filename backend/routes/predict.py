import cv2
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.inference_service import InferenceService


router = APIRouter()

# Shared inference service
inference_service = None


def initialize_inference():
    global inference_service

    if inference_service is None:
        inference_service = InferenceService()


@router.post("/predict")
async def predict_sign(
    frames: list[UploadFile] = File(...)
):

    if len(frames) != 16:
        raise HTTPException(
            status_code=400,
            detail="Exactly 16 frames are required."
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

        result = inference_service.predict(decoded_frames)

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