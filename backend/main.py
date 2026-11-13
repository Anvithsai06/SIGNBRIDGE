from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.predict import (
    router as predict_router,
    initialize_inference
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting SIGNBRIDGE backend...")

    # Load model once when the application starts
    initialize_inference()

    print("SIGNBRIDGE backend ready!")

    yield

    print("SIGNBRIDGE backend shutting down...")


app = FastAPI(
    title="SIGNBRIDGE API",
    description=(
        "Real-Time Indian Sign Language "
        "Recognition System"
    ),
    version="1.0.0",
    lifespan=lifespan
)


# React development server access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def root():

    return {
        "message": "Welcome to SIGNBRIDGE API",
        "status": "running"
    }


@app.get("/api/health")
def health():

    return {
        "success": True,
        "message": "SIGNBRIDGE API is healthy"
    }


app.include_router(
    predict_router,
    prefix="/api",
    tags=["Sign Recognition"]
)