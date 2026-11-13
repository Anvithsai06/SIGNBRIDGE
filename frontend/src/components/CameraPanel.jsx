import { useRef, useState, useEffect } from "react";
import axios from "axios";
import {
  Camera,
  CameraOff,
  Play,
  Square,
  Loader2,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000/api/predict";

export default function CameraPanel({ onPrediction }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const busyRef = useRef(false);

  const [cameraActive, setCameraActive] = useState(false);
  const [recognizing, setRecognizing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  async function startCamera() {
    try {
      setError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
    } catch (err) {
      setError(
        err.name === "NotAllowedError"
          ? "Camera permission was denied."
          : "Unable to access your camera."
      );
    }
  }

  function stopCamera() {
    stopRecognition();

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
  }

  function captureFrames() {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || video.readyState < 2) {
      throw new Error("Camera is not ready.");
    }

    const context = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const frames = [];

    for (let i = 0; i < 16; i++) {
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataURL = canvas.toDataURL("image/jpeg", 0.85);
      const base64 = dataURL.split(",")[1];

      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);

      for (let j = 0; j < binary.length; j++) {
        bytes[j] = binary.charCodeAt(j);
      }

      frames.push(
        new Blob([bytes], { type: "image/jpeg" })
      );
    }

    return frames;
  }

  async function predictSign() {
    if (busyRef.current) return;

    busyRef.current = true;
    setLoading(true);

    try {
      const frames = captureFrames();
      const formData = new FormData();

      frames.forEach((frame, index) => {
        formData.append("frames", frame, `frame_${index}.jpg`);
      });

      const response = await axios.post(API_URL, formData, {
        timeout: 120000,
      });

      onPrediction?.(response.data.result);
      setError("");
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Prediction failed."
      );
    } finally {
      busyRef.current = false;
      setLoading(false);
    }
  }

  function startRecognition() {
    if (!cameraActive || recognizing) return;

    setRecognizing(true);

    predictSign();

    timerRef.current = setInterval(() => {
      predictSign();
    }, 4000);
  }

  function stopRecognition() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setRecognizing(false);
  }

  return (
    <div>
      <div className="relative aspect-video bg-slate-950 rounded-xl overflow-hidden flex items-center justify-center">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-contain ${
            cameraActive ? "block" : "hidden"
          }`}
        />

        {!cameraActive && (
          <div className="text-center text-white">
            <CameraOff size={42} className="mx-auto mb-4 text-slate-500" />
            <p className="font-semibold">Camera is inactive</p>
            <p className="text-sm text-slate-400 mt-2">
              Enable your webcam to begin.
            </p>
          </div>
        )}

        {cameraActive && (
          <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/60 px-3 py-2 rounded-lg text-white text-xs">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            Camera active
          </div>
        )}

        {loading && (
          <div className="absolute bottom-4 right-4 bg-black/70 text-white rounded-lg px-3 py-2 flex items-center gap-2 text-xs">
            <Loader2 size={15} className="animate-spin" />
            Processing frames...
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />

      {error && (
        <p className="mt-3 text-sm text-red-600">{error}</p>
      )}

      <div className="flex flex-wrap gap-3 mt-5">
        {!cameraActive ? (
          <button
            onClick={startCamera}
            className="flex-1 flex items-center justify-center gap-2 bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700"
          >
            <Camera size={18} />
            Enable Camera
          </button>
        ) : (
          <>
            {!recognizing ? (
              <button
                onClick={startRecognition}
                className="flex-1 flex items-center justify-center gap-2 bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700"
              >
                <Play size={18} />
                Start Recognition
              </button>
            ) : (
              <button
                onClick={stopRecognition}
                className="flex-1 flex items-center justify-center gap-2 bg-red-600 text-white py-3 rounded-xl font-semibold hover:bg-red-700"
              >
                <Square size={17} />
                Stop Recognition
              </button>
            )}

            <button
              onClick={stopCamera}
              className="px-5 py-3 border border-slate-200 rounded-xl text-slate-600 hover:bg-slate-50"
            >
              Stop Camera
            </button>
          </>
        )}
      </div>
    </div>
  );
}