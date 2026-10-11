"""
backend/main.py
----------------
FastAPI Backend Application for Deepfake & Media Verification Platform.

Provides RESTful endpoints for image analysis, video frame analysis,
history management, and model evaluation metrics.
"""
import os
import sys
import time
import io
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.predictor import DeepfakePredictor

# Initialize FastAPI App
app = FastAPI(
    title="AI-Powered Deepfake Detection Platform API",
    description="Hybrid CNN–Transformer architecture backend for image & video deepfake detection.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for frontend assets (CSS, JS)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global Predictor Instance
predictor = DeepfakePredictor(model_type="hybrid")

# In-memory history log storage
history_db: List[Dict[str, Any]] = []


class HistoryItem(BaseModel):
    id: str
    filename: str
    media_type: str
    label: str
    confidence: float
    timestamp: str
    inference_time_ms: float
    details: Optional[Dict[str, Any]] = None


@app.get("/")
def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {
        "status": "online",
        "message": "AI-Powered Deepfake & Media Verification Platform API",
        "docs": "/docs",
        "model_architecture": "Hybrid CNN-Transformer (EfficientNet-B4 + ViT Encoder)",
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(predictor.device),
        "pytorch_version": torch.__version__,
        "model_type": predictor.model_type,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v1/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image or auto-route video file.
    """
    filename_lower = file.filename.lower()
    if file.content_type.startswith("video/") or filename_lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        return await analyze_video(file=file, max_frames=16)

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Perform Prediction
    result = predictor.predict_image(image)

    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    response_data = {
        "id": record_id,
        "filename": file.filename,
        "media_type": "image",
        "timestamp": timestamp,
        **result,
    }

    # Save to history
    history_db.insert(0, response_data)

    return response_data


@app.post("/api/v1/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    max_frames: int = Query(16, ge=4, le=64, description="Number of frames to extract and analyze"),
):
    """
    Analyze an uploaded video file (MP4, AVI, MOV) by extracting face sequence frames.
    """
    if not file.content_type.startswith("video/") and not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="File uploaded must be a video.")

    contents = await file.read()
    temp_video_path = os.path.join(PROJECT_ROOT, f"temp_{uuid.uuid4().hex[:6]}.mp4")

    try:
        with open(temp_video_path, "wb") as f:
            f.write(contents)

        # Extract frames using OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise HTTPException(status_code=400, detail="Could not read frames from video file.")

        step = max(1, total_frames // max_frames)
        extracted_frames = []

        frame_idx = 0
        while cap.isOpened() and len(extracted_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                extracted_frames.append(pil_img)

            frame_idx += 1

        cap.release()

        if not extracted_frames:
            raise HTTPException(status_code=400, detail="No valid frames extracted from video.")

        # Run Video Prediction
        result = predictor.predict_video_frames(extracted_frames)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    response_data = {
        "id": record_id,
        "filename": file.filename,
        "media_type": "video",
        "timestamp": timestamp,
        **result,
    }

    # Save to history
    history_db.insert(0, response_data)

    return response_data


@app.get("/api/v1/history")
def get_history(limit: int = Query(20, ge=1, le=100)):
    """
    Get analysis logs history.
    """
    return {
        "total_records": len(history_db),
        "history": history_db[:limit],
    }


@app.delete("/api/v1/history/{item_id}")
def delete_history_item(item_id: str):
    """
    Delete a specific prediction record from history log.
    """
    global history_db
    initial_len = len(history_db)
    history_db = [item for item in history_db if item.get("id") != item_id]

    if len(history_db) == initial_len:
        raise HTTPException(status_code=404, detail="Item not found.")

    return {"status": "success", "deleted_id": item_id}


@app.get("/api/v1/models")
def list_models():
    """
    List model architectures evaluated in this platform.
    """
    return {
        "active_model": predictor.model_type,
        "models": [
            {
                "id": "hybrid",
                "name": "Hybrid CNN-Transformer",
                "status": "active",
                "description": "Combines EfficientNet-B4 local feature extraction with Multi-head Attention transformer encoder for spatial & temporal modeling.",
            },
            {
                "id": "cnn",
                "name": "CNN Baseline (EfficientNet-B4)",
                "status": "ready",
                "description": "Standard convolutional neural network baseline fine-tuned for facial artifact detection.",
            },
            {
                "id": "transformer",
                "name": "Vision Transformer (ViT)",
                "status": "ready",
                "description": "Pure transformer baseline splitting input frames into 16x16 visual patches.",
            },
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
