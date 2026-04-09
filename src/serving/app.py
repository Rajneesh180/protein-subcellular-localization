"""FastAPI inference endpoint for protein localization prediction."""

import io
import os
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

from src.models.factory import build_model
from src.data.dataset import HPA_LABELS
from src.data.augmentation import HPA_MEAN, HPA_STD
from src.serving.schemas import PredictionResult, HealthResponse


MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/best_model.pth")
MODEL_NAME = os.environ.get("MODEL_NAME", "hybrid")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "256"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = build_model(MODEL_NAME, num_classes=28, pretrained=False)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        print(f"Loaded {MODEL_NAME} from {MODEL_PATH}")
    else:
        print(f"Warning: checkpoint not found at {MODEL_PATH}")
    yield


app = FastAPI(
    title="Protein Subcellular Localization API",
    description="Multi-label prediction from fluorescence microscopy images",
    version="1.0.0",
    lifespan=lifespan,
)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Load uploaded image, convert to 4-channel tensor.

    Accepts either a single RGBA/RGB image or expects 4 grayscale channels
    packed into an RGBA png.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    img_array = np.array(img, dtype=np.float32)

    if img_array.ndim == 2:
        # single channel — stack 4x
        img_array = np.stack([img_array] * 4, axis=-1)
    elif img_array.shape[-1] == 3:
        # RGB — add dummy 4th channel
        dummy = np.zeros((*img_array.shape[:2], 1), dtype=np.float32)
        img_array = np.concatenate([img_array, dummy], axis=-1)
    elif img_array.shape[-1] > 4:
        img_array = img_array[:, :, :4]

    # normalize with same stats as training and convert to (1, 4, H, W) tensor
    mean = torch.tensor(HPA_MEAN).view(4, 1, 1)
    std = torch.tensor(HPA_STD).view(4, 1, 1)
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = (tensor - mean) / std
    return tensor


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    image = preprocess_image(contents).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    predicted_labels = [HPA_LABELS[i] for i in range(len(probs)) if probs[i] >= THRESHOLD]
    prob_dict = {HPA_LABELS[i]: round(float(probs[i]), 4) for i in range(len(probs))}

    return PredictionResult(
        labels=predicted_labels,
        probabilities=prob_dict,
        threshold=THRESHOLD,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
    )
