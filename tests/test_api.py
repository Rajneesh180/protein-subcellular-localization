"""Tests for FastAPI inference endpoint."""

import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture
def client():
    return TestClient(app)


def _create_test_image() -> bytes:
    """Create a dummy RGBA image."""
    img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
    pil_img = Image.fromarray(img, "RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_predict_no_model(client):
    """Without a loaded model, /predict should return 503."""
    image_bytes = _create_test_image()
    resp = client.post("/predict", files={"file": ("test.png", image_bytes, "image/png")})
    # model won't be loaded in test — expect 503
    assert resp.status_code == 503


def test_predict_empty_file(client):
    resp = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code in (400, 503)
