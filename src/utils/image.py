"""Image and depth processing utilities."""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from PIL import Image

from src.utils.logging import get_logger

log = get_logger(__name__)


def decode_image_base64(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded image to a numpy RGB array (H, W, 3), uint8."""
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def decode_depth_base64(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded 16-bit PNG depth map to numpy array (H, W), uint16.

    Values in millimetres (RealSense convention).
    """
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img, dtype=np.uint16)


def encode_image_base64(img_array: np.ndarray, fmt: str = "PNG") -> str:
    """Encode a numpy image array to base64 string."""
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_if_needed(
    img: np.ndarray, max_width: int = 640, max_height: int = 480
) -> np.ndarray:
    """Resize image if it exceeds max dimensions, preserving aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_width and h <= max_height:
        return img
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil_img)


def depth_completion_ip_basic(depth: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Fill depth holes using morphological closing (IP-Basic style).

    Fast (<5ms) depth completion for missing pixels from RealSense on
    reflective / wet surfaces. Kept as a lightweight CPU fallback.

    Args:
        depth: (H, W) uint16 depth map with 0 = missing.
        kernel_size: Morphological kernel size.

    Returns:
        Filled depth map, same shape and dtype.
    """
    import cv2

    filled = depth.copy()
    mask = (filled == 0).astype(np.uint8)

    # Dilate valid pixels to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for _ in range(4):
        dilated = cv2.dilate(filled, kernel, iterations=1)
        filled = np.where(mask, dilated, filled)
        mask = (filled == 0).astype(np.uint8)

    # Median blur to smooth filled regions
    filled = cv2.medianBlur(filled, 5)
    return filled.astype(np.uint16)


# ── Depth Anything V2 ──

# Singleton to avoid reloading per request
_depth_anything_model = None
_depth_anything_transform = None


def _load_depth_anything_v2():
    """Lazy-load Depth Anything V2 model (singleton)."""
    global _depth_anything_model, _depth_anything_transform

    if _depth_anything_model is not None:
        return _depth_anything_model, _depth_anything_transform

    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    from src.config import settings

    model_id = settings.model.depth_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("loading_depth_anything_v2", model=model_id, device=device)
    _depth_anything_transform = AutoImageProcessor.from_pretrained(model_id)
    _depth_anything_model = AutoModelForDepthEstimation.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    _depth_anything_model.eval()
    log.info("depth_anything_v2_ready", model=model_id)

    return _depth_anything_model, _depth_anything_transform


def depth_completion_depth_anything_v2(
    rgb: np.ndarray,
    depth: np.ndarray | None = None,
    scale_to_metric: bool = True,
) -> np.ndarray:
    """Complete / predict depth using Depth Anything V2.

    Handles reflective surfaces, wet ceramics, and transparent glass far
    better than morphological hole-filling because the model has learned
    geometric priors for these materials.

    If an existing depth map is provided, the model's prediction is used
    only to fill holes (pixels == 0) so we preserve accurate sensor
    readings where they exist.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        depth: (H, W) uint16 optional sensor depth map (0 = missing). If
            None, the model predicts a full depth map from monocular RGB.
        scale_to_metric: Align the relative prediction to metric scale
            using valid sensor pixels (requires ``depth`` to be provided).

    Returns:
        (H, W) uint16 depth map in millimetres.
    """
    import torch

    model, processor = _load_depth_anything_v2()
    device = next(model.parameters()).device

    pil_img = Image.fromarray(rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted = outputs.predicted_depth  # (1, H', W') relative depth

    # Resize prediction to original image dimensions
    predicted = torch.nn.functional.interpolate(
        predicted.unsqueeze(0),
        size=rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    if depth is not None and scale_to_metric:
        # Align relative prediction to metric scale using valid sensor pixels
        valid_mask = depth > 0
        if valid_mask.sum() > 100:
            sensor_vals = depth[valid_mask].astype(np.float64)
            pred_vals = predicted[valid_mask].astype(np.float64)
            # Least-squares: sensor ≈ scale * predicted + offset
            scale = np.dot(sensor_vals, pred_vals) / (np.dot(pred_vals, pred_vals) + 1e-8)
            predicted = predicted * scale
        # Use model prediction only where sensor has holes
        result = depth.copy().astype(np.float64)
        hole_mask = depth == 0
        result[hole_mask] = predicted[hole_mask]
        return np.clip(result, 0, 65535).astype(np.uint16)

    # No sensor depth — return normalised prediction in mm (assume 0.3–2m range)
    pred_min, pred_max = predicted.min(), predicted.max()
    if pred_max - pred_min > 1e-6:
        normalised = (predicted - pred_min) / (pred_max - pred_min)
    else:
        normalised = np.zeros_like(predicted)
    depth_mm = (normalised * 1700 + 300)  # map to 300–2000 mm
    return depth_mm.astype(np.uint16)


def create_synthetic_rgbd(
    width: int = 640,
    height: int = 480,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic RGB-D pair for testing.

    Returns:
        (rgb, depth) where rgb is (H,W,3) uint8 and depth is (H,W) uint16.
    """
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    depth = rng.integers(300, 1500, (height, width), dtype=np.uint16)
    # Add some depth holes (simulating reflective surfaces)
    hole_mask = rng.random((height, width)) < 0.02
    depth[hole_mask] = 0
    return rgb, depth
