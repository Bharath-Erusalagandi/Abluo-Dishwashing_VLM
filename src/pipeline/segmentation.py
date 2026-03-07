"""Grounded SAM 2 object segmentation pipeline.

Replaces heuristic DBSCAN clustering with foundation-model-based
zero-shot segmentation.  Pipeline:

  RGB image
    → Grounding DINO (open-vocabulary detection)
    → SAM 2 (per-object mask prediction)
    → list[SegmentedObject] with pixel-precise masks

This is the single biggest accuracy upgrade for DishSpace: it handles
transparent glass, wet ceramics, and stacked/nested objects that
break colour/depth-based heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

# Kitchen-domain object labels for Grounding DINO
KITCHEN_OBJECT_LABELS = (
    "mug . plate . bowl . wine glass . tumbler . champagne glass . "
    "measuring cup . pot . pan . fork . knife . spoon . spatula . "
    "ladle . tray . cutting board . cup . dish . saucer . glass"
)


@dataclass
class DetectedObject:
    """A single detected + segmented object."""

    label: str = "unknown"
    score: float = 0.0
    bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None  # (H, W) bool — pixel-precise mask
    rgb_crop: Optional[np.ndarray] = None


# ── Singletons ──
_grounding_dino_model = None
_grounding_dino_processor = None
_sam2_model = None
_sam2_processor = None


def _load_grounding_dino():
    """Lazy-load Grounding DINO model."""
    global _grounding_dino_model, _grounding_dino_processor

    if _grounding_dino_model is not None:
        return _grounding_dino_model, _grounding_dino_processor

    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = settings.model.segmentation_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("loading_grounding_dino", model=model_id, device=device)
    _grounding_dino_processor = AutoProcessor.from_pretrained(model_id)
    _grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    _grounding_dino_model.eval()
    log.info("grounding_dino_ready")

    return _grounding_dino_model, _grounding_dino_processor


def _load_sam2():
    """Lazy-load SAM 2 model."""
    global _sam2_model, _sam2_processor

    if _sam2_model is not None:
        return _sam2_model, _sam2_processor

    import torch
    from transformers import AutoProcessor, AutoModel

    model_id = settings.model.sam_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("loading_sam2", model=model_id, device=device)
    _sam2_processor = AutoProcessor.from_pretrained(model_id)
    _sam2_model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    _sam2_model.eval()
    log.info("sam2_ready")

    return _sam2_model, _sam2_processor


def detect_objects_grounding_dino(
    rgb: np.ndarray,
    text_prompt: str = KITCHEN_OBJECT_LABELS,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> list[DetectedObject]:
    """Detect kitchen objects using Grounding DINO (open-vocabulary).

    Args:
        rgb: (H, W, 3) uint8 image.
        text_prompt: Dot-separated list of object labels.
        box_threshold: Minimum box confidence.
        text_threshold: Minimum text-match score.

    Returns:
        List of DetectedObject with bounding boxes and scores.
    """
    import torch
    from PIL import Image

    model, processor = _load_grounding_dino()
    device = next(model.parameters()).device

    pil_img = Image.fromarray(rgb)
    inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[rgb.shape[:2]],
    )[0]

    detections: list[DetectedObject] = []
    h, w = rgb.shape[:2]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        detections.append(DetectedObject(
            label=label,
            score=float(score),
            bbox=[x1, y1, x2, y2],
            rgb_crop=rgb[y1:y2, x1:x2].copy() if (x2 - x1 > 0 and y2 - y1 > 0) else None,
        ))

    log.info("grounding_dino_detections", count=len(detections))
    return detections


def segment_with_sam2(
    rgb: np.ndarray,
    detections: list[DetectedObject],
) -> list[DetectedObject]:
    """Refine bounding-box detections into pixel-precise masks with SAM 2.

    Takes the bounding boxes from Grounding DINO and prompts SAM 2 to
    produce per-object segmentation masks.

    Args:
        rgb: (H, W, 3) uint8 image.
        detections: Objects with bounding boxes from Grounding DINO.

    Returns:
        Same detections list, now with ``.mask`` populated.
    """
    if not detections:
        return detections

    import torch
    from PIL import Image

    model, processor = _load_sam2()
    device = next(model.parameters()).device
    pil_img = Image.fromarray(rgb)

    # Batch all bounding boxes into a single SAM 2 call
    input_boxes = [[d.bbox for d in detections]]

    inputs = processor(
        images=pil_img,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )[0]  # first (and only) image

    for i, det in enumerate(detections):
        if i < len(masks):
            # Take highest-IoU mask prediction (SAM returns 3 per prompt)
            mask = masks[i].cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]  # best mask
            det.mask = mask.astype(bool)

    log.info("sam2_masks_generated", count=sum(1 for d in detections if d.mask is not None))
    return detections


def segment_objects(
    rgb: np.ndarray,
    text_prompt: str = KITCHEN_OBJECT_LABELS,
    box_threshold: float = 0.30,
) -> list[DetectedObject]:
    """Full Grounded SAM 2 pipeline: detect + segment.

    This is the main entry point. It chains Grounding DINO → SAM 2 to
    produce pixel-precise, labelled object masks from a single RGB image.

    Args:
        rgb: (H, W, 3) uint8 image.
        text_prompt: Object classes to detect.
        box_threshold: Minimum detection confidence.

    Returns:
        List of DetectedObject with labels, scores, bboxes, and masks.
    """
    detections = detect_objects_grounding_dino(rgb, text_prompt, box_threshold)
    detections = segment_with_sam2(rgb, detections)
    return detections


def detections_to_label_map(
    detections: list[DetectedObject],
    height: int,
    width: int,
) -> np.ndarray:
    """Convert detected objects into an integer label map.

    Returns:
        (H, W) int32 array where 0 = background, 1..N = object index.
    """
    label_map = np.zeros((height, width), dtype=np.int32)
    for i, det in enumerate(detections, start=1):
        if det.mask is not None and det.mask.shape == (height, width):
            label_map[det.mask] = i
    return label_map
