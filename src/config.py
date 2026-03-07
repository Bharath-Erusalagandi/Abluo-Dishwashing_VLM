"""Shared configuration for DishSpace."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──
ROOT_DIR = Path(os.getenv("DISHSPACE_ROOT", Path(__file__).resolve().parent.parent)).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data"))).resolve()
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT_DIR / "models"))).resolve()
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(ROOT_DIR / ".cache"))).resolve()


@dataclass(frozen=True)
class SupabaseConfig:
    url: str = os.getenv("SUPABASE_URL", "")
    key: str = os.getenv("SUPABASE_KEY", "")
    service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.key)


@dataclass(frozen=True)
class ModelConfig:
    base_model: str = os.getenv("BASE_MODEL", "physical-intelligence/pi0-base")
    lora_adapter_path: str = os.getenv(
        "LORA_ADAPTER_PATH", str(MODELS_DIR / "dora" / "kitchen_default_v0.2")
    )
    default_profile: str = os.getenv("DEFAULT_KITCHEN_PROFILE", "default")
    device: str = os.getenv("DEVICE", "cuda")

    # DoRA defaults (Weight-Decomposed Low-Rank Adaptation)
    adapter_type: str = "dora"  # "dora" | "lora" — DoRA outperforms LoRA by 1-3%
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Fine-tune all linear layers (modern best practice) instead of just q_proj/v_proj
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Depth foundation model
    depth_model: str = os.getenv("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Large")

    # Segmentation foundation model
    segmentation_model: str = os.getenv("SEG_MODEL", "IDEA-Research/grounding-dino-base")
    sam_model: str = os.getenv("SAM_MODEL", "facebook/sam2-hiera-large")


@dataclass(frozen=True)
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    api_key: str = os.getenv("DISHSPACE_API_KEY", "dev-key-change-me")
    max_grasps: int = int(os.getenv("MAX_GRASPS", "5"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.7"))
    collision_check: bool = os.getenv("COLLISION_CHECK", "true").lower() == "true"
    rate_limit_per_min: int = 100


@dataclass(frozen=True)
class InferenceConfig:
    max_grasps: int = 5
    min_confidence: float = 0.7
    collision_check: bool = True
    coordinate_frame: str = "camera"  # camera | world | robot_base
    depth_completion: str = os.getenv(
        "DEPTH_COMPLETION", "depth_anything_v2"
    )  # depth_anything_v2 | ip_basic | none
    segmentation: str = os.getenv(
        "SEGMENTATION", "grounded_sam2"
    )  # grounded_sam2 | dbscan | none


@dataclass(frozen=True)
class Settings:
    supabase: SupabaseConfig = field(default_factory=SupabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
