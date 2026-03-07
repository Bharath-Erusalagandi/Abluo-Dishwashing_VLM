"""Pydantic schemas for the DishSpace API and data pipeline.

These are the canonical data shapes used across:
- REST API request / response
- Supabase storage
- Training data annotation
- ROS trajectory export
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ──


class ObjectType(str, Enum):
    MUG = "mug"
    PLATE = "plate"
    BOWL = "bowl"
    WINE_GLASS = "wine_glass"
    TUMBLER = "tumbler"
    CHAMPAGNE_GLASS = "champagne_glass"
    MEASURING_CUP = "measuring_cup"
    POT = "pot"
    PAN = "pan"
    FORK = "fork"
    KNIFE = "knife"
    SPOON = "spoon"
    SPATULA = "spatula"
    LADLE = "ladle"
    TRAY = "tray"
    CUTTING_BOARD = "cutting_board"
    OTHER = "other"


class ObjectMaterial(str, Enum):
    CERAMIC_DRY = "ceramic_dry"
    CERAMIC_WET = "ceramic_wet"
    GLASS = "glass"
    METAL = "metal"
    PLASTIC = "plastic"
    WOOD = "wood"
    SILICONE = "silicone"


class FailureMode(str, Enum):
    SLIP = "slip"
    COLLISION = "collision"
    OCCLUSION = "occlusion"
    SOAP = "soap"
    DEPTH_HOLE = "depth_hole"
    FRAGILE_BREAK = "fragile_break"
    INCORRECT_POSE = "incorrect_pose"
    GRIP_FORCE = "grip_force"
    NONE = "none"  # successful grasp


class GraspType(str, Enum):
    RIM_PINCH = "rim_pinch"
    EDGE_PINCH = "edge_pinch"
    HANDLE_WRAP = "handle_wrap"
    BODY_WRAP = "body_wrap"
    STEM_PINCH = "stem_pinch"
    FLAT_SUCTION = "flat_suction"
    PARALLEL_JAW = "parallel_jaw"
    CUSTOM = "custom"


class DataSource(str, Enum):
    MUJOCO_SIM = "mujoco_sim"
    YOUTUBE = "youtube"
    ROS_BAG = "ros_bag"
    PILOT = "pilot"
    MANUAL = "manual"


class SceneType(str, Enum):
    ISOLATED = "isolated"
    SINK_SINGLE = "sink_single"
    SINK_CLUTTERED = "sink_cluttered"
    UTENSIL_BUNDLE = "utensil_bundle"
    DRYING_RACK = "drying_rack"
    DISHWASHER_RACK = "dishwasher_rack"


class TaskTarget(str, Enum):
    SINK = "sink"
    DRYING_RACK = "drying_rack"
    DISHWASHER_TOP_RACK = "dishwasher_top_rack"
    DISHWASHER_BOTTOM_RACK = "dishwasher_bottom_rack"
    UTENSIL_CADDY = "utensil_caddy"


class CoordinateFrame(str, Enum):
    CAMERA = "camera"
    WORLD = "world"
    ROBOT_BASE = "robot_base"


# ── Sub-models ──


class EnvironmentConditions(BaseModel):
    wet: bool = False
    soap: bool = False
    steam: bool = False
    lighting: str = "overhead_bright"
    rack_type: str = "none"
    temperature_c: Optional[float] = None
    scene_type: SceneType = SceneType.SINK_SINGLE
    visible_object_count: int = Field(1, ge=1, le=20)
    occlusion_level: float = Field(0.0, ge=0.0, le=1.0)
    target_zone: TaskTarget = TaskTarget.DRYING_RACK


class RobotConfig(BaseModel):
    arm: str = "UR5"
    gripper: str = "Robotiq_2F85"
    camera: str = "RealSense_D435"
    camera_pose: list[float] = Field(
        default_factory=lambda: [0.5, 0.0, 0.8, 0.0, 0.785, 0.0],
        description="[x, y, z, rx, ry, rz] of camera in robot base frame",
    )


class FailureRisk(BaseModel):
    slip: float = Field(0.0, ge=0.0, le=1.0)
    collision: float = Field(0.0, ge=0.0, le=1.0)
    occlusion: float = Field(0.0, ge=0.0, le=1.0)
    depth_hole: float = Field(0.0, ge=0.0, le=1.0)


class SceneMetadata(BaseModel):
    objects_detected: int = 0
    depth_quality: float = Field(1.0, ge=0.0, le=1.0)
    wet_surface_detected: bool = False
    soap_presence: bool = False
    steam_detected: bool = False


# ── Annotation Schema (Data Pipeline) ──


class GraspAnnotation(BaseModel):
    """Single annotated grasp attempt — the core training data unit.

    Stored in Supabase `grasp_annotations` table.
    """

    sample_id: str = Field(default_factory=lambda: f"gs_{uuid.uuid4().hex[:8]}")
    source: DataSource = DataSource.MUJOCO_SIM
    object_type: ObjectType = ObjectType.OTHER
    object_material: ObjectMaterial = ObjectMaterial.CERAMIC_DRY

    grasp_point_xyz: list[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] grasp contact point"
    )
    grasp_normal: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 1.0],
        min_length=3,
        max_length=3,
        description="Surface normal at grasp point",
    )
    approach_vector: list[float] = Field(
        default_factory=lambda: [0.0, -1.0, 0.0],
        min_length=3,
        max_length=3,
        description="Approach direction of gripper",
    )
    grip_width_mm: float = Field(80.0, gt=0.0, description="Gripper opening width in mm")
    grip_force_n: float = Field(5.0, gt=0.0, description="Applied grip force in Newtons")

    success: bool = False
    failure_mode: FailureMode = FailureMode.NONE
    failure_detail: str = ""

    environment: EnvironmentConditions = Field(default_factory=EnvironmentConditions)
    robot_config: RobotConfig = Field(default_factory=RobotConfig)

    rgb_path: str = ""
    depth_path: str = ""
    pointcloud_path: str = ""

    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


# ── API Request / Response ──


class GraspOptions(BaseModel):
    collision_check: bool = True
    max_grasps: int = Field(5, ge=1, le=20)
    min_confidence: float = Field(0.7, ge=0.0, le=1.0)
    coordinate_frame: CoordinateFrame = CoordinateFrame.CAMERA


class GraspRequest(BaseModel):
    """POST /grasp_plan request body."""

    image_base64: str = Field(..., description="RGB frame, base64-encoded PNG/JPEG")
    depth_base64: Optional[str] = Field(None, description="Depth map, base64-encoded 16-bit PNG")
    kitchen_profile: str = "default"
    robot: str = "UR5_realsense"
    base_model: str = "pi0-base"
    options: GraspOptions = Field(default_factory=GraspOptions)


class GraspPose(BaseModel):
    """A single predicted grasp."""

    pose: list[float] = Field(
        ..., min_length=6, max_length=6, description="[x, y, z, rx, ry, rz]"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    object: ObjectType = ObjectType.OTHER
    object_bbox: list[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] bounding box in pixel coords",
    )
    grasp_type: GraspType = GraspType.PARALLEL_JAW
    grip_force_n: float = Field(5.0, gt=0.0)
    failure_risk: FailureRisk = Field(default_factory=FailureRisk)


class GraspResponse(BaseModel):
    """POST /grasp_plan response body."""

    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    grasp_plan: list[GraspPose] = Field(default_factory=list)
    scene_metadata: SceneMetadata = Field(default_factory=SceneMetadata)
    collision_free: bool = True
    latency_ms: float = 0.0
    model_version: str = "dishspace-pi0-dora-v0.2.0"
    profile_used: str = "default"


# ── Fine-Tune ──


class AdapterConfig(BaseModel):
    """DoRA / LoRA adapter configuration.

    DoRA (Weight-Decomposed Low-Rank Adaptation) is the default. It
    decomposes weights into magnitude + direction and applies low-rank
    updates to the direction component, outperforming standard LoRA by
    1-3% at equal parameter count.
    """

    adapter_type: str = Field("dora", description="'dora' or 'lora'")
    rank: int = Field(16, ge=1, le=128)
    alpha: int = Field(32, ge=1, le=256)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    epochs: int = Field(5, ge=1, le=50)
    learning_rate: float = Field(2e-4, gt=0.0)
    dropout: float = Field(0.05, ge=0.0, le=0.5)


class FineTuneRequest(BaseModel):
    """POST /fine_tune request body."""

    profile_name: str
    base_model: str = "pi0-base"
    training_data_folder: str = Field(..., description="Supabase storage folder path")
    sample_count: int = Field(..., ge=10)
    adapter_config: AdapterConfig = Field(default_factory=AdapterConfig)
    eval_holdout_pct: float = Field(0.15, ge=0.05, le=0.5)


class FineTuneResponse(BaseModel):
    job_id: str = Field(default_factory=lambda: f"ft_{uuid.uuid4().hex[:8]}")
    status: str = "queued"
    estimated_duration_min: int = 180
    estimated_cost_usd: float = 18.50


class FineTuneStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    progress_pct: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 5
    train_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    error_message: Optional[str] = None


# ── Evaluation ──


class EvalRequest(BaseModel):
    """POST /evaluate request body."""

    profile_name: str
    benchmark: str = "sinkbench_v1"
    categories: list[str] = Field(default_factory=list)


class CategoryResult(BaseModel):
    category: str
    scenarios_total: int = 10
    scenarios_passed: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    common_failure: Optional[FailureMode] = None


class EvalResponse(BaseModel):
    profile_name: str
    benchmark: str = "dishbench_v1"
    overall_success_rate: float = 0.0
    categories: list[CategoryResult] = Field(default_factory=list)
    baseline_comparison: Optional[float] = Field(
        None, description="Improvement over baseline in absolute percentage points"
    )


# ── Profile ──


class KitchenProfile(BaseModel):
    """A customer-specific kitchen fine-tune profile."""

    profile_id: str = Field(default_factory=lambda: f"kp_{uuid.uuid4().hex[:8]}")
    name: str
    description: str = ""
    base_model: str = "pi0-base"
    adapter_path: str = ""
    adapter_type: str = "dora"
    training_samples: int = 0
    eval_success_rate: Optional[float] = None
    object_types: list[ObjectType] = Field(default_factory=list)
    rack_type: str = "commercial_grid"
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


# ── ROS Output ──


class ROSPose(BaseModel):
    position: dict[str, float]  # {x, y, z}
    orientation: dict[str, float]  # {x, y, z, w} quaternion


class GripCommand(BaseModel):
    width_mm: float
    force_n: float
    speed: float = 0.1


class ROSTrajectory(BaseModel):
    """MoveIt-compatible trajectory output."""

    header: dict[str, str | float] = Field(
        default_factory=lambda: {"frame_id": "camera", "stamp": time.time()}
    )
    poses: list[ROSPose] = Field(default_factory=list)
    grip_commands: list[GripCommand] = Field(default_factory=list)
