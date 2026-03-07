"""MuJoCo synthetic grasp data generator.

Generates labeled grasp attempts on kitchen objects for training data.
Uses MuJoCo physics simulation to determine grasp success/failure
with realistic contact dynamics.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.models.schemas import (
    DataSource,
    EnvironmentConditions,
    FailureMode,
    GraspAnnotation,
    ObjectMaterial,
    ObjectType,
    RobotConfig,
    SceneType,
    TaskTarget,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


def _pick(rng: np.random.Generator, items: list):
    """Pick a random item from *items* without numpy coercing enum types.

    ``rng.choice()`` converts ``str, Enum`` members into ``np.str_``
    with garbled values on NumPy ≥ 2 / Python ≥ 3.14.  Working around
    by indexing with ``rng.integers()`` instead.
    """
    return items[int(rng.integers(len(items)))]


# Kitchen object physical properties
KITCHEN_OBJECTS = {
    ObjectType.MUG: {
        "size": [0.08, 0.08, 0.10],  # x, y, z in meters
        "mass": 0.35,
        "friction_dry": 0.7,
        "friction_wet": 0.3,
        "fragile": False,
        "has_handle": True,
        "grasp_types": ["rim_pinch", "handle_wrap", "body_wrap"],
    },
    ObjectType.PLATE: {
        "size": [0.25, 0.25, 0.02],
        "mass": 0.45,
        "friction_dry": 0.6,
        "friction_wet": 0.25,
        "fragile": False,
        "has_handle": False,
        "grasp_types": ["edge_pinch", "flat_suction"],
    },
    ObjectType.BOWL: {
        "size": [0.16, 0.16, 0.08],
        "mass": 0.40,
        "friction_dry": 0.65,
        "friction_wet": 0.28,
        "fragile": False,
        "has_handle": False,
        "grasp_types": ["rim_pinch", "body_wrap"],
    },
    ObjectType.WINE_GLASS: {
        "size": [0.07, 0.07, 0.20],
        "mass": 0.18,
        "friction_dry": 0.5,
        "friction_wet": 0.15,
        "fragile": True,
        "has_handle": False,
        "grasp_types": ["stem_pinch", "body_wrap"],
    },
    ObjectType.TUMBLER: {
        "size": [0.07, 0.07, 0.12],
        "mass": 0.25,
        "friction_dry": 0.55,
        "friction_wet": 0.2,
        "fragile": True,
        "has_handle": False,
        "grasp_types": ["body_wrap", "rim_pinch"],
    },
    ObjectType.FORK: {
        "size": [0.02, 0.02, 0.20],
        "mass": 0.05,
        "friction_dry": 0.4,
        "friction_wet": 0.15,
        "fragile": False,
        "has_handle": True,
        "grasp_types": ["parallel_jaw"],
    },
    ObjectType.KNIFE: {
        "size": [0.02, 0.02, 0.22],
        "mass": 0.07,
        "friction_dry": 0.4,
        "friction_wet": 0.12,
        "fragile": False,
        "has_handle": True,
        "grasp_types": ["parallel_jaw"],
    },
    ObjectType.SPOON: {
        "size": [0.03, 0.03, 0.18],
        "mass": 0.04,
        "friction_dry": 0.45,
        "friction_wet": 0.18,
        "fragile": False,
        "has_handle": True,
        "grasp_types": ["parallel_jaw"],
    },
    ObjectType.SPATULA: {
        "size": [0.08, 0.02, 0.30],
        "mass": 0.10,
        "friction_dry": 0.5,
        "friction_wet": 0.2,
        "fragile": False,
        "has_handle": True,
        "grasp_types": ["parallel_jaw", "handle_wrap"],
    },
    ObjectType.POT: {
        "size": [0.20, 0.20, 0.15],
        "mass": 1.2,
        "friction_dry": 0.5,
        "friction_wet": 0.2,
        "fragile": False,
        "has_handle": True,
        "grasp_types": ["handle_wrap", "rim_pinch"],
    },
    ObjectType.PAN: {
        "size": [0.25, 0.25, 0.05],
        "mass": 0.9,
        "friction_dry": 0.5,
        "friction_wet": 0.2,
        "fragile": False,
        "has_handle": True,
        "grasp_types": ["handle_wrap"],
    },
}

LIGHTING_CONDITIONS = [
    "overhead_bright",
    "overhead_dim",
    "side_bright",
    "specular_overhead",
    "mixed",
]

RACK_TYPES = ["none", "commercial_grid", "home_dishwasher", "drying_rack", "conveyor"]


@dataclass
class SimConfig:
    """Configuration for a single simulation run."""

    object_type: ObjectType
    material: ObjectMaterial
    is_wet: bool
    has_soap: bool
    has_steam: bool
    lighting: str
    rack_type: str
    scene_type: SceneType
    target_zone: TaskTarget
    visible_object_count: int
    occlusion_level: float
    object_tilt_deg: float  # 0 = upright
    stacking: int  # 1 = single, 2+ = stacked
    gripper_noise_mm: float  # positional noise in mm


SCENE_TEMPLATES = {
    SceneType.ISOLATED: {
        "visible_object_count": [1],
        "occlusion_range": (0.0, 0.05),
        "stacking": [1],
        "target_zones": [TaskTarget.DRYING_RACK],
        "rack_type": "none",
    },
    SceneType.SINK_SINGLE: {
        "visible_object_count": [1, 1, 2],
        "occlusion_range": (0.0, 0.15),
        "stacking": [1, 1, 2],
        "target_zones": [TaskTarget.DRYING_RACK, TaskTarget.DISHWASHER_TOP_RACK],
        "rack_type": "none",
    },
    SceneType.SINK_CLUTTERED: {
        "visible_object_count": [4, 5, 6, 7, 8],
        "occlusion_range": (0.3, 0.8),
        "stacking": [2, 2, 3],
        "target_zones": [TaskTarget.DRYING_RACK, TaskTarget.DISHWASHER_TOP_RACK, TaskTarget.DISHWASHER_BOTTOM_RACK],
        "rack_type": "none",
    },
    SceneType.UTENSIL_BUNDLE: {
        "visible_object_count": [5, 6, 7, 8],
        "occlusion_range": (0.4, 0.9),
        "stacking": [2, 3],
        "target_zones": [TaskTarget.UTENSIL_CADDY],
        "rack_type": "none",
    },
    SceneType.DRYING_RACK: {
        "visible_object_count": [2, 3, 4, 5],
        "occlusion_range": (0.2, 0.6),
        "stacking": [1, 2, 2],
        "target_zones": [TaskTarget.DISHWASHER_TOP_RACK, TaskTarget.DISHWASHER_BOTTOM_RACK],
        "rack_type": "drying_rack",
    },
    SceneType.DISHWASHER_RACK: {
        "visible_object_count": [4, 5, 6],
        "occlusion_range": (0.25, 0.7),
        "stacking": [1, 2],
        "target_zones": [TaskTarget.DISHWASHER_TOP_RACK, TaskTarget.DISHWASHER_BOTTOM_RACK],
        "rack_type": "home_dishwasher",
    },
}


def _sample_scene_config(
    rng: np.random.Generator,
    object_type: ObjectType,
) -> tuple[SceneType, int, float, int, TaskTarget, str]:
    """Sample a sink-scene template for the current object.

    Keeps utensils biased toward cluttered bundle scenes and larger items
    biased toward sink / rack transfer scenes.
    """
    if object_type in (ObjectType.FORK, ObjectType.KNIFE, ObjectType.SPOON, ObjectType.SPATULA):
        scene_type = _pick(rng, [SceneType.UTENSIL_BUNDLE, SceneType.SINK_CLUTTERED, SceneType.SINK_SINGLE])
    elif object_type in (ObjectType.POT, ObjectType.PAN, ObjectType.PLATE, ObjectType.BOWL):
        scene_type = _pick(rng, [SceneType.SINK_CLUTTERED, SceneType.DRYING_RACK, SceneType.SINK_SINGLE])
    else:
        scene_type = _pick(rng, list(SCENE_TEMPLATES.keys()))

    template = SCENE_TEMPLATES[scene_type]
    visible_object_count = int(_pick(rng, template["visible_object_count"]))
    occlusion_low, occlusion_high = template["occlusion_range"]
    occlusion_level = float(rng.uniform(occlusion_low, occlusion_high))
    stacking = int(_pick(rng, template["stacking"]))
    target_zone = _pick(rng, template["target_zones"])
    rack_type = template["rack_type"] or _pick(rng, RACK_TYPES)

    return scene_type, visible_object_count, occlusion_level, stacking, target_zone, rack_type


def _compute_grasp_success(
    obj_props: dict,
    sim_config: SimConfig,
    grasp_offset_mm: float,
    grip_force_n: float,
    rng: np.random.Generator,
) -> tuple[bool, FailureMode, str]:
    """Physics-inspired grasp success model.

    Not a full MuJoCo simulation yet — uses analytical contact model
    for fast batch generation. Replace with MuJoCo forward sim in Week 2.

    Returns:
        (success, failure_mode, detail_string)
    """
    # Base friction
    friction = obj_props["friction_wet"] if sim_config.is_wet else obj_props["friction_dry"]
    if sim_config.has_soap:
        friction *= 0.5

    # Required force to hold object
    gravity_force = obj_props["mass"] * 9.81
    required_grip = gravity_force / (2 * friction)

    # Tilt penalty — tilted objects are harder
    tilt_rad = math.radians(sim_config.object_tilt_deg)
    tilt_factor = 1.0 + 0.5 * abs(math.sin(tilt_rad))
    required_grip *= tilt_factor

    # Stacking penalty — more objects = more collision risk
    collision_prob = 0.0
    if sim_config.stacking > 1:
        collision_prob = 0.15 * (sim_config.stacking - 1)
    collision_prob += min(0.25, 0.03 * max(sim_config.visible_object_count - 1, 0))
    collision_prob += 0.20 * sim_config.occlusion_level

    # Gripper noise penalty
    offset_penalty = grasp_offset_mm / 50.0  # 50mm offset = total failure

    # Fragile object penalty — too much force breaks it
    if obj_props["fragile"] and grip_force_n > 8.0:
        return False, FailureMode.FRAGILE_BREAK, (
            f"grip force {grip_force_n:.1f}N exceeded fragile threshold 8.0N"
        )

    # Check collision
    if rng.random() < collision_prob:
        return False, FailureMode.COLLISION, (
            f"collision with adjacent object in stack of {sim_config.stacking}"
        )

    # Soap occlusion check
    if sim_config.has_soap and rng.random() < 0.20:
        return False, FailureMode.SOAP, "soap suds occluded grasp contact surface"

    # Depth hole (transparent / reflective)
    if sim_config.material in (ObjectMaterial.GLASS, ObjectMaterial.METAL):
        depth_hole_prob = 0.25 if sim_config.is_wet else 0.15
        if rng.random() < depth_hole_prob:
            return False, FailureMode.DEPTH_HOLE, (
                f"depth sensor failed on {sim_config.material.value} surface"
            )

    # Steam occlusion
    if sim_config.has_steam and rng.random() < 0.12:
        return False, FailureMode.OCCLUSION, "steam obscured camera view"

    # Grip force vs required force (slip check)
    effective_grip = grip_force_n * (1.0 - offset_penalty) + rng.normal(0, 0.5)
    if effective_grip < required_grip:
        slip_speed = (required_grip - effective_grip) * 3.0
        return False, FailureMode.SLIP, (
            f"wet surface reduced friction to {friction:.2f}, "
            f"object slipped at {slip_speed:.1f}mm/s"
        )

    return True, FailureMode.NONE, ""


def generate_synthetic_sample(
    rng: np.random.Generator,
    object_type: Optional[ObjectType] = None,
) -> GraspAnnotation:
    """Generate one synthetic grasp annotation with randomized conditions.

    Uses analytical physics model for fast generation.
    """
    # Random object type if not specified
    if object_type is None:
        object_type = _pick(rng, list(KITCHEN_OBJECTS.keys()))

    obj_props = KITCHEN_OBJECTS[object_type]

    # Randomize conditions
    is_wet = rng.random() < 0.4
    has_soap = is_wet and rng.random() < 0.3
    has_steam = rng.random() < 0.15

    material_choices = {
        True: [ObjectMaterial.CERAMIC_WET],
        False: [ObjectMaterial.CERAMIC_DRY, ObjectMaterial.GLASS, ObjectMaterial.METAL,
                ObjectMaterial.PLASTIC, ObjectMaterial.WOOD],
    }
    if object_type in (ObjectType.WINE_GLASS, ObjectType.TUMBLER, ObjectType.MEASURING_CUP):
        material = ObjectMaterial.GLASS
    elif object_type in (ObjectType.FORK, ObjectType.KNIFE, ObjectType.SPOON):
        material = ObjectMaterial.METAL
    elif is_wet:
        material = ObjectMaterial.CERAMIC_WET
    else:
        material = _pick(rng, material_choices[False])

    sim_config = SimConfig(
        object_type=object_type,
        material=material,
        is_wet=is_wet,
        has_soap=has_soap,
        has_steam=has_steam,
        lighting=_pick(rng, LIGHTING_CONDITIONS),
        rack_type="none",
        scene_type=SceneType.SINK_SINGLE,
        target_zone=TaskTarget.DRYING_RACK,
        visible_object_count=1,
        occlusion_level=0.0,
        object_tilt_deg=_pick(rng, [0, 0, 0, 15, 30, 45]),  # weighted toward upright
        stacking=1,
        gripper_noise_mm=abs(rng.normal(0, 5)),
    )

    (
        sim_config.scene_type,
        sim_config.visible_object_count,
        sim_config.occlusion_level,
        sim_config.stacking,
        sim_config.target_zone,
        sim_config.rack_type,
    ) = _sample_scene_config(rng, object_type)

    # Random grasp point (relative to object center)
    size = obj_props["size"]
    grasp_point = [
        float(rng.normal(0, size[0] * 0.3)),
        float(rng.normal(0, size[1] * 0.3)),
        float(rng.uniform(0, size[2])),
    ]

    grip_force = float(rng.uniform(2.0, 12.0))
    grip_width = float(max(size[0], size[1]) * 1000 * rng.uniform(0.8, 1.3))

    # Simulate
    grasp_offset = np.linalg.norm(grasp_point[:2]) * 1000  # mm
    success, failure_mode, detail = _compute_grasp_success(
        obj_props, sim_config, grasp_offset, grip_force, rng
    )

    return GraspAnnotation(
        sample_id=f"syn_{uuid.uuid4().hex[:8]}",
        source=DataSource.MUJOCO_SIM,
        object_type=object_type,
        object_material=material,
        grasp_point_xyz=grasp_point,
        grasp_normal=[0.0, 0.0, 1.0],
        approach_vector=[0.0, -1.0, 0.0],
        grip_width_mm=grip_width,
        grip_force_n=grip_force,
        success=success,
        failure_mode=failure_mode,
        failure_detail=detail,
        environment=EnvironmentConditions(
            wet=is_wet,
            soap=has_soap,
            steam=has_steam,
            lighting=sim_config.lighting,
            rack_type=sim_config.rack_type,
            scene_type=sim_config.scene_type,
            visible_object_count=sim_config.visible_object_count,
            occlusion_level=sim_config.occlusion_level,
            target_zone=sim_config.target_zone,
        ),
        robot_config=RobotConfig(),
    )


def generate_balanced_batch(
    count: int = 5000,
    seed: int = 42,
    object_types: Optional[list[ObjectType]] = None,
) -> list[GraspAnnotation]:
    """Generate a more evenly covered synthetic dataset.

    Cycles through object types instead of fully random sampling so small runs
    still cover the major kitchen object families.
    """
    rng = np.random.default_rng(seed)
    annotations = []
    choices = object_types or list(KITCHEN_OBJECTS.keys())

    for index in range(count):
        obj_type = choices[index % len(choices)]
        annotations.append(generate_synthetic_sample(rng, obj_type))

        if (index + 1) % 1000 == 0:
            log.info("synthetic_data_progress", generated=index + 1, total=count)

    success_count = sum(1 for a in annotations if a.success)
    object_distribution: dict[str, int] = {}
    for ann in annotations:
        object_distribution[ann.object_type.value] = object_distribution.get(ann.object_type.value, 0) + 1

    log.info(
        "synthetic_balanced_batch_complete",
        total=count,
        success_rate=f"{success_count / max(count, 1):.1%}",
        object_distribution=object_distribution,
    )

    return annotations


def generate_batch(
    count: int = 5000,
    seed: int = 42,
    object_types: Optional[list[ObjectType]] = None,
) -> list[GraspAnnotation]:
    """Generate a batch of synthetic grasp annotations.

    Args:
        count: Number of samples to generate.
        seed: Random seed for reproducibility.
        object_types: If provided, only generate these object types.

    Returns:
        List of GraspAnnotation.
    """
    rng = np.random.default_rng(seed)
    annotations = []

    for i in range(count):
        if object_types:
            obj_type = _pick(rng, object_types)
        else:
            obj_type = None

        annotations.append(generate_synthetic_sample(rng, obj_type))

        if (i + 1) % 1000 == 0:
            log.info("synthetic_data_progress", generated=i + 1, total=count)

    # Log distribution
    success_count = sum(1 for a in annotations if a.success)
    failure_counts: dict[str, int] = {}
    for a in annotations:
        mode = a.failure_mode.value
        failure_counts[mode] = failure_counts.get(mode, 0) + 1

    log.info(
        "synthetic_batch_complete",
        total=count,
        success_rate=f"{success_count / count:.1%}",
        failure_distribution=failure_counts,
    )

    return annotations
