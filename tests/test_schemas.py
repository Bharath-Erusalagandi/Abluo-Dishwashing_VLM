"""Tests for Pydantic schemas — validation, serialization, defaults."""

import pytest
from src.models.schemas import (
    GraspAnnotation,
    GraspRequest,
    GraspResponse,
    GraspOptions,
    FineTuneRequest,
    EvalRequest,
    EvalResponse,
    KitchenProfile,
    AdapterConfig,
    ObjectType,
    ObjectMaterial,
    GraspType,
    DataSource,
    FailureMode,
    EnvironmentConditions,
    RobotConfig,
    SceneType,
    TaskTarget,
)


class TestGraspAnnotation:
    """GraspAnnotation model tests."""

    def test_minimal_valid(self):
        ann = GraspAnnotation(
            object_type=ObjectType.MUG,
            object_material=ObjectMaterial.CERAMIC_DRY,
            grasp_point_xyz=[0.1, 0.2, 0.3],
            success=True,
            source=DataSource.MUJOCO_SIM,
        )
        assert ann.object_type == ObjectType.MUG
        assert ann.success is True
        assert ann.sample_id is not None
        assert ann.sample_id.startswith("gs_")

    def test_full_annotation(self):
        ann = GraspAnnotation(
            object_type=ObjectType.PLATE,
            object_material=ObjectMaterial.GLASS,
            grasp_point_xyz=[0.0, 0.0, 0.0],
            approach_vector=[0.0, 0.0, -1.0],
            success=False,
            failure_mode=FailureMode.SLIP,
            failure_detail="slipped due to wet surface",
            source=DataSource.PILOT,
            grip_width_mm=85.0,
            grip_force_n=5.0,
            environment=EnvironmentConditions(
                wet=True,
                soap=True,
                steam=False,
                lighting="dim",
            ),
            robot_config=RobotConfig(arm="UR5", gripper="Robotiq_2F85"),
        )
        assert ann.success is False
        assert ann.failure_mode == FailureMode.SLIP
        assert ann.environment.soap is True

    def test_grasp_point_length_validation(self):
        """grasp_point_xyz must have exactly 3 elements."""
        with pytest.raises(Exception):
            GraspAnnotation(
                object_type=ObjectType.MUG,
                object_material=ObjectMaterial.CERAMIC_DRY,
                grasp_point_xyz=[0.1, 0.2],  # only 2 — should fail
                success=True,
                source=DataSource.MUJOCO_SIM,
            )

    def test_serialization_roundtrip(self):
        ann = GraspAnnotation(
            object_type=ObjectType.BOWL,
            object_material=ObjectMaterial.PLASTIC,
            grasp_point_xyz=[0.1, 0.2, 0.3],
            success=True,
            source=DataSource.MUJOCO_SIM,
        )
        data = ann.model_dump(mode="json")
        restored = GraspAnnotation(**data)
        assert restored.object_type == ann.object_type
        assert restored.sample_id == ann.sample_id

    def test_defaults(self):
        ann = GraspAnnotation(grasp_point_xyz=[0.0, 0.0, 0.0])
        assert ann.source == DataSource.MUJOCO_SIM
        assert ann.object_type == ObjectType.OTHER
        assert ann.object_material == ObjectMaterial.CERAMIC_DRY
        assert ann.failure_mode == FailureMode.NONE
        assert ann.success is False
        assert ann.grip_width_mm == 80.0
        assert ann.grip_force_n == 5.0
        assert ann.environment.scene_type == SceneType.SINK_SINGLE
        assert ann.environment.target_zone == TaskTarget.DRYING_RACK


class TestGraspRequest:
    """GraspRequest validation tests."""

    def test_minimal_request(self):
        req = GraspRequest(image_base64="abc123")
        assert req.image_base64 == "abc123"
        assert req.kitchen_profile == "default"
        assert req.robot == "UR5_realsense"

    def test_with_options(self):
        req = GraspRequest(
            image_base64="abc123",
            kitchen_profile="commercial_rack",
            robot="Franka_realsense",
            options=GraspOptions(
                collision_check=True,
                max_grasps=3,
                min_confidence=0.7,
                coordinate_frame="robot_base",
            ),
        )
        assert req.options.max_grasps == 3

    def test_default_options(self):
        req = GraspRequest(image_base64="test")
        assert req.options.collision_check is True
        assert req.options.max_grasps == 5
        assert req.options.min_confidence == 0.7


class TestKitchenProfile:
    """KitchenProfile tests."""

    def test_defaults(self):
        profile = KitchenProfile(name="test")
        assert profile.rack_type == "commercial_grid"
        assert profile.base_model == "pi0-base"
        assert profile.training_samples == 0
        assert profile.profile_id.startswith("kp_")

    def test_custom_profile(self):
        profile = KitchenProfile(
            name="industrial",
            description="High-throughput dishwashing line",
            rack_type="conveyor",
            object_types=[ObjectType.PLATE, ObjectType.TRAY],
        )
        assert len(profile.object_types) == 2
        assert profile.rack_type == "conveyor"


class TestFineTuneRequest:
    """FineTuneRequest tests."""

    def test_required_fields(self):
        req = FineTuneRequest(
            profile_name="test",
            training_data_folder="data/training/v1",
            sample_count=500,
        )
        assert req.base_model == "pi0-base"
        assert req.adapter_config.rank == 16
        assert req.adapter_config.epochs == 5

    def test_custom_lora(self):
        req = FineTuneRequest(
            profile_name="custom",
            training_data_folder="data/training/v2",
            sample_count=1000,
            adapter_config=AdapterConfig(rank=32, alpha=64, epochs=10),
        )
        assert req.adapter_config.rank == 32
        assert req.adapter_config.epochs == 10


class TestEvalRequest:
    """EvalRequest tests."""

    def test_defaults(self):
        req = EvalRequest(profile_name="test-profile")
        assert req.benchmark == "sinkbench_v1"
        assert req.categories == []

    def test_custom_categories(self):
        req = EvalRequest(
            profile_name="test",
            categories=["wet_ceramics", "transparent_glass"],
        )
        assert len(req.categories) == 2
