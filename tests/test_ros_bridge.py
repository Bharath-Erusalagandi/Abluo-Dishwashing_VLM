"""Tests for ROS bridge / trajectory serialization."""

import math
import pytest
from src.models.schemas import (
    CoordinateFrame,
    GraspPose,
    GraspType,
    FailureRisk,
    ObjectType,
)
from src.pipeline.ros_bridge import (
    rpy_to_quaternion,
    quaternion_to_rpy,
    grasps_to_ros_trajectory,
    trajectory_to_moveit_json,
)


class TestQuaternionConversion:
    """Test RPY ↔ quaternion round-trip."""

    def test_identity(self):
        q = rpy_to_quaternion(0.0, 0.0, 0.0)
        assert isinstance(q, dict)
        assert set(q.keys()) == {"x", "y", "z", "w"}
        assert abs(q["w"] - 1.0) < 1e-6

    def test_roundtrip(self):
        roll, pitch, yaw = 0.1, 0.2, 0.3
        q = rpy_to_quaternion(roll, pitch, yaw)
        r2, p2, y2 = quaternion_to_rpy(q)
        assert abs(r2 - roll) < 1e-6
        assert abs(p2 - pitch) < 1e-6
        assert abs(y2 - yaw) < 1e-6

    def test_90_deg_rotation(self):
        q = rpy_to_quaternion(0.0, 0.0, math.pi / 2)
        r, p, y = quaternion_to_rpy(q)
        assert abs(y - math.pi / 2) < 1e-6

    def test_unit_quaternion(self):
        """Quaternion should have unit norm."""
        q = rpy_to_quaternion(0.5, 0.3, 0.7)
        norm = math.sqrt(q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2 + q["w"] ** 2)
        assert abs(norm - 1.0) < 1e-6


class TestTrajectoryGeneration:
    """Test grasp-to-trajectory conversion."""

    def _make_grasp(self, pose, confidence=0.9, obj=ObjectType.MUG):
        return GraspPose(
            pose=pose,
            confidence=confidence,
            object=obj,
            object_bbox=[100, 100, 200, 200],
            grasp_type=GraspType.PARALLEL_JAW,
            grip_force_n=5.0,
            failure_risk=FailureRisk(),
        )

    def test_basic_trajectory(self):
        grasps = [self._make_grasp([0.3, 0.1, 0.05, 0.0, 1.57, 0.0])]
        traj = grasps_to_ros_trajectory(grasps)
        assert traj is not None
        assert len(traj.poses) == 1
        assert len(traj.grip_commands) == 1

    def test_multiple_grasps(self):
        grasps = [
            self._make_grasp([0.3, 0.1, 0.05, 0.0, 1.57, 0.0]),
            self._make_grasp([0.5, -0.1, 0.02, 0.0, 1.57, 0.0], obj=ObjectType.PLATE),
        ]
        traj = grasps_to_ros_trajectory(grasps)
        assert len(traj.poses) == 2
        assert len(traj.grip_commands) == 2

    def test_moveit_json_format(self):
        grasps = [self._make_grasp([0.4, 0.0, 0.02, 0.0, 1.57, 0.0])]
        traj = grasps_to_ros_trajectory(grasps)
        moveit_json = trajectory_to_moveit_json(traj)
        assert "header" in moveit_json
        assert "waypoints" in moveit_json
        assert "planner_id" in moveit_json
        assert len(moveit_json["waypoints"]) == 1

    def test_robot_base_frame(self):
        grasps = [self._make_grasp([0.3, 0.1, 0.5, 0.0, 0.0, 0.0])]
        traj = grasps_to_ros_trajectory(grasps, frame=CoordinateFrame.ROBOT_BASE)
        assert traj.header["frame_id"] == "base_link"
