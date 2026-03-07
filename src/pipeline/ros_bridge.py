"""ROS trajectory serialization.

Converts DishSpace grasp poses to MoveIt-compatible trajectory JSON
that can be consumed directly by roslibpy or ROS 2 nodes.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from src.models.schemas import (
    CoordinateFrame,
    GraspPose,
    GripCommand,
    ROSPose,
    ROSTrajectory,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> dict[str, float]:
    """Convert Roll-Pitch-Yaw (radians) to quaternion {x, y, z, w}.

    Uses ZYX Euler convention (standard in ROS).
    """
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    return {
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
        "w": cr * cp * cy + sr * sp * sy,
    }


def quaternion_to_rpy(
    q_or_x: dict[str, float] | float,
    y: float | None = None,
    z: float | None = None,
    w: float | None = None,
) -> tuple[float, float, float]:
    """Convert quaternion to (roll, pitch, yaw) radians.

    Accepts either:
      - A single dict ``{x, y, z, w}``
      - Four positional floats ``(x, y, z, w)``
    """
    if isinstance(q_or_x, dict):
        x_val = q_or_x["x"]
        y_val = q_or_x["y"]
        z_val = q_or_x["z"]
        w_val = q_or_x["w"]
    else:
        assert y is not None and z is not None and w is not None
        x_val, y_val, z_val, w_val = q_or_x, y, z, w

    sinr_cosp = 2 * (w_val * x_val + y_val * z_val)
    cosr_cosp = 1 - 2 * (x_val * x_val + y_val * y_val)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w_val * y_val - z_val * x_val)
    pitch = math.asin(max(-1, min(1, sinp)))

    siny_cosp = 2 * (w_val * z_val + x_val * y_val)
    cosy_cosp = 1 - 2 * (y_val * y_val + z_val * z_val)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def transform_pose_camera_to_robot(
    pose_camera: list[float],
    camera_extrinsics: Optional[np.ndarray] = None,
) -> list[float]:
    """Transform a pose from camera frame to robot base frame.

    Args:
        pose_camera: [x, y, z, rx, ry, rz] in camera frame.
        camera_extrinsics: 4x4 transformation matrix (camera → robot base).
            If None, uses a default eye-in-hand mount estimate.

    Returns:
        [x, y, z, rx, ry, rz] in robot base frame.
    """
    if camera_extrinsics is None:
        # Default eye-in-hand mount (camera looking down, 0.8m above table)
        camera_extrinsics = np.array([
            [1, 0, 0, 0.0],
            [0, -1, 0, 0.0],
            [0, 0, -1, 0.8],
            [0, 0, 0, 1.0],
        ], dtype=np.float64)

    # Position transform
    pos = np.array([pose_camera[0], pose_camera[1], pose_camera[2], 1.0])
    pos_robot = camera_extrinsics @ pos

    # For orientation, we'd need full rotation composition
    # Simplified: just return the orientation as-is for MVP
    return [
        float(pos_robot[0]),
        float(pos_robot[1]),
        float(pos_robot[2]),
        pose_camera[3],
        pose_camera[4],
        pose_camera[5],
    ]


def grasps_to_ros_trajectory(
    grasps: list[GraspPose],
    frame: CoordinateFrame = CoordinateFrame.CAMERA,
    camera_extrinsics: Optional[np.ndarray] = None,
) -> ROSTrajectory:
    """Convert DishSpace grasp poses to MoveIt-compatible trajectory.

    Args:
        grasps: List of predicted grasp poses.
        frame: Target coordinate frame for output.
        camera_extrinsics: Camera-to-robot-base transform (4x4).

    Returns:
        ROSTrajectory ready for roslibpy or direct ROS consumption.
    """
    poses: list[ROSPose] = []
    grip_commands: list[GripCommand] = []

    for g in grasps:
        pose = g.pose  # [x, y, z, rx, ry, rz]

        # Transform if needed
        if frame == CoordinateFrame.ROBOT_BASE:
            pose = transform_pose_camera_to_robot(pose, camera_extrinsics)

        # Position
        position = {"x": pose[0], "y": pose[1], "z": pose[2]}

        # Orientation (RPY → quaternion)
        orientation = rpy_to_quaternion(pose[3], pose[4], pose[5])

        poses.append(ROSPose(position=position, orientation=orientation))

        # Gripper command
        grip_commands.append(GripCommand(
            width_mm=85.0,  # default Robotiq 2F-85 opening
            force_n=g.grip_force_n,
            speed=0.1,
        ))

    frame_id = {
        CoordinateFrame.CAMERA: "camera_color_optical_frame",
        CoordinateFrame.WORLD: "world",
        CoordinateFrame.ROBOT_BASE: "base_link",
    }.get(frame, "camera_color_optical_frame")

    trajectory = ROSTrajectory(
        header={"frame_id": frame_id, "stamp": time.time()},
        poses=poses,
        grip_commands=grip_commands,
    )

    log.info(
        "ros_trajectory_created",
        frame=frame_id,
        num_poses=len(poses),
    )
    return trajectory


def trajectory_to_moveit_json(trajectory: ROSTrajectory) -> dict:
    """Convert ROSTrajectory to a dict matching MoveIt's expected format.

    This can be published via roslibpy to /move_group/goal.
    """
    waypoints = []
    for pose, grip in zip(trajectory.poses, trajectory.grip_commands):
        waypoints.append({
            "pose": {
                "position": pose.position,
                "orientation": pose.orientation,
            },
            "gripper": {
                "position": grip.width_mm / 1000.0,  # convert to meters
                "max_effort": grip.force_n,
            },
        })

    return {
        "header": {
            "frame_id": trajectory.header.get("frame_id", "base_link"),
            "stamp": {
                "secs": int(trajectory.header.get("stamp", 0)),
                "nsecs": 0,
            },
        },
        "waypoints": waypoints,
        "max_velocity_scaling": 0.3,
        "max_acceleration_scaling": 0.3,
        "planner_id": "RRTConnect",
        "planning_time": 5.0,
    }


def publish_to_ros(
    trajectory: ROSTrajectory,
    ros_host: str = "localhost",
    ros_port: int = 9090,
    topic: str = "/dishspace/grasp_trajectory",
) -> bool:
    """Publish trajectory to a ROS system via roslibpy websocket bridge.

    Requires rosbridge_server running on the robot.

    Args:
        trajectory: The trajectory to publish.
        ros_host: ROS bridge websocket host.
        ros_port: ROS bridge websocket port.
        topic: ROS topic to publish to.

    Returns:
        True if successfully published.
    """
    try:
        import roslibpy

        client = roslibpy.Ros(host=ros_host, port=ros_port)
        client.run()

        publisher = roslibpy.Topic(
            client,
            topic,
            "std_msgs/String",  # simplified — would use custom msg type
        )

        import json
        msg_data = json.dumps(trajectory_to_moveit_json(trajectory))
        publisher.publish(roslibpy.Message({"data": msg_data}))

        log.info("ros_published", topic=topic, host=ros_host)
        publisher.unadvertise()
        client.terminate()
        return True

    except Exception as e:
        log.warning("ros_publish_failed", error=str(e))
        return False
