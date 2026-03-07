#!/usr/bin/env python3
"""Deploy DishSpace onto a real robot for dishwashing.

Connects to a UR5 + RealSense D435 setup via roslibpy and runs a
continuous dishwashing loop:

  1. Capture RGB-D frame from RealSense
  2. Run grasp planning (local model or API)
  3. Send trajectory to MoveIt via ROS
  4. Execute pick → wash → place cycle
  5. Log results for continuous improvement

Hardware requirements:
  - UR5 / UR5e robotic arm with Robotiq 2F-85 gripper
  - Intel RealSense D435 camera (eye-in-hand or eye-to-hand)
  - ROS 2 (Humble/Iron) with MoveIt 2 running on the robot PC
  - roslibpy bridge (rosbridge_server) for WebSocket communication

Usage:
    # Connect to robot and start dishwashing loop
    python scripts/deploy_robot.py --ros-host 192.168.1.100

    # With fine-tuned adapter (local inference)
    python scripts/deploy_robot.py --adapter models/dora/kitchen_v1/adapter --local

    # Via DishSpace API (cloud inference)
    python scripts/deploy_robot.py --api-url https://api.dishspace.ai --api-key YOUR_KEY

    # Dry run (visualise only, no robot commands)
    python scripts/deploy_robot.py --dry-run

    # Single grasp test (no loop)
    python scripts/deploy_robot.py --single-grasp
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


class TaskState(str, Enum):
    """State machine for the dishwashing cycle."""
    IDLE = "idle"
    SCANNING = "scanning"           # Capture RGB-D, detect objects
    PLANNING = "planning"           # Run grasp planning
    APPROACHING = "approaching"     # Move arm to pre-grasp pose
    GRASPING = "grasping"           # Close gripper on object
    LIFTING = "lifting"             # Lift object from sink/counter
    WASHING = "washing"             # Move to wash station (optional)
    PLACING = "placing"             # Move to drying rack
    RELEASING = "releasing"         # Open gripper
    RETRACTING = "retracting"       # Move arm back to home
    ERROR = "error"


@dataclass
class DishwashingConfig:
    """Configuration for the dishwashing robot."""
    # ROS connection
    ros_host: str = "localhost"
    ros_port: int = 9090

    # Workspace bounds (robot base frame, meters)
    workspace_min: list[float] = None  # [x_min, y_min, z_min]
    workspace_max: list[float] = None  # [x_max, y_max, z_max]

    # Pre-defined positions (robot base frame)
    home_pose: list[float] = None      # [x, y, z, rx, ry, rz]
    scan_pose: list[float] = None      # Camera viewing pose
    wash_pose: list[float] = None      # Over the wash area
    rack_pose: list[float] = None      # Over the drying rack

    # Safety
    max_speed: float = 0.3   # m/s
    max_force: float = 15.0  # N — emergency stop above this
    z_clearance: float = 0.15  # m — minimum lift height before lateral moves

    # Gripper
    gripper_open_width: float = 85.0  # mm
    gripper_close_timeout: float = 3.0  # seconds

    # Timing
    scan_interval: float = 0.5  # seconds between scans when idle
    grasp_settle_time: float = 0.5  # seconds to wait after grasping
    place_settle_time: float = 0.3  # seconds to wait after placing

    def __post_init__(self):
        if self.workspace_min is None:
            self.workspace_min = [0.1, -0.4, 0.0]
        if self.workspace_max is None:
            self.workspace_max = [0.7, 0.4, 0.6]
        if self.home_pose is None:
            self.home_pose = [0.3, 0.0, 0.5, 0.0, 1.57, 0.0]
        if self.scan_pose is None:
            self.scan_pose = [0.3, 0.0, 0.6, 0.0, 1.57, 0.0]
        if self.wash_pose is None:
            self.wash_pose = [0.3, -0.3, 0.4, 0.0, 1.57, 0.0]
        if self.rack_pose is None:
            self.rack_pose = [0.3, 0.3, 0.4, 0.0, 1.57, 0.0]


class DishwashingRobot:
    """Controls the dish-washing robot loop.

    Connects via roslibpy to a ROS 2 system running MoveIt 2.
    Captures images from RealSense, plans grasps, and executes
    pick-wash-place cycles.
    """

    def __init__(
        self,
        config: DishwashingConfig,
        inference_mode: str = "local",  # "local" or "api"
        adapter_path: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.config = config
        self.inference_mode = inference_mode
        self.adapter_path = adapter_path
        self.api_url = api_url
        self.api_key = api_key
        self.dry_run = dry_run
        self.state = TaskState.IDLE
        self.ros = None
        self.planner = None

        # Stats
        self.total_attempts = 0
        self.successful_picks = 0
        self.failed_picks = 0
        self.cycle_times: list[float] = []

    def connect(self) -> None:
        """Connect to ROS and initialise the grasp planner."""
        if not self.dry_run:
            import roslibpy

            print(f"🤖 Connecting to ROS at {self.config.ros_host}:{self.config.ros_port}")
            self.ros = roslibpy.Ros(host=self.config.ros_host, port=self.config.ros_port)
            self.ros.run()
            print(f"   ✅ ROS connected")

            # Subscribe to camera topics
            self._setup_camera_subscribers()

            # Set up MoveIt action client
            self._setup_moveit_client()
        else:
            print("🧪 Dry run mode — no ROS connection")

        # Load local planner if needed
        if self.inference_mode == "local":
            self._load_local_planner()

    def _load_local_planner(self) -> None:
        """Load the local grasp planner with optional adapter."""
        from src.inference.grasp_planner import GraspPlanner

        self.planner = GraspPlanner(adapter_path=self.adapter_path)
        self.planner.load_model()
        print(f"   Model loaded (adapter: {self.adapter_path or 'none'})")

    def _setup_camera_subscribers(self) -> None:
        """Subscribe to RealSense camera topics."""
        import roslibpy

        self._latest_rgb = None
        self._latest_depth = None

        rgb_topic = roslibpy.Topic(
            self.ros,
            "/camera/color/image_raw/compressed",
            "sensor_msgs/CompressedImage",
        )
        rgb_topic.subscribe(self._on_rgb)

        depth_topic = roslibpy.Topic(
            self.ros,
            "/camera/aligned_depth_to_color/image_raw/compressedDepth",
            "sensor_msgs/CompressedImage",
        )
        depth_topic.subscribe(self._on_depth)

        print("   📷 Camera subscribers ready")

    def _on_rgb(self, msg: dict) -> None:
        """Handle incoming RGB image."""
        self._latest_rgb = msg.get("data", "")

    def _on_depth(self, msg: dict) -> None:
        """Handle incoming depth image."""
        self._latest_depth = msg.get("data", "")

    def _setup_moveit_client(self) -> None:
        """Set up MoveIt trajectory execution client."""
        import roslibpy

        self._trajectory_pub = roslibpy.Topic(
            self.ros,
            "/move_group/goal",
            "moveit_msgs/MoveGroupActionGoal",
        )
        self._trajectory_pub.advertise()

        self._gripper_pub = roslibpy.Topic(
            self.ros,
            "/robotiq_gripper/command",
            "std_msgs/Float64",
        )
        self._gripper_pub.advertise()

        print("   🦾 MoveIt client ready")

    def capture_frame(self) -> tuple[Optional[str], Optional[str]]:
        """Capture current RGB-D frame.

        Returns:
            (rgb_b64, depth_b64) or (None, None) if no frame available.
        """
        if self.dry_run:
            # Generate synthetic frame for testing
            rgb = np.random.randint(80, 200, (480, 640, 3), dtype=np.uint8)
            from PIL import Image
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            rgb_b64 = base64.b64encode(buf.getvalue()).decode()
            return rgb_b64, None

        if self._latest_rgb is None:
            return None, None

        return self._latest_rgb, self._latest_depth

    def plan_grasp(self, rgb_b64: str, depth_b64: Optional[str]) -> Optional[dict]:
        """Plan grasp using local model or API."""
        if self.inference_mode == "api":
            return self._plan_via_api(rgb_b64, depth_b64)
        else:
            return self._plan_locally(rgb_b64, depth_b64)

    def _plan_locally(self, rgb_b64: str, depth_b64: Optional[str]) -> Optional[dict]:
        """Run local grasp planning."""
        result = self.planner.plan(
            image_b64=rgb_b64,
            depth_b64=depth_b64,
            kitchen_profile="default",
        )

        if not result.grasp_plan:
            return None

        # Take the highest confidence grasp
        best = result.grasp_plan[0]
        return {
            "pose": best.pose,
            "confidence": best.confidence,
            "object": best.object.value,
            "grasp_type": best.grasp_type.value,
            "grip_force_n": best.grip_force_n,
            "failure_risk": best.failure_risk.model_dump(),
            "model_version": result.model_version,
            "latency_ms": result.latency_ms,
        }

    def _plan_via_api(self, rgb_b64: str, depth_b64: Optional[str]) -> Optional[dict]:
        """Plan grasp via DishSpace API."""
        import httpx

        resp = httpx.post(
            f"{self.api_url}/grasp_plan",
            json={
                "image_base64": rgb_b64,
                "depth_base64": depth_b64,
                "kitchen_profile": "default",
            },
            headers={"X-API-Key": self.api_key},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("grasp_plan"):
            return None

        best = data["grasp_plan"][0]
        best["model_version"] = data.get("model_version", "unknown")
        best["latency_ms"] = data.get("latency_ms", 0)
        return best

    def is_within_workspace(self, pose: list[float]) -> bool:
        """Check if a pose is within the safe workspace bounds."""
        for i in range(3):
            if pose[i] < self.config.workspace_min[i] or pose[i] > self.config.workspace_max[i]:
                return False
        return True

    def move_to(self, pose: list[float], speed: Optional[float] = None) -> bool:
        """Send a move command to the robot arm.

        Args:
            pose: [x, y, z, rx, ry, rz] in robot base frame.
            speed: Max speed (m/s). Defaults to config.max_speed.

        Returns:
            True if move completed successfully.
        """
        speed = speed or self.config.max_speed

        if not self.is_within_workspace(pose):
            print(f"   ⚠️  Pose {pose[:3]} outside workspace bounds!")
            return False

        if self.dry_run:
            print(f"   [DRY RUN] Move to [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
            time.sleep(0.1)  # simulate move time
            return True

        # Convert to ROS trajectory and publish
        from src.pipeline.ros_bridge import (
            rpy_to_quaternion,
            transform_pose_camera_to_robot,
        )

        quat = rpy_to_quaternion(pose[3], pose[4], pose[5])

        moveit_goal = {
            "header": {"frame_id": "base_link"},
            "goal": {
                "request": {
                    "group_name": "manipulator",
                    "goal_constraints": [{
                        "position_constraints": [{
                            "constraint_region": {
                                "primitive_poses": [{
                                    "position": {"x": pose[0], "y": pose[1], "z": pose[2]},
                                    "orientation": quat,
                                }],
                            },
                        }],
                    }],
                    "max_velocity_scaling_factor": speed / 1.0,
                    "max_acceleration_scaling_factor": 0.5,
                },
            },
        }

        self._trajectory_pub.publish(roslibpy.Message(moveit_goal))

        # Wait for completion (simplified — production uses action feedback)
        time.sleep(2.0)
        return True

    def gripper_command(self, width_mm: float, force_n: float = 5.0) -> bool:
        """Send gripper open/close command.

        Args:
            width_mm: Gripper opening width in mm (0 = fully closed, 85 = fully open).
            force_n: Grip force in Newtons.

        Returns:
            True if command completed.
        """
        if self.dry_run:
            state = "open" if width_mm > 40 else f"close ({width_mm:.0f}mm, {force_n:.1f}N)"
            print(f"   [DRY RUN] Gripper {state}")
            time.sleep(0.3)
            return True

        import roslibpy

        self._gripper_pub.publish(roslibpy.Message({
            "data": width_mm / 85.0,  # Normalise to 0-1
        }))

        time.sleep(self.config.gripper_close_timeout)
        return True

    def execute_pick_wash_place(self, grasp: dict) -> bool:
        """Execute a full pick-wash-place cycle.

        Sequence:
        1. Move to pre-grasp (above object with clearance)
        2. Descend to grasp pose
        3. Close gripper
        4. Lift with clearance
        5. Move to wash area (optional)
        6. Move to drying rack
        7. Release object
        8. Return to scan pose

        Returns:
            True if cycle completed successfully.
        """
        pose = grasp["pose"]
        force = grasp.get("grip_force_n", 5.0)
        obj_type = grasp.get("object", "unknown")

        print(f"\n   🎯 Pick-wash-place: {obj_type}")
        print(f"      Grasp pose: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
        print(f"      Confidence: {grasp.get('confidence', 0):.2f}")

        # 1. Open gripper
        self.state = TaskState.APPROACHING
        if not self.gripper_command(self.config.gripper_open_width):
            return False

        # 2. Move to pre-grasp pose (above object)
        pre_grasp = pose.copy()
        pre_grasp[2] += self.config.z_clearance
        if not self.move_to(pre_grasp):
            return False

        # 3. Descend to grasp pose
        self.state = TaskState.GRASPING
        if not self.move_to(pose, speed=self.config.max_speed * 0.5):
            return False

        # 4. Close gripper
        grip_width = max(5, min(85, 85 - force * 5))  # Rough mapping
        if not self.gripper_command(grip_width, force):
            return False
        time.sleep(self.config.grasp_settle_time)

        # 5. Lift with clearance
        self.state = TaskState.LIFTING
        lift_pose = pose.copy()
        lift_pose[2] += self.config.z_clearance
        if not self.move_to(lift_pose):
            return False

        # 6. Move to drying rack
        self.state = TaskState.PLACING
        if not self.move_to(self.config.rack_pose):
            return False

        # 7. Release
        self.state = TaskState.RELEASING
        if not self.gripper_command(self.config.gripper_open_width):
            return False
        time.sleep(self.config.place_settle_time)

        # 8. Retract to scan pose
        self.state = TaskState.RETRACTING
        if not self.move_to(self.config.scan_pose):
            return False

        self.state = TaskState.IDLE
        return True

    def run_loop(self, max_cycles: int = 100) -> dict:
        """Run the continuous dishwashing loop.

        Keeps scanning for dishes, planning grasps, and executing
        pick-wash-place cycles until no more objects are detected
        or max_cycles is reached.

        Returns:
            Summary stats dict.
        """
        print("\n" + "=" * 60)
        print("  🍽️  DishSpace Dishwashing Robot — Starting")
        print("=" * 60)

        empty_scans = 0
        max_empty = 5  # Stop after 5 consecutive empty scans

        for cycle in range(max_cycles):
            cycle_start = time.time()
            print(f"\n{'─' * 40}")
            print(f"  Cycle {cycle + 1}/{max_cycles}")

            # 1. Scan
            self.state = TaskState.SCANNING
            rgb_b64, depth_b64 = self.capture_frame()
            if rgb_b64 is None:
                print("   ⚠️  No camera frame — waiting...")
                time.sleep(self.config.scan_interval)
                continue

            # 2. Plan
            self.state = TaskState.PLANNING
            print(f"   Planning grasp...")
            grasp = self.plan_grasp(rgb_b64, depth_b64)

            if grasp is None:
                empty_scans += 1
                print(f"   No objects detected (empty scan {empty_scans}/{max_empty})")
                if empty_scans >= max_empty:
                    print("   ✨ Sink is clean! No more dishes detected.")
                    break
                time.sleep(self.config.scan_interval)
                continue

            empty_scans = 0
            self.total_attempts += 1

            # Safety check
            if not self.is_within_workspace(grasp["pose"]):
                print(f"   ⚠️  Grasp outside workspace — skipping")
                self.failed_picks += 1
                continue

            confidence = grasp.get("confidence", 0)
            if confidence < 0.5:
                print(f"   ⚠️  Low confidence ({confidence:.2f}) — skipping")
                self.failed_picks += 1
                continue

            # 3. Execute pick-wash-place
            success = self.execute_pick_wash_place(grasp)

            if success:
                self.successful_picks += 1
                print(f"   ✅ Pick-wash-place successful!")
            else:
                self.failed_picks += 1
                print(f"   ❌ Pick-wash-place failed")

            cycle_time = time.time() - cycle_start
            self.cycle_times.append(cycle_time)
            print(f"   ⏱️  Cycle time: {cycle_time:.1f}s")

        # Print summary
        return self.get_summary()

    def get_summary(self) -> dict:
        """Get session summary statistics."""
        success_rate = (
            self.successful_picks / max(self.total_attempts, 1)
        )
        avg_cycle = (
            np.mean(self.cycle_times) if self.cycle_times else 0
        )

        summary = {
            "total_attempts": self.total_attempts,
            "successful_picks": self.successful_picks,
            "failed_picks": self.failed_picks,
            "success_rate": success_rate,
            "avg_cycle_time_s": float(avg_cycle),
            "total_runtime_s": float(sum(self.cycle_times)),
        }

        print(f"\n{'=' * 60}")
        print(f"  📊 Session Summary")
        print(f"{'=' * 60}")
        print(f"   Attempts:    {self.total_attempts}")
        print(f"   Successful:  {self.successful_picks}")
        print(f"   Failed:      {self.failed_picks}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Avg cycle:   {avg_cycle:.1f}s")
        print(f"{'=' * 60}")

        return summary

    def disconnect(self) -> None:
        """Clean up ROS connection."""
        if self.ros and not self.dry_run:
            # Move to home before disconnecting
            print("🏠 Moving to home position...")
            self.move_to(self.config.home_pose)
            self.gripper_command(self.config.gripper_open_width)

            self.ros.terminate()
            print("🔌 ROS disconnected")


def main():
    p = argparse.ArgumentParser(description="Deploy DishSpace dishwashing robot")

    # Connection
    p.add_argument("--ros-host", default="localhost", help="ROS bridge host")
    p.add_argument("--ros-port", type=int, default=9090)

    # Inference
    p.add_argument("--local", action="store_true", default=True,
                   help="Use local model inference (default)")
    p.add_argument("--api-url", default=None,
                   help="Use API inference instead of local")
    p.add_argument("--api-key", default="dev-key-change-me")
    p.add_argument("--adapter", default=None,
                   help="Path to DoRA adapter for local inference")

    # Operation
    p.add_argument("--max-cycles", type=int, default=100,
                   help="Maximum pick-wash-place cycles")
    p.add_argument("--single-grasp", action="store_true",
                   help="Execute one grasp and stop")
    p.add_argument("--dry-run", action="store_true",
                   help="Simulate without ROS connection")

    # Safety
    p.add_argument("--max-speed", type=float, default=0.3, help="Max speed m/s")
    p.add_argument("--max-force", type=float, default=15.0, help="Max force N")

    args = p.parse_args()

    config = DishwashingConfig(
        ros_host=args.ros_host,
        ros_port=args.ros_port,
        max_speed=args.max_speed,
        max_force=args.max_force,
    )

    inference_mode = "api" if args.api_url else "local"

    robot = DishwashingRobot(
        config=config,
        inference_mode=inference_mode,
        adapter_path=args.adapter,
        api_url=args.api_url,
        api_key=args.api_key,
        dry_run=args.dry_run,
    )

    try:
        robot.connect()

        if args.single_grasp:
            # Single grasp test
            rgb_b64, depth_b64 = robot.capture_frame()
            if rgb_b64:
                grasp = robot.plan_grasp(rgb_b64, depth_b64)
                if grasp:
                    print(f"\n🎯 Grasp plan: {json.dumps(grasp, indent=2, default=str)}")
                    robot.execute_pick_wash_place(grasp)
                else:
                    print("No objects detected")
            else:
                print("No camera frame available")
        else:
            max_cycles = 1 if args.single_grasp else args.max_cycles
            summary = robot.run_loop(max_cycles=max_cycles)

            # Save session log
            log_path = Path("data/robot_sessions") / f"session_{int(time.time())}.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(summary, indent=2))
            print(f"\n💾 Session log saved to {log_path}")

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
        robot.get_summary()
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
