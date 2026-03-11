"""
TurtleBot Navigation MCP Server V3.

Navigation primitives backed by ROS2 odometry (/odom).
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
import uuid
from typing import Any, Optional

from fastmcp import Client, FastMCP

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    RCLPY_AVAILABLE = True
    RCLPY_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:
    rclpy = None  # type: ignore[assignment]
    Odometry = object  # type: ignore[assignment]
    Node = object  # type: ignore[assignment]
    qos_profile_sensor_data = None  # type: ignore[assignment]
    RCLPY_AVAILABLE = False
    RCLPY_IMPORT_ERROR = e


logger = logging.getLogger(__name__)
mcp_nav_v3 = FastMCP("TurtleBot Navigation MCP Server V3")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


ODOM_TOPIC = os.getenv("ODOM_TOPIC", "/odom")
ODOM_TIMEOUT_S = _env_float("ODOM_TIMEOUT_S", 1.0)
ODOM_STALE_TIMEOUT_S = _env_float("ODOM_STALE_TIMEOUT_S", 0.5)
ODOM_NODE_NAME_PREFIX = os.getenv("ODOM_NODE_NAME_PREFIX", "tbot_nav_v3_odom")

MOTION_MCP_URL_V3 = os.getenv("TBOT_MOTION_MCP_URL_V3", "http://127.0.0.1:18210/turtlebot-motion-v3")
LIDAR_MCP_URL_V3 = os.getenv("TBOT_LIDAR_MCP_URL_V3", "http://127.0.0.1:18212/turtlebot-lidar-v3")

NAV_CONTROL_TICK_S = max(0.05, _env_float("NAV_CONTROL_TICK_S", 0.2))
NAV_HEADING_KP = _env_float("NAV_HEADING_KP", 1.8)
NAV_FINAL_YAW_KP = _env_float("NAV_FINAL_YAW_KP", 2.0)
NAV_FRONT_THRESHOLD_M = max(0.05, _env_float("NAV_FRONT_THRESHOLD_M", 0.1))
NAV_MAX_CONSECUTIVE_ODOM_FAILURES = max(1, int(_env_float("NAV_MAX_CONSECUTIVE_ODOM_FAILURES", 2.0)))

_rclpy_init_lock = threading.Lock()


def _validate_finite(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _clamp(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return max(-limit, min(limit, value))


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _extract_tool_result_dict(tool_result: Any) -> dict[str, Any]:
    if isinstance(tool_result, dict):
        return tool_result
    structured = getattr(tool_result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    structured_camel = getattr(tool_result, "structuredContent", None)
    if isinstance(structured_camel, dict):
        return structured_camel
    content = getattr(tool_result, "content", None)
    if isinstance(content, list):
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
    return {}


def _is_odom_stale(odom: dict[str, Any], stale_timeout_s: float) -> bool:
    if stale_timeout_s <= 0:
        return False

    age_mono = odom.get("age_mono_s")
    if isinstance(age_mono, (int, float)) and math.isfinite(float(age_mono)):
        if float(age_mono) > stale_timeout_s:
            return True

    stamp_s = odom.get("stamp_sec")
    if isinstance(stamp_s, (int, float)) and math.isfinite(float(stamp_s)):
        age_wall = time.time() - float(stamp_s)
        # Ignore obviously different time domains (e.g., simulation clock mismatch).
        if 0.0 <= age_wall <= 60.0 and age_wall > stale_timeout_s:
            return True

    return False


def _get_one_odom_sync(timeout_s: float) -> dict[str, Any]:
    if not RCLPY_AVAILABLE:
        raise RuntimeError(f"rclpy is not available: {RCLPY_IMPORT_ERROR}")

    with _rclpy_init_lock:
        if not rclpy.ok():
            rclpy.init()

    node_name = f"{ODOM_NODE_NAME_PREFIX}_{uuid.uuid4().hex[:8]}"
    odom_data: dict[str, Any] = {}
    received = threading.Event()

    class _OneShotOdomNode(Node):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(node_name)
            self.create_subscription(
                Odometry,
                ODOM_TOPIC,
                self._callback,
                qos_profile_sensor_data,
            )

        def _callback(self, msg: Any) -> None:
            if received.is_set():
                return

            now_mono = time.monotonic()
            stamp_s: float | None = None
            if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
                stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1_000_000_000.0

            px = float(msg.pose.pose.position.x)
            py = float(msg.pose.pose.position.y)
            qx = float(msg.pose.pose.orientation.x)
            qy = float(msg.pose.pose.orientation.y)
            qz = float(msg.pose.pose.orientation.z)
            qw = float(msg.pose.pose.orientation.w)
            yaw = _yaw_from_quaternion(qx, qy, qz, qw)

            odom_data.update(
                {
                    "topic": ODOM_TOPIC,
                    "frame_id": msg.header.frame_id,
                    "child_frame_id": msg.child_frame_id,
                    "x_m": px,
                    "y_m": py,
                    "yaw_rad": yaw,
                    "linear_mps": float(msg.twist.twist.linear.x),
                    "angular_rps": float(msg.twist.twist.angular.z),
                    "covariance_pose": [float(v) for v in msg.pose.covariance],
                    "covariance_twist": [float(v) for v in msg.twist.covariance],
                    "stamp_sec": stamp_s,
                    "received_unix_s": time.time(),
                    "received_mono_s": now_mono,
                }
            )
            received.set()

    node = _OneShotOdomNode()
    try:
        deadline = time.monotonic() + timeout_s
        while not received.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(f"No odometry received within {timeout_s}s on topic {ODOM_TOPIC!r}")
            rclpy.spin_once(node, timeout_sec=min(0.1, remaining))
    finally:
        node.destroy_node()

    odom_data["age_mono_s"] = max(0.0, time.monotonic() - odom_data["received_mono_s"])
    return odom_data


async def _get_one_odom(timeout_s: float = ODOM_TIMEOUT_S) -> dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_one_odom_sync, timeout_s)


async def _safe_motion_stop(motion: Client) -> None:
    try:
        await motion.call_tool("tbot_motion_stop", {})
    except Exception:
        pass


@mcp_nav_v3.tool()
async def tbot_nav_get_pose(timeout_s: float = ODOM_TIMEOUT_S) -> dict[str, Any]:
    """
    Return the robot pose/velocity from ROS2 /odom.
    """
    timeout_f = _validate_finite("timeout_s", timeout_s)
    if timeout_f <= 0:
        raise ValueError("timeout_s must be > 0")

    if not RCLPY_AVAILABLE:
        return {"status": "no_odom", "topic": ODOM_TOPIC, "error": str(RCLPY_IMPORT_ERROR)}

    try:
        odom = await _get_one_odom(timeout_f)
    except Exception as e:
        return {"status": "no_odom", "topic": ODOM_TOPIC, "error": str(e)}

    if _is_odom_stale(odom, ODOM_STALE_TIMEOUT_S):
        return {"status": "no_odom", "topic": ODOM_TOPIC, "error": "Odometry is stale"}

    return {
        "status": "ok",
        "topic": ODOM_TOPIC,
        "frame_id": odom.get("frame_id"),
        "child_frame_id": odom.get("child_frame_id"),
        "x_m": odom.get("x_m"),
        "y_m": odom.get("y_m"),
        "yaw_rad": odom.get("yaw_rad"),
        "linear_mps": odom.get("linear_mps"),
        "angular_rps": odom.get("angular_rps"),
        "covariance_pose": odom.get("covariance_pose"),
        "covariance_twist": odom.get("covariance_twist"),
        "stamp_sec": odom.get("stamp_sec"),
        "age_mono_s": odom.get("age_mono_s"),
    }


@mcp_nav_v3.tool()
async def tbot_nav_go_to_pose(
    target_x_m: float,
    target_y_m: float,
    target_yaw_rad: float | None = None,
    pos_tolerance_m: float = 0.10,
    yaw_tolerance_rad: float = 0.15,
    max_linear_mps: float = 0.12,
    max_angular_rps: float = 0.6,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """
    Go to a target pose using /odom feedback and short motion ticks.
    """
    x_target = _validate_finite("target_x_m", target_x_m)
    y_target = _validate_finite("target_y_m", target_y_m)
    yaw_target = _validate_finite("target_yaw_rad", target_yaw_rad) if target_yaw_rad is not None else None
    pos_tol = _validate_finite("pos_tolerance_m", pos_tolerance_m)
    yaw_tol = _validate_finite("yaw_tolerance_rad", yaw_tolerance_rad)
    max_linear = abs(_validate_finite("max_linear_mps", max_linear_mps))
    max_angular = abs(_validate_finite("max_angular_rps", max_angular_rps))
    timeout_f = _validate_finite("timeout_s", timeout_s)

    if pos_tol <= 0:
        raise ValueError("pos_tolerance_m must be > 0")
    if yaw_tol <= 0:
        raise ValueError("yaw_tolerance_rad must be > 0")
    if max_linear <= 0:
        raise ValueError("max_linear_mps must be > 0")
    if max_angular <= 0:
        raise ValueError("max_angular_rps must be > 0")
    if timeout_f <= 0:
        raise ValueError("timeout_s must be > 0")

    started = time.monotonic()
    steps = 0
    consecutive_odom_failures = 0
    last_pose: dict[str, Any] | None = None

    async with Client(MOTION_MCP_URL_V3) as motion:
        async with Client(LIDAR_MCP_URL_V3) as lidar:
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_f:
                    await _safe_motion_stop(motion)
                    return {
                        "status": "timeout",
                        "target": {"x_m": x_target, "y_m": y_target, "yaw_rad": yaw_target},
                        "distance_remaining_m": None,
                        "yaw_error_rad": None,
                        "elapsed_s": elapsed,
                        "steps": steps,
                        "last_pose": last_pose,
                    }

                try:
                    odom_timeout = max(0.1, min(ODOM_TIMEOUT_S, timeout_f - elapsed))
                    odom = await _get_one_odom(odom_timeout)
                    if _is_odom_stale(odom, ODOM_STALE_TIMEOUT_S):
                        raise RuntimeError("stale odometry")
                    consecutive_odom_failures = 0
                except Exception as e:
                    consecutive_odom_failures += 1
                    if consecutive_odom_failures >= NAV_MAX_CONSECUTIVE_ODOM_FAILURES:
                        await _safe_motion_stop(motion)
                        return {
                            "status": "no_odom",
                            "target": {"x_m": x_target, "y_m": y_target, "yaw_rad": yaw_target},
                            "distance_remaining_m": None,
                            "yaw_error_rad": None,
                            "elapsed_s": time.monotonic() - started,
                            "steps": steps,
                            "last_pose": last_pose,
                            "error": str(e),
                        }
                    await asyncio.sleep(0.05)
                    continue

                x_cur = float(odom["x_m"])
                y_cur = float(odom["y_m"])
                yaw_cur = float(odom["yaw_rad"])
                last_pose = {
                    "x_m": x_cur,
                    "y_m": y_cur,
                    "yaw_rad": yaw_cur,
                    "frame_id": odom.get("frame_id"),
                    "child_frame_id": odom.get("child_frame_id"),
                    "stamp_sec": odom.get("stamp_sec"),
                }

                dx = x_target - x_cur
                dy = y_target - y_cur
                distance = math.hypot(dx, dy)
                heading = math.atan2(dy, dx)
                heading_error = _wrap_to_pi(heading - yaw_cur)

                desired_final_yaw = yaw_target if yaw_target is not None else heading
                yaw_error = _wrap_to_pi(desired_final_yaw - yaw_cur)

                reached_pos = distance <= pos_tol
                reached_yaw = abs(yaw_error) <= yaw_tol if yaw_target is not None else True
                if reached_pos and reached_yaw:
                    await _safe_motion_stop(motion)
                    return {
                        "status": "reached",
                        "target": {"x_m": x_target, "y_m": y_target, "yaw_rad": yaw_target},
                        "distance_remaining_m": distance,
                        "yaw_error_rad": yaw_error,
                        "elapsed_s": time.monotonic() - started,
                        "steps": steps,
                        "last_pose": last_pose,
                    }

                try:
                    collision_raw = await lidar.call_tool(
                        "tbot_lidar_get_obstacle_distances",
                        {"sector": "front"},
                    )
                    collision = _extract_tool_result_dict(collision_raw)
                    front_distance = (collision.get("distances") or {}).get("front")
                    if isinstance(front_distance, (int, float)):
                        risk_level = "stop" if front_distance < NAV_FRONT_THRESHOLD_M else "clear"
                    else:
                        risk_level = "stop"
                except Exception as e:
                    risk_level = "stop"
                    front_distance = None
                    collision = {"error": str(e)}

                if risk_level == "stop" and not reached_pos:
                    await _safe_motion_stop(motion)
                    return {
                        "status": "collision_blocked",
                        "target": {"x_m": x_target, "y_m": y_target, "yaw_rad": yaw_target},
                        "distance_remaining_m": distance,
                        "yaw_error_rad": yaw_error,
                        "elapsed_s": time.monotonic() - started,
                        "steps": steps,
                        "last_pose": last_pose,
                        "front_distance_m": front_distance,
                        "collision": collision,
                    }

                linear_cmd = min(max_linear, max(0.03, distance))
                if reached_pos:
                    linear_cmd = 0.0

                if abs(heading_error) > math.radians(50.0):
                    linear_cmd = 0.0
                elif abs(heading_error) > math.radians(25.0):
                    linear_cmd *= 0.4

                if reached_pos and yaw_target is not None:
                    angular_cmd = _clamp(NAV_FINAL_YAW_KP * yaw_error, max_angular)
                else:
                    angular_cmd = _clamp(NAV_HEADING_KP * heading_error, max_angular)

                try:
                    await motion.call_tool(
                        "tbot_motion_move_timed",
                        {"linear": linear_cmd, "angular": angular_cmd, "duration_s": NAV_CONTROL_TICK_S},
                    )
                except Exception as e:
                    await _safe_motion_stop(motion)
                    return {
                        "status": "error",
                        "target": {"x_m": x_target, "y_m": y_target, "yaw_rad": yaw_target},
                        "distance_remaining_m": distance,
                        "yaw_error_rad": yaw_error,
                        "elapsed_s": time.monotonic() - started,
                        "steps": steps,
                        "last_pose": last_pose,
                        "error": f"motion command failed: {e}",
                    }

                steps += 1


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18213,
    path: str = "/turtlebot-nav-v3",
    options: dict = {},
) -> None:
    """Run the TurtleBot Navigation MCP Server V3."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting TurtleBot Nav MCP V3 odom_topic=%s at %s:%s%s",
        ODOM_TOPIC,
        host,
        port,
        path,
    )
    mcp_nav_v3.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
