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
VISION_MCP_URL_V3 = os.getenv("TBOT_VISION_MCP_URL_V3", "http://127.0.0.1:18211/turtlebot-vision-v3")

NAV_HEADING_KP = _env_float("NAV_HEADING_KP", 1.8)
NAV_FINAL_YAW_KP = _env_float("NAV_FINAL_YAW_KP", 2.0)
NAV_FRONT_THRESHOLD_M = max(0.05, _env_float("NAV_FRONT_THRESHOLD_M", 0.1))
NAV_MAX_CONSECUTIVE_ODOM_FAILURES = max(1, int(_env_float("NAV_MAX_CONSECUTIVE_ODOM_FAILURES", 2.0)))
NAV_FORWARD_STEP_M = max(0.03, _env_float("NAV_FORWARD_STEP_M", 0.20))
NAV_TURN_STEP_DEG = max(1.0, _env_float("NAV_TURN_STEP_DEG", 12.0))
NAV_HEADING_ALIGN_DEG = max(5.0, _env_float("NAV_HEADING_ALIGN_DEG", 15.0))
TBOT_CAMERA_FOV_DEG = _env_float("TBOT_CAMERA_FOV_DEG", 62.0)
NAV_OBJECT_APPROACH_STEP_M = max(0.05, _env_float("NAV_OBJECT_APPROACH_STEP_M", 0.30))
NAV_OBJECT_MOVE_SPEED_MPS = max(0.03, _env_float("NAV_OBJECT_MOVE_SPEED_MPS", 0.10))
NAV_OBJECT_TURN_SPEED_RPS = max(0.05, _env_float("NAV_OBJECT_TURN_SPEED_RPS", 0.30))
NAV_OBJECT_MAX_APPROACH_STEPS = max(1, int(_env_float("NAV_OBJECT_MAX_APPROACH_STEPS", 80.0)))
NAV_OBJECT_RECENTER_EPS_DEG = max(0.5, _env_float("NAV_OBJECT_RECENTER_EPS_DEG", 0.30 * TBOT_CAMERA_FOV_DEG))
NAV_OBJECT_LOSS_CONFIRM_FRAMES = max(1, int(_env_float("NAV_OBJECT_LOSS_CONFIRM_FRAMES", 2.0)))
NAV_OBJECT_DEFAULT_STOP_DISTANCE_M = max(0.05, _env_float("NAV_OBJECT_DEFAULT_STOP_DISTANCE_M", 0.20))
NAV_OBJECT_CLOSE_BBOX_AREA_THRESHOLD = _env_float("NAV_OBJECT_CLOSE_BBOX_AREA_THRESHOLD", 0.25)
NAV_MIDPOINT_TIMEOUT_S = max(5.0, _env_float("NAV_MIDPOINT_TIMEOUT_S", 45.0))
NAV_FORWARD_CONE_HALF_WIDTH_DEG = max(5.0, _env_float("NAV_FORWARD_CONE_HALF_WIDTH_DEG", 30.0))
NAV_MIDPOINT_SAMPLE_OFFSETS_DEG = (-10.0, 0.0, 10.0)

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


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        parsed = json.loads(text[start:end])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


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


def _extract_front_distance(collision: dict[str, Any]) -> float | None:
    front_distance = collision.get("min_forward_distance_m")
    if isinstance(front_distance, (int, float)) and math.isfinite(float(front_distance)):
        return float(front_distance)
    distances = collision.get("distances")
    if isinstance(distances, dict):
        front = distances.get("front")
        if isinstance(front, (int, float)) and math.isfinite(float(front)):
            return float(front)
    return None


def _extract_valid_distance(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _heading_offset_from_bbox_deg(bbox: Any, camera_fov_deg: float = TBOT_CAMERA_FOV_DEG) -> float | None:
    if not isinstance(bbox, dict):
        return None
    cx = bbox.get("cx")
    if not isinstance(cx, (int, float)) or not math.isfinite(float(cx)):
        return None
    cx_f = max(0.0, min(1.0, float(cx)))
    return (cx_f - 0.5) * float(camera_fov_deg)


def _heuristic_distance_from_bbox_m(bbox: Any) -> float | None:
    if not isinstance(bbox, dict):
        return None
    h = bbox.get("h")
    if not isinstance(h, (int, float)) or not math.isfinite(float(h)):
        return None
    h_f = max(0.01, min(1.0, float(h)))
    # Coarse fallback when LiDAR is unavailable: larger bbox height implies a nearer object.
    return max(0.2, min(3.0, 0.9 / h_f))


async def _turn_by_degrees(motion: Client, turn_deg: float, speed_rps: float) -> None:
    if abs(turn_deg) < 0.5:
        return
    direction = "left" if turn_deg >= 0 else "right"
    duration_s = math.radians(abs(turn_deg)) / max(abs(speed_rps), 0.01)
    await motion.call_tool(
        "tbot_motion_turn",
        {
            "direction": direction,
            "speed": max(abs(speed_rps), 0.05),
            "duration_seconds": duration_s,
        },
    )


async def _call_lidar_collision(lidar: Client) -> dict[str, Any]:
    raw = await lidar.call_tool(
        "tbot_lidar_check_collision",
        {
            "front_threshold_m": NAV_FRONT_THRESHOLD_M,
            "sector_half_width_deg": NAV_FORWARD_CONE_HALF_WIDTH_DEG,
        },
    )
    return _extract_tool_result_dict(raw)


async def _guard_motion_and_get_scale(
    lidar: Client,
    motion: Client,
) -> tuple[str, float, dict[str, Any]]:
    try:
        collision = await _call_lidar_collision(lidar)
    except Exception as e:
        collision = {"risk_level": "stop", "error": str(e)}

    risk_level = str(collision.get("risk_level", "unknown"))
    if risk_level in ("stop", "unknown"):
        await _safe_motion_stop(motion)
        return "stop", 0.0, collision
    if risk_level == "caution":
        return "caution", 0.5, collision
    return "clear", 1.0, collision


async def _vision_find_target(
    vision: Client,
    target: str,
    qualifier: str | None,
) -> dict[str, Any]:
    combined = f"{qualifier.strip()} {target}" if isinstance(qualifier, str) and qualifier.strip() else target
    payload: dict[str, Any] = {"object_name": combined}
    raw = await vision.call_tool("tbot_vision_find_object", payload)
    return _extract_tool_result_dict(raw)


async def _vision_get_target_bbox(
    vision: Client,
    target: str,
    qualifier: str | None,
) -> dict[str, Any]:
    combined = f"{qualifier.strip()} {target}" if isinstance(qualifier, str) and qualifier.strip() else target
    payload: dict[str, Any] = {"object_name": combined}
    raw = await vision.call_tool("tbot_vision_get_object_bbox", payload)
    return _extract_tool_result_dict(raw)


async def _scene_confirms_target_visible(
    vision: Client,
    target: str,
    qualifier: str | None,
    close_hint: bool = False,
) -> bool:
    qualifier_text = f" with qualifier '{qualifier.strip()}'" if isinstance(qualifier, str) and qualifier.strip() else ""
    close_text = (
        f" The robot is very close and the {target} may be filling or cropping the frame — "
        "count it as visible even if only partially in view."
        if close_hint else ""
    )
    prompt = (
        "Return JSON only with keys visible (boolean) and confidence (0..1). "
        f"Is the target object '{target}'{qualifier_text} currently visible in frame?{close_text}"
    )
    raw = await vision.call_tool("tbot_vision_describe_scene", {"prompt": prompt})
    scene = _extract_tool_result_dict(raw)
    description = str(scene.get("description", ""))
    parsed = _extract_first_json_object(description) or {}
    visible = parsed.get("visible")
    if isinstance(visible, bool):
        return visible

    desc_lower = description.lower()
    target_lower = target.lower()
    if "not visible" in desc_lower or "not present" in desc_lower or "cannot see" in desc_lower:
        return False
    if target_lower in desc_lower and any(token in desc_lower for token in ("visible", "present", "confirmed", "seen")):
        return True
    return False


async def _target_visible_or_close(
    vision: Client,
    target: str,
    qualifier: str | None,
) -> bool:
    """Return True if the target is confirmed visible, OR if its bbox area
    indicates it is very close (filling/cropping the frame), which covers the
    case where the vision LLM incorrectly says 'not visible' because the object
    is too large in the frame."""
    if await _scene_confirms_target_visible(vision=vision, target=target, qualifier=qualifier, close_hint=True):
        return True
    bbox_view = await _vision_get_target_bbox(vision=vision, target=target, qualifier=qualifier)
    bbox = bbox_view.get("bbox") if isinstance(bbox_view, dict) else None
    if isinstance(bbox, dict):
        w = bbox.get("w")
        h = bbox.get("h")
        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
            if float(w) * float(h) >= NAV_OBJECT_CLOSE_BBOX_AREA_THRESHOLD:
                return True
    return False


async def _recenter_to_visible_target_if_needed(
    motion: Client,
    target_view: dict[str, Any],
) -> bool:
    heading_offset_deg = _heading_offset_from_bbox_deg(
        target_view.get("bbox"),
        TBOT_CAMERA_FOV_DEG,
    )
    if heading_offset_deg is None or abs(heading_offset_deg) < NAV_OBJECT_RECENTER_EPS_DEG:
        return False
    await _turn_by_degrees(
        motion=motion,
        turn_deg=heading_offset_deg,
        speed_rps=NAV_OBJECT_TURN_SPEED_RPS,
    )
    return True


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
                    if reached_pos and yaw_target is not None:
                        turn_error = yaw_error
                    else:
                        turn_error = heading_error

                    if abs(turn_error) > math.radians(NAV_HEADING_ALIGN_DEG):
                        base_turn_step_deg = min(
                            NAV_TURN_STEP_DEG,
                            math.degrees(abs(turn_error)),
                        )
                        turn_step_deg = max(1.0, base_turn_step_deg)
                        if reached_pos and yaw_target is not None:
                            turn_speed = abs(_clamp(NAV_FINAL_YAW_KP * turn_error, max_angular))
                        else:
                            turn_speed = abs(_clamp(NAV_HEADING_KP * turn_error, max_angular))
                        await _turn_by_degrees(
                            motion=motion,
                            turn_deg=math.copysign(turn_step_deg, turn_error),
                            speed_rps=max(0.2, turn_speed),
                        )
                    else:
                        try:
                            collision = await _call_lidar_collision(lidar)
                            risk_level = collision.get("risk_level", "unknown")
                            front_distance = _extract_front_distance(collision)
                        except Exception as e:
                            risk_level = "stop"
                            front_distance = None
                            collision = {"error": str(e)}

                        if risk_level in ("stop", "unknown") and not reached_pos:
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

                        caution_scale = 0.5 if risk_level == "caution" else 1.0
                        segment_speed = min(max_linear, max(0.03, distance))
                        segment_speed = max(0.03, segment_speed * caution_scale)
                        base_step_m = min(distance, NAV_FORWARD_STEP_M) * caution_scale
                        step_distance_m = base_step_m
                        if front_distance is not None:
                            clearance_cap_m = max(0.0, front_distance - NAV_FRONT_THRESHOLD_M)
                            step_distance_m = min(step_distance_m, clearance_cap_m)
                        if step_distance_m <= 0.0 and not reached_pos:
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
                        await motion.call_tool(
                            "tbot_motion_move_forward_distance",
                            {"distance_m": step_distance_m, "speed": segment_speed},
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


@mcp_nav_v3.tool()
async def tbot_estimate_object_pose(
    target: str,
    qualifier: str | None = None,
) -> dict[str, Any]:
    """
    Estimate an object's position in odometry frame using vision heading + LiDAR distance.
    """
    target_clean = target.strip() if isinstance(target, str) else ""
    if not target_clean:
        raise ValueError("target must be a non-empty string")

    async with Client(VISION_MCP_URL_V3) as vision:
        find = await _vision_find_target(vision, target_clean, qualifier)

    if not bool(find.get("visible")):
        return {
            "success": False,
            "x": None,
            "y": None,
            "heading_deg": None,
            "distance_m": None,
            "confidence": "low",
            "error": "target_not_found",
        }

    bbox = find.get("bbox")
    heading_offset_deg = _heading_offset_from_bbox_deg(bbox, TBOT_CAMERA_FOV_DEG)
    if heading_offset_deg is None:
        position = str(find.get("position", "")).lower()
        if position == "left":
            heading_offset_deg = -TBOT_CAMERA_FOV_DEG / 4.0
        elif position == "right":
            heading_offset_deg = TBOT_CAMERA_FOV_DEG / 4.0
        else:
            heading_offset_deg = 0.0

    lidar_distance_m: float | None = None
    confidence = "low"
    lidar_raw: dict[str, Any] = {}
    try:
        async with Client(LIDAR_MCP_URL_V3) as lidar:
            lidar_resp = await lidar.call_tool(
                "tbot_lidar_get_distance_at_angle",
                {"angle_deg": heading_offset_deg},
            )
            lidar_raw = _extract_tool_result_dict(lidar_resp)
        lidar_value = lidar_raw.get("distance_m")
        if isinstance(lidar_value, (int, float)) and math.isfinite(float(lidar_value)):
            lidar_distance_m = float(lidar_value)
            valid_count = lidar_raw.get("valid_count")
            confidence = "high" if isinstance(valid_count, int) and valid_count >= 3 else "medium"
    except Exception as e:
        lidar_raw = {"error": str(e)}

    if lidar_distance_m is None:
        lidar_distance_m = _heuristic_distance_from_bbox_m(bbox)
        confidence = "low"

    if lidar_distance_m is None:
        return {
            "success": False,
            "x": None,
            "y": None,
            "heading_deg": None,
            "distance_m": None,
            "confidence": "low",
            "error": "distance_unavailable",
        }

    pose = await tbot_nav_get_pose()
    if pose.get("status") != "ok":
        return {
            "success": False,
            "x": None,
            "y": None,
            "heading_deg": None,
            "distance_m": lidar_distance_m,
            "confidence": confidence,
            "error": pose.get("error", "no_odom"),
            "lidar": lidar_raw,
        }

    robot_x = float(pose["x_m"])
    robot_y = float(pose["y_m"])
    robot_heading_deg = math.degrees(float(pose["yaw_rad"]))
    object_heading_deg = robot_heading_deg + heading_offset_deg
    object_heading_rad = math.radians(object_heading_deg)
    object_x = robot_x + (lidar_distance_m * math.cos(object_heading_rad))
    object_y = robot_y + (lidar_distance_m * math.sin(object_heading_rad))

    return {
        "success": True,
        "x": object_x,
        "y": object_y,
        "heading_deg": object_heading_deg,
        "distance_m": lidar_distance_m,
        "confidence": confidence,
        "lidar": lidar_raw,
    }


@mcp_nav_v3.tool()
async def tbot_nav_go_to_midpoint_between_objects(
    object_1: str,
    object_2: str,
    qualifier_1: str | None = None,
    qualifier_2: str | None = None,
    timeout_s: float = NAV_MIDPOINT_TIMEOUT_S,
) -> dict[str, Any]:
    """
    Estimate two object poses, compute midpoint, pre-check with LiDAR, bypass if needed, then navigate.
    """
    object_1_clean = object_1.strip() if isinstance(object_1, str) else ""
    object_2_clean = object_2.strip() if isinstance(object_2, str) else ""
    if not object_1_clean:
        raise ValueError("object_1 must be a non-empty string")
    if not object_2_clean:
        raise ValueError("object_2 must be a non-empty string")
    timeout_f = _validate_finite("timeout_s", timeout_s)
    if timeout_f <= 0:
        raise ValueError("timeout_s must be > 0")

    pose_1 = await tbot_estimate_object_pose(target=object_1_clean, qualifier=qualifier_1)
    if not bool(pose_1.get("success")):
        return {
            "success": False,
            "error": "object_1_not_found_or_unresolved",
            "object_1_pose": pose_1,
            "object_2_pose": None,
            "relative_from_object_1": None,
            "midpoint": None,
            "midpoint_safety": None,
            "used_bypass": False,
            "bypass_result": {},
            "nav_result": None,
        }

    pose_2 = await tbot_estimate_object_pose(target=object_2_clean, qualifier=qualifier_2)
    if not bool(pose_2.get("success")):
        return {
            "success": False,
            "error": "object_2_not_found_or_unresolved",
            "object_1_pose": pose_1,
            "object_2_pose": pose_2,
            "relative_from_object_1": None,
            "midpoint": None,
            "midpoint_safety": None,
            "used_bypass": False,
            "bypass_result": {},
            "nav_result": None,
        }

    x1 = float(pose_1["x"])
    y1 = float(pose_1["y"])
    x2 = float(pose_2["x"])
    y2 = float(pose_2["y"])
    delta_x = x2 - x1
    delta_y = y2 - y1
    separation_m = math.hypot(delta_x, delta_y)
    midpoint_x = x1 + (delta_x / 2.0)
    midpoint_y = y1 + (delta_y / 2.0)

    robot_pose = await tbot_nav_get_pose()
    if robot_pose.get("status") != "ok":
        return {
            "success": False,
            "error": robot_pose.get("error", "no_odom"),
            "object_1_pose": pose_1,
            "object_2_pose": pose_2,
            "relative_from_object_1": {
                "dx_m": delta_x,
                "dy_m": delta_y,
                "separation_m": separation_m,
            },
            "midpoint": {"x_m": midpoint_x, "y_m": midpoint_y},
            "midpoint_safety": None,
            "used_bypass": False,
            "bypass_result": {},
            "nav_result": None,
        }

    robot_x = float(robot_pose["x_m"])
    robot_y = float(robot_pose["y_m"])
    robot_yaw = float(robot_pose["yaw_rad"])
    to_mid_x = midpoint_x - robot_x
    to_mid_y = midpoint_y - robot_y
    distance_to_midpoint_m = math.hypot(to_mid_x, to_mid_y)
    heading_to_midpoint = math.atan2(to_mid_y, to_mid_x)
    relative_angle_deg = math.degrees(_wrap_to_pi(heading_to_midpoint - robot_yaw))
    required_clearance_m = distance_to_midpoint_m + NAV_FRONT_THRESHOLD_M

    lidar_check: dict[str, Any] = {}
    lidar_samples: list[dict[str, Any]] = []
    valid_sample_distances: list[float] = []
    measured_lidar_distance_m: float | None = None
    midpoint_safety_class = "inconclusive"
    try:
        async with Client(LIDAR_MCP_URL_V3) as lidar:
            for offset_deg in NAV_MIDPOINT_SAMPLE_OFFSETS_DEG:
                sample_angle_deg = relative_angle_deg + float(offset_deg)
                sample_payload: dict[str, Any]
                sample_distance_m: float | None = None
                try:
                    lidar_raw = await lidar.call_tool(
                        "tbot_lidar_get_distance_at_angle",
                        {"angle_deg": sample_angle_deg},
                    )
                    sample_payload = _extract_tool_result_dict(lidar_raw)
                    sample_distance_m = _extract_valid_distance(sample_payload.get("distance_m"))
                    if sample_distance_m is not None:
                        valid_sample_distances.append(sample_distance_m)
                except Exception as sample_error:
                    sample_payload = {"error": str(sample_error)}

                if abs(float(offset_deg)) < 1e-6:
                    lidar_check = sample_payload
                    measured_lidar_distance_m = sample_distance_m

                lidar_samples.append(
                    {
                        "angle_deg": sample_angle_deg,
                        "offset_deg": float(offset_deg),
                        "distance_m": sample_distance_m,
                        "status": sample_payload.get("status"),
                        "valid_count": sample_payload.get("valid_count"),
                        "error": sample_payload.get("error"),
                    }
                )
    except Exception as e:
        lidar_check = {"error": str(e)}
        lidar_samples = [
            {
                "angle_deg": relative_angle_deg + float(offset_deg),
                "offset_deg": float(offset_deg),
                "distance_m": None,
                "status": None,
                "valid_count": None,
                "error": str(e),
            }
            for offset_deg in NAV_MIDPOINT_SAMPLE_OFFSETS_DEG
        ]

    if measured_lidar_distance_m is None:
        measured_lidar_distance_m = _extract_valid_distance(lidar_check.get("distance_m"))
    if measured_lidar_distance_m is None and valid_sample_distances:
        measured_lidar_distance_m = valid_sample_distances[0]

    if valid_sample_distances:
        all_clear = all(distance >= required_clearance_m for distance in valid_sample_distances)
        all_blocked = all(distance < required_clearance_m for distance in valid_sample_distances)
        if all_clear:
            midpoint_safety_class = "clear"
        elif all_blocked:
            midpoint_safety_class = "blocked"

    midpoint_is_safe = midpoint_safety_class == "clear"
    midpoint_safety = {
        "relative_angle_deg": relative_angle_deg,
        "distance_to_midpoint_m": distance_to_midpoint_m,
        "required_clearance_m": required_clearance_m,
        "measured_lidar_distance_m": measured_lidar_distance_m,
        "is_safe": midpoint_is_safe,
        "classification": midpoint_safety_class,
        "samples": lidar_samples,
        "lidar": lidar_check,
    }

    used_bypass = False
    bypass_result: dict[str, Any] = {}
    if midpoint_safety_class == "blocked":
        used_bypass = True
        try:
            async with Client(MOTION_MCP_URL_V3) as motion:
                bypass_raw = await motion.call_tool("tbot_motion_bypass_obstacle", {})
                bypass_result = _extract_tool_result_dict(bypass_raw)
        except Exception as e:
            bypass_result = {"status": "error", "error": str(e)}

        if bypass_result.get("status") != "completed":
            return {
                "success": False,
                "error": "midpoint_unsafe_bypass_failed",
                "object_1_pose": pose_1,
                "object_2_pose": pose_2,
                "relative_from_object_1": {
                    "dx_m": delta_x,
                    "dy_m": delta_y,
                    "separation_m": separation_m,
                },
                "midpoint": {"x_m": midpoint_x, "y_m": midpoint_y},
                "midpoint_safety": midpoint_safety,
                "used_bypass": used_bypass,
                "bypass_result": bypass_result,
                "nav_result": None,
            }

    nav_result = await tbot_nav_go_to_pose(
        target_x_m=midpoint_x,
        target_y_m=midpoint_y,
        timeout_s=timeout_f,
    )

    if nav_result.get("status") == "collision_blocked" and not used_bypass:
        used_bypass = True
        try:
            async with Client(MOTION_MCP_URL_V3) as motion:
                bypass_retry_raw = await motion.call_tool("tbot_motion_bypass_obstacle", {})
                bypass_result = _extract_tool_result_dict(bypass_retry_raw)
        except Exception as e:
            bypass_result = {"status": "error", "error": str(e)}

        if bypass_result.get("status") == "completed":
            retry_timeout_s = max(5.0, timeout_f * 0.5)
            nav_result = await tbot_nav_go_to_pose(
                target_x_m=midpoint_x,
                target_y_m=midpoint_y,
                timeout_s=retry_timeout_s,
            )
        else:
            return {
                "success": False,
                "error": "midpoint_retry_bypass_failed",
                "object_1_pose": pose_1,
                "object_2_pose": pose_2,
                "relative_from_object_1": {
                    "dx_m": delta_x,
                    "dy_m": delta_y,
                    "separation_m": separation_m,
                },
                "midpoint": {"x_m": midpoint_x, "y_m": midpoint_y},
                "midpoint_safety": midpoint_safety,
                "used_bypass": used_bypass,
                "bypass_result": bypass_result,
                "nav_result": nav_result,
            }

    reached = nav_result.get("status") == "reached"
    return {
        "success": reached,
        "error": None if reached else str(nav_result.get("status", "midpoint_nav_failed")),
        "object_1_pose": pose_1,
        "object_2_pose": pose_2,
        "relative_from_object_1": {
            "dx_m": delta_x,
            "dy_m": delta_y,
            "separation_m": separation_m,
        },
        "midpoint": {"x_m": midpoint_x, "y_m": midpoint_y},
        "midpoint_safety": midpoint_safety,
        "used_bypass": used_bypass,
        "bypass_result": bypass_result,
        "nav_result": nav_result,
    }


@mcp_nav_v3.tool()
async def tbot_navigate_to_object(
    target: str,
    qualifier: str | None = None,
    stop_distance: float = NAV_OBJECT_DEFAULT_STOP_DISTANCE_M,
    confirm_in_frame: bool = True,
) -> dict[str, Any]:
    """
    Approach a visible object with collision checks and optional visual confirmation.
    If the object is visible but off-center, recenter in-place via bbox instead of rescanning.
    Stops when the robot is within stop_distance and the object is still visible.
    """
    target_clean = target.strip() if isinstance(target, str) else ""
    if not target_clean:
        raise ValueError("target must be a non-empty string")
    stop_distance_f = _validate_finite("stop_distance", stop_distance)
    if stop_distance_f <= 0:
        raise ValueError("stop_distance must be > 0")

    final_pose = await tbot_nav_get_pose()
    object_in_frame = False
    scene_description = ""
    stopped_reason = "target_not_visible"

    async with Client(VISION_MCP_URL_V3) as vision:
        async with Client(MOTION_MCP_URL_V3) as motion:
            async with Client(LIDAR_MCP_URL_V3) as lidar:
                initial_visible = await _scene_confirms_target_visible(
                    vision=vision,
                    target=target_clean,
                    qualifier=qualifier,
                )
                if not initial_visible:
                    await _safe_motion_stop(motion)
                    final_pose = await tbot_nav_get_pose()
                    return {
                        "success": False,
                        "final_pose": final_pose,
                        "object_in_frame": False,
                        "scene_description": "",
                        "stopped_reason": "target_not_visible",
                    }

                object_in_frame = True
                initial_bbox_view = await _vision_get_target_bbox(
                    vision=vision,
                    target=target_clean,
                    qualifier=qualifier,
                )
                approach_steps = 0
                while approach_steps < NAV_OBJECT_MAX_APPROACH_STEPS:
                    risk_level, scale, collision = await _guard_motion_and_get_scale(lidar, motion)
                    front_distance_m = _extract_front_distance(collision)
                    target_distance_m: float | None = None
                    bbox_for_range: dict[str, Any] | None = None
                    if front_distance_m is None or risk_level == "stop":
                        bbox_for_range = await _vision_get_target_bbox(
                            vision=vision,
                            target=target_clean,
                            qualifier=qualifier,
                        )
                        if bool(bbox_for_range.get("visible")):
                            heading_offset_deg = _heading_offset_from_bbox_deg(
                                bbox_for_range.get("bbox"),
                                TBOT_CAMERA_FOV_DEG,
                            )
                            if heading_offset_deg is not None:
                                distance_raw = await lidar.call_tool(
                                    "tbot_lidar_get_distance_at_angle",
                                    {"angle_deg": heading_offset_deg},
                                )
                                distance_result = _extract_tool_result_dict(distance_raw)
                                measured = distance_result.get("distance_m")
                                if isinstance(measured, (int, float)) and math.isfinite(float(measured)):
                                    target_distance_m = float(measured)

                    if risk_level == "stop":
                        visible_for_stop = await _target_visible_or_close(
                            vision=vision,
                            target=target_clean,
                            qualifier=qualifier,
                        )
                        object_in_frame = bool(visible_for_stop) or bool(bbox_for_range and bbox_for_range.get("visible"))
                        close_enough = (
                            (front_distance_m is not None and front_distance_m <= stop_distance_f)
                            or (target_distance_m is not None and target_distance_m <= stop_distance_f)
                        )

                        await _safe_motion_stop(motion)
                        if object_in_frame and close_enough:
                            stopped_reason = "reached_target"
                            break
                        if object_in_frame:
                            final_pose = await tbot_nav_get_pose()
                            return {
                                "success": False,
                                "final_pose": final_pose,
                                "object_in_frame": True,
                                "scene_description": "",
                                "stopped_reason": "target_blocked_visible",
                                "collision": collision,
                                "front_distance_m": front_distance_m,
                                "target_distance_m": target_distance_m,
                            }

                        bypass_raw = await motion.call_tool("tbot_motion_bypass_obstacle", {})
                        bypass = _extract_tool_result_dict(bypass_raw)
                        if bypass.get("status") != "completed":
                            await _safe_motion_stop(motion)
                            final_pose = await tbot_nav_get_pose()
                            return {
                                "success": False,
                                "final_pose": final_pose,
                                "object_in_frame": object_in_frame,
                                "scene_description": "",
                                "stopped_reason": "collision_stop",
                                "collision": collision,
                                "bypass": bypass,
                            }
                        continue

                    distances_for_limit = [
                        d for d in (target_distance_m, front_distance_m)
                        if isinstance(d, (int, float)) and math.isfinite(float(d))
                    ]
                    distance_for_limit = min(distances_for_limit) if distances_for_limit else None
                    allowed_step = NAV_OBJECT_APPROACH_STEP_M * scale
                    if distance_for_limit is not None:
                        allowed_step = min(allowed_step, max(0.0, distance_for_limit - stop_distance_f))
                    if front_distance_m is not None:
                        # Keep a stop buffer from the nearest obstacle in the forward +/-cone.
                        allowed_step = min(allowed_step, max(0.0, front_distance_m - stop_distance_f))

                    near_target = distance_for_limit is not None and distance_for_limit <= stop_distance_f
                    blocked = allowed_step <= 0.0
                    if near_target or blocked:
                        visible_for_stop = await _target_visible_or_close(
                            vision=vision,
                            target=target_clean,
                            qualifier=qualifier,
                        )
                        object_in_frame = bool(visible_for_stop) or bool(bbox_for_range and bbox_for_range.get("visible"))
                        if object_in_frame:
                            await _safe_motion_stop(motion)
                            stopped_reason = "reached_target"
                            break
                        if blocked:
                            await _safe_motion_stop(motion)
                            final_pose = await tbot_nav_get_pose()
                            return {
                                "success": False,
                                "final_pose": final_pose,
                                "object_in_frame": False,
                                "scene_description": "",
                                "stopped_reason": "target_not_visible_near_obstacle",
                                "collision": collision,
                                "front_distance_m": front_distance_m,
                                "target_distance_m": target_distance_m,
                            }

                    move_speed = max(0.03, NAV_OBJECT_MOVE_SPEED_MPS * scale)
                    await motion.call_tool(
                        "tbot_motion_move_forward_distance",
                        {"distance_m": allowed_step, "speed": move_speed},
                    )
                    approach_steps += 1

                    post_move_visible = await _target_visible_or_close(
                        vision=vision,
                        target=target_clean,
                        qualifier=qualifier,
                    )
                    object_in_frame = bool(post_move_visible)
                    if object_in_frame:
                        bbox_view = await _vision_get_target_bbox(
                            vision=vision,
                            target=target_clean,
                            qualifier=qualifier,
                        )
                        if bool(bbox_view.get("visible")):
                            await _recenter_to_visible_target_if_needed(motion, bbox_view)
                        continue

                    # Confirm absence across additional frame checks before declaring target_lost.
                    for _ in range(max(0, NAV_OBJECT_LOSS_CONFIRM_FRAMES - 1)):
                        confirm_visible = await _target_visible_or_close(
                            vision=vision,
                            target=target_clean,
                            qualifier=qualifier,
                        )
                        if bool(confirm_visible):
                            object_in_frame = True
                            confirm_bbox = await _vision_get_target_bbox(
                                vision=vision,
                                target=target_clean,
                                qualifier=qualifier,
                            )
                            if bool(confirm_bbox.get("visible")):
                                await _recenter_to_visible_target_if_needed(motion, confirm_bbox)
                            break

                    if not object_in_frame:
                        await _safe_motion_stop(motion)
                        final_pose = await tbot_nav_get_pose()
                        return {
                            "success": False,
                            "final_pose": final_pose,
                            "object_in_frame": False,
                            "scene_description": "",
                            "stopped_reason": "target_lost",
                        }

                if stopped_reason != "reached_target":
                    await _safe_motion_stop(motion)
                    final_pose = await tbot_nav_get_pose()
                    return {
                        "success": False,
                        "final_pose": final_pose,
                        "object_in_frame": False,
                        "scene_description": "",
                        "stopped_reason": "max_retries",
                    }

                if not object_in_frame:
                    object_in_frame = await _target_visible_or_close(
                        vision=vision,
                        target=target_clean,
                        qualifier=qualifier,
                    )
                if confirm_in_frame:
                    scene_raw = await vision.call_tool(
                        "tbot_vision_describe_scene",
                        {"prompt": (
                            f"The robot has stopped close to a {target_clean}. "
                            f"The object may be very close and only partially visible or filling the entire frame. "
                            f"Confirm whether any part of the {target_clean} is still visible, "
                            f"even if it is cropped at the edges."
                        )},
                    )
                    scene = _extract_tool_result_dict(scene_raw)
                    scene_description = str(scene.get("description", ""))

    final_pose = await tbot_nav_get_pose()
    return {
        "success": (stopped_reason == "reached_target" and (not confirm_in_frame or object_in_frame)),
        "final_pose": final_pose,
        "object_in_frame": object_in_frame,
        "scene_description": scene_description,
        "stopped_reason": stopped_reason,
    }


@mcp_nav_v3.tool()
async def tbot_nav_patrol(
    waypoints: list[dict[str, Any]],
    cycles: int = 1,
    timeout_s_total: float = 120.0,
) -> dict[str, Any]:
    """Patrol through waypoints for N cycles using tbot_nav_go_to_pose."""
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("waypoints must be a non-empty list")
    cycles_i = int(cycles)
    if cycles_i <= 0:
        raise ValueError("cycles must be >= 1")
    timeout_total = _validate_finite("timeout_s_total", timeout_s_total)
    if timeout_total <= 0:
        raise ValueError("timeout_s_total must be > 0")

    started = time.monotonic()
    completed_waypoints = 0
    total_targets = len(waypoints) * cycles_i

    for _cycle in range(cycles_i):
        for waypoint in waypoints:
            elapsed = time.monotonic() - started
            remaining = timeout_total - elapsed
            if remaining <= 0:
                return {
                    "status": "timeout",
                    "completed_waypoints": completed_waypoints,
                    "total_waypoints": total_targets,
                    "elapsed_s": elapsed,
                }

            x = waypoint.get("x_m")
            y = waypoint.get("y_m")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError("each waypoint must include numeric x_m and y_m")

            result = await tbot_nav_go_to_pose(
                target_x_m=float(x),
                target_y_m=float(y),
                target_yaw_rad=float(waypoint["yaw_rad"]) if isinstance(waypoint.get("yaw_rad"), (int, float)) else None,
                timeout_s=min(float(waypoint.get("timeout_s", remaining)), remaining),
            )

            if result.get("status") == "reached":
                completed_waypoints += 1
                continue

            return {
                "status": "partial_completed",
                "completed_waypoints": completed_waypoints,
                "total_waypoints": total_targets,
                "failed_waypoint": waypoint,
                "last_result": result,
                "elapsed_s": time.monotonic() - started,
            }

    return {
        "status": "completed",
        "completed_waypoints": completed_waypoints,
        "total_waypoints": total_targets,
        "elapsed_s": time.monotonic() - started,
    }


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
