"""
TurtleBot Navigation MCP Server V3.

Navigation primitives backed by ROS2 odometry (/odom).
"""

import asyncio
import json
import logging
import math
import os
from pathlib import Path
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
NAV_FORWARD_STEP_M = max(0.03, _env_float("NAV_FORWARD_STEP_M", 0.12))
NAV_TURN_STEP_DEG = max(1.0, _env_float("NAV_TURN_STEP_DEG", 12.0))
NAV_HEADING_ALIGN_DEG = max(5.0, _env_float("NAV_HEADING_ALIGN_DEG", 15.0))
TBOT_CAMERA_FOV_DEG = _env_float("TBOT_CAMERA_FOV_DEG", 62.0)
NAV_OBJECT_APPROACH_STEP_M = max(0.05, _env_float("NAV_OBJECT_APPROACH_STEP_M", 0.25))
NAV_OBJECT_MOVE_SPEED_MPS = max(0.03, _env_float("NAV_OBJECT_MOVE_SPEED_MPS", 0.10))
NAV_OBJECT_TURN_SPEED_RPS = max(0.05, _env_float("NAV_OBJECT_TURN_SPEED_RPS", 0.30))
NAV_OBJECT_MAX_APPROACH_STEPS = max(1, int(_env_float("NAV_OBJECT_MAX_APPROACH_STEPS", 80.0)))
NAV_OBJECT_RECENTER_EPS_DEG = max(0.5, _env_float("NAV_OBJECT_RECENTER_EPS_DEG", 1.0))
NAV_OBJECT_LOSS_CONFIRM_FRAMES = max(1, int(_env_float("NAV_OBJECT_LOSS_CONFIRM_FRAMES", 2.0)))
NAV_OBJECT_DEFAULT_STOP_DISTANCE_M = max(0.05, _env_float("NAV_OBJECT_DEFAULT_STOP_DISTANCE_M", 0.20))
NAV_MIDPOINT_TIMEOUT_S = max(5.0, _env_float("NAV_MIDPOINT_TIMEOUT_S", 45.0))
NAV_CONTINUOUS_SEGMENT_S = max(0.10, _env_float("NAV_CONTINUOUS_SEGMENT_S", 0.35))
NAV_MIDPOINT_SAMPLE_OFFSETS_DEG = (-10.0, 0.0, 10.0)
NAV_MAP_OBJECT_TTL_S = max(1.0, _env_float("NAV_MAP_OBJECT_TTL_S", 120.0))
NAV_SESSION_MAP_TRAIL_MAX_POINTS = max(10, int(_env_float("NAV_SESSION_MAP_TRAIL_MAX_POINTS", 200.0)))
NAV_SESSION_MAP_PERSIST_ENABLED = _env_bool("TBOT_SESSION_MAP_PERSIST_ENABLED", True)
NAV_SESSION_MAP_SNAPSHOT_PATH = os.getenv("TBOT_SESSION_MAP_SNAPSHOT_PATH", "").strip()
NAV_SESSION_MAP_SNAPSHOT_DIR = os.getenv("TBOT_SESSION_MAP_SNAPSHOT_DIR", "/tmp/onit").strip() or "/tmp/onit"

_rclpy_init_lock = threading.Lock()
_session_state_lock = threading.Lock()
_active_session_state: dict[str, Any] | None = None


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


def _normalize_session_id(session_id: str | None) -> str:
    if isinstance(session_id, str):
        cleaned = session_id.strip()
        if cleaned:
            return cleaned
    return "default"


def _normalize_qualifier(qualifier: str | None) -> str:
    if not isinstance(qualifier, str):
        return ""
    return qualifier.strip().lower()


def _object_track_key(object_name: str, qualifier: str | None) -> str:
    base = object_name.strip().lower()
    qual = _normalize_qualifier(qualifier)
    return f"{base}::{qual}" if qual else base


def _sanitize_session_fragment(session_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_id)


def _snapshot_path_for_session(session_id: str) -> Path:
    safe_id = _sanitize_session_fragment(session_id)
    if NAV_SESSION_MAP_SNAPSHOT_PATH:
        if "{session_id}" in NAV_SESSION_MAP_SNAPSHOT_PATH:
            return Path(NAV_SESSION_MAP_SNAPSHOT_PATH.format(session_id=safe_id))
        if safe_id == "default":
            return Path(NAV_SESSION_MAP_SNAPSHOT_PATH)
        base, ext = os.path.splitext(NAV_SESSION_MAP_SNAPSHOT_PATH)
        suffix = ext if ext else ".json"
        return Path(f"{base}_{safe_id}{suffix}")
    return Path(NAV_SESSION_MAP_SNAPSHOT_DIR) / f"tbot_session_map_{safe_id}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _serialize_session_state(state: dict[str, Any], include_history: bool = False) -> dict[str, Any]:
    objects = [dict(track) for track in state.get("objects", {}).values()]
    payload: dict[str, Any] = {
        "status": "ok",
        "session_id": state.get("session_id"),
        "active": True,
        "started_at_mono_s": state.get("started_at_mono_s"),
        "updated_at_unix_s": time.time(),
        "initial_pose": state.get("initial_pose"),
        "current_pose": state.get("current_pose"),
        "object_count": len(objects),
        "objects": objects,
        "snapshot_path": str(_snapshot_path_for_session(str(state.get("session_id", "default")))),
        "meta": {
            "object_ttl_s": NAV_MAP_OBJECT_TTL_S,
            "trail_max_points": NAV_SESSION_MAP_TRAIL_MAX_POINTS,
        },
    }
    if include_history:
        payload["robot_trail"] = list(state.get("robot_trail", []))
    return payload


def _persist_active_session_snapshot() -> None:
    if not NAV_SESSION_MAP_PERSIST_ENABLED:
        return
    with _session_state_lock:
        state = dict(_active_session_state) if isinstance(_active_session_state, dict) else None
        if state is None:
            return
        # Keep nested copies immutable outside the lock.
        state["objects"] = {
            str(key): dict(value) for key, value in dict(_active_session_state.get("objects", {})).items()  # type: ignore[union-attr]
        }
        state["robot_trail"] = list(_active_session_state.get("robot_trail", []))  # type: ignore[union-attr]
    payload = _serialize_session_state(state, include_history=True)
    _atomic_write_json(_snapshot_path_for_session(str(state.get("session_id", "default"))), payload)


def _session_upsert_current_pose(x_m: float, y_m: float, yaw_rad: float, stamp_sec: float | None = None) -> None:
    with _session_state_lock:
        if not isinstance(_active_session_state, dict):
            return
        pose = {
            "x_m": float(x_m),
            "y_m": float(y_m),
            "yaw_rad": float(yaw_rad),
            "stamp_sec": float(stamp_sec) if isinstance(stamp_sec, (int, float)) and math.isfinite(float(stamp_sec)) else None,
            "updated_at_mono_s": time.monotonic(),
        }
        _active_session_state["current_pose"] = pose
        if _active_session_state.get("initial_pose") is None:
            _active_session_state["initial_pose"] = dict(pose)
        trail = _active_session_state.setdefault("robot_trail", [])
        if isinstance(trail, list):
            trail.append({"x_m": pose["x_m"], "y_m": pose["y_m"], "yaw_rad": pose["yaw_rad"]})
            if len(trail) > NAV_SESSION_MAP_TRAIL_MAX_POINTS:
                del trail[:-NAV_SESSION_MAP_TRAIL_MAX_POINTS]
    _persist_active_session_snapshot()


def _session_upsert_object_track(
    object_name: str,
    qualifier: str | None,
    object_x_m: float,
    object_y_m: float,
    distance_m: float,
    heading_deg: float,
    confidence: str,
    source: str,
) -> None:
    with _session_state_lock:
        if not isinstance(_active_session_state, dict):
            return
        key = _object_track_key(object_name, qualifier)
        tracks = _active_session_state.setdefault("objects", {})
        if not isinstance(tracks, dict):
            return
        existing = tracks.get(key, {})
        seen_count = int(existing.get("seen_count", 0)) + 1 if isinstance(existing, dict) else 1
        track = {
            "key": key,
            "object_name": object_name,
            "qualifier": qualifier.strip() if isinstance(qualifier, str) and qualifier.strip() else None,
            "estimated_object_pose": {"x_m": float(object_x_m), "y_m": float(object_y_m)},
            "distance_m": float(distance_m),
            "heading_deg": float(heading_deg),
            "confidence": confidence,
            "source": source,
            "seen_count": seen_count,
            "last_seen_unix_s": time.time(),
            "last_seen_mono_s": time.monotonic(),
        }
        tracks[key] = track
    _persist_active_session_snapshot()


def _session_get_fresh_object_track(
    object_name: str,
    qualifier: str | None,
    ttl_s: float = NAV_MAP_OBJECT_TTL_S,
) -> dict[str, Any] | None:
    with _session_state_lock:
        if not isinstance(_active_session_state, dict):
            return None
        tracks = _active_session_state.get("objects")
        if not isinstance(tracks, dict):
            return None
        key = _object_track_key(object_name, qualifier)
        track = tracks.get(key)
        if not isinstance(track, dict):
            return None
        seen_mono = track.get("last_seen_mono_s")
        if not isinstance(seen_mono, (int, float)) or not math.isfinite(float(seen_mono)):
            return None
        if time.monotonic() - float(seen_mono) > float(ttl_s):
            return None
        return dict(track)


def _track_to_pose_estimate(track: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(track, dict):
        return None
    pose = track.get("estimated_object_pose")
    if not isinstance(pose, dict):
        return None
    x_m = pose.get("x_m")
    y_m = pose.get("y_m")
    if not isinstance(x_m, (int, float)) or not isinstance(y_m, (int, float)):
        return None
    heading = track.get("heading_deg")
    distance = track.get("distance_m")
    confidence = track.get("confidence", "low")
    return {
        "success": True,
        "x": float(x_m),
        "y": float(y_m),
        "heading_deg": float(heading) if isinstance(heading, (int, float)) else None,
        "distance_m": float(distance) if isinstance(distance, (int, float)) else None,
        "confidence": confidence,
        "source": "session_map",
    }


def _session_has_active_state() -> bool:
    with _session_state_lock:
        return isinstance(_active_session_state, dict)


async def _session_capture_current_pose() -> dict[str, Any] | None:
    if not _session_has_active_state():
        return None
    pose = await tbot_nav_get_pose()
    if pose.get("status") != "ok":
        return None
    _session_upsert_current_pose(
        x_m=float(pose["x_m"]),
        y_m=float(pose["y_m"]),
        yaw_rad=float(pose["yaw_rad"]),
        stamp_sec=float(pose["stamp_sec"]) if isinstance(pose.get("stamp_sec"), (int, float)) else None,
    )
    return pose


def _session_snapshot(include_history: bool = False) -> dict[str, Any]:
    with _session_state_lock:
        if not isinstance(_active_session_state, dict):
            return {"status": "no_session", "active": False}
        state = dict(_active_session_state)
        state["objects"] = {str(k): dict(v) for k, v in dict(_active_session_state.get("objects", {})).items()}
        state["robot_trail"] = list(_active_session_state.get("robot_trail", []))
    return _serialize_session_state(state, include_history=include_history)


def _session_upsert_current_pose_from_nav_pose(pose: dict[str, Any]) -> None:
    if not isinstance(pose, dict):
        return
    status = pose.get("status")
    if status is not None and status != "ok":
        return
    x_m = pose.get("x_m")
    y_m = pose.get("y_m")
    yaw_rad = pose.get("yaw_rad")
    if not isinstance(x_m, (int, float)) or not isinstance(y_m, (int, float)) or not isinstance(yaw_rad, (int, float)):
        return
    stamp = pose.get("stamp_sec")
    _session_upsert_current_pose(
        x_m=float(x_m),
        y_m=float(y_m),
        yaw_rad=float(yaw_rad),
        stamp_sec=float(stamp) if isinstance(stamp, (int, float)) and math.isfinite(float(stamp)) else None,
    )


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


def _extract_sector_distance(payload: dict[str, Any], sector: str) -> float | None:
    direct = payload.get("distance_m")
    if isinstance(direct, (int, float)) and math.isfinite(float(direct)):
        return float(direct)
    distances = payload.get("distances")
    if isinstance(distances, dict):
        value = distances.get(sector)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _extract_valid_distance(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _qualifier_to_attributes(qualifier: str | None) -> list[str] | None:
    if isinstance(qualifier, str):
        cleaned = qualifier.strip()
        if cleaned:
            return [cleaned]
    return None


def _qualifier_from_list(qualifiers: list[str] | None, index: int) -> str | None:
    if not isinstance(qualifiers, list):
        return None
    if index < 0 or index >= len(qualifiers):
        return None
    value = qualifiers[index]
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


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
        {"front_threshold_m": NAV_FRONT_THRESHOLD_M},
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
    payload: dict[str, Any] = {"object_name": target}
    attrs = _qualifier_to_attributes(qualifier)
    if attrs:
        payload["attributes"] = attrs
    raw = await vision.call_tool("tbot_vision_find_object", payload)
    return _extract_tool_result_dict(raw)


async def _vision_get_target_bbox(
    vision: Client,
    target: str,
    qualifier: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"object_name": target}
    attrs = _qualifier_to_attributes(qualifier)
    if attrs:
        payload["attributes"] = attrs
    raw = await vision.call_tool("tbot_vision_get_object_bbox", payload)
    return _extract_tool_result_dict(raw)


async def _scene_confirms_target_visible(
    vision: Client,
    target: str,
    qualifier: str | None,
) -> bool:
    qualifier_text = f" with qualifier '{qualifier.strip()}'" if isinstance(qualifier, str) and qualifier.strip() else ""
    prompt = (
        "Return JSON only with keys visible (boolean) and confidence (0..1). "
        f"Is the target object '{target}'{qualifier_text} currently visible in frame?"
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
async def tbot_nav_start_session(
    session_id: str | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    """
    Start or refresh a navigation session map and capture initial/current pose.
    """
    session_id_clean = _normalize_session_id(session_id)
    pose = await tbot_nav_get_pose()
    if pose.get("status") != "ok":
        return {
            "status": "no_odom",
            "session_id": session_id_clean,
            "active": False,
            "error": pose.get("error", "no_odom"),
        }

    current_pose = {
        "x_m": float(pose["x_m"]),
        "y_m": float(pose["y_m"]),
        "yaw_rad": float(pose["yaw_rad"]),
        "stamp_sec": float(pose["stamp_sec"]) if isinstance(pose.get("stamp_sec"), (int, float)) else None,
        "updated_at_mono_s": time.monotonic(),
    }

    global _active_session_state
    with _session_state_lock:
        prior = _active_session_state if isinstance(_active_session_state, dict) else None
        reuse_prior = (
            prior is not None
            and not bool(reset)
            and str(prior.get("session_id")) == session_id_clean
        )
        if reuse_prior:
            initial_pose = dict(prior.get("initial_pose") or current_pose)
            objects = {
                str(key): dict(value)
                for key, value in dict(prior.get("objects", {})).items()
            }
            trail = list(prior.get("robot_trail", []))
            started_at = float(prior.get("started_at_mono_s", time.monotonic()))
        else:
            initial_pose = dict(current_pose)
            objects = {}
            trail = []
            started_at = time.monotonic()

        trail.append(
            {
                "x_m": current_pose["x_m"],
                "y_m": current_pose["y_m"],
                "yaw_rad": current_pose["yaw_rad"],
            }
        )
        if len(trail) > NAV_SESSION_MAP_TRAIL_MAX_POINTS:
            del trail[:-NAV_SESSION_MAP_TRAIL_MAX_POINTS]

        _active_session_state = {
            "session_id": session_id_clean,
            "started_at_mono_s": started_at,
            "initial_pose": initial_pose,
            "current_pose": current_pose,
            "objects": objects,
            "robot_trail": trail,
        }

    _persist_active_session_snapshot()
    return _session_snapshot(include_history=True)


@mcp_nav_v3.tool()
async def tbot_nav_get_session_map(include_history: bool = False) -> dict[str, Any]:
    """
    Return the active session map (initial pose, current pose, tracked objects).
    """
    await _session_capture_current_pose()
    return _session_snapshot(include_history=bool(include_history))


@mcp_nav_v3.tool()
async def tbot_nav_reset_session_map(session_id: str | None = None) -> dict[str, Any]:
    """
    Clear tracked objects and reset the active session baseline pose.
    """
    return await tbot_nav_start_session(session_id=session_id, reset=True)


@mcp_nav_v3.tool()
async def tbot_nav_inspect_and_map_objects(
    objects: list[str],
    qualifiers: list[str] | None = None,
) -> dict[str, Any]:
    """
    Inspect a list of objects and upsert LiDAR-backed pose estimates into the session map.
    """
    if not isinstance(objects, list) or not objects:
        raise ValueError("objects must be a non-empty list of strings")
    if isinstance(qualifiers, list) and len(qualifiers) not in (0, len(objects)):
        raise ValueError("qualifiers must be empty or match objects length")

    if not _session_has_active_state():
        start_result = await tbot_nav_start_session()
        if start_result.get("status") != "ok":
            return {
                "status": "no_session",
                "error": start_result.get("error", "failed_to_start_session"),
                "session": start_result,
                "results": [],
            }

    results: list[dict[str, Any]] = []
    mapped_count = 0
    for idx, raw_name in enumerate(objects):
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if not name:
            results.append(
                {
                    "object_name": raw_name,
                    "qualifier": _qualifier_from_list(qualifiers, idx),
                    "success": False,
                    "error": "invalid_object_name",
                }
            )
            continue
        qualifier = _qualifier_from_list(qualifiers, idx)
        estimate = await tbot_estimate_object_pose(target=name, qualifier=qualifier)
        mapped = bool(estimate.get("success"))
        if mapped:
            mapped_count += 1
        results.append(
            {
                "object_name": name,
                "qualifier": qualifier,
                "success": mapped,
                "estimate": estimate,
            }
        )

    snapshot = _session_snapshot(include_history=False)
    return {
        "status": "ok",
        "requested_count": len(objects),
        "mapped_count": mapped_count,
        "results": results,
        "session": snapshot,
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
                _session_upsert_current_pose(
                    x_m=x_cur,
                    y_m=y_cur,
                    yaw_rad=yaw_cur,
                    stamp_sec=float(odom["stamp_sec"]) if isinstance(odom.get("stamp_sec"), (int, float)) else None,
                )

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
                            collision_raw = await lidar.call_tool(
                                "tbot_lidar_check_collision",
                                {"front_threshold_m": NAV_FRONT_THRESHOLD_M},
                            )
                            collision = _extract_tool_result_dict(collision_raw)
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
                        max_duration_for_distance = distance / max(segment_speed, 1e-6)
                        segment_duration = min(NAV_CONTINUOUS_SEGMENT_S, max_duration_for_distance)
                        if segment_duration <= 0.0:
                            await asyncio.sleep(0.05)
                            continue
                        await motion.call_tool(
                            "tbot_motion_move_forward_continuous",
                            {"duration_seconds": segment_duration, "speed": segment_speed},
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

    _session_upsert_current_pose(
        x_m=robot_x,
        y_m=robot_y,
        yaw_rad=float(pose["yaw_rad"]),
        stamp_sec=float(pose["stamp_sec"]) if isinstance(pose.get("stamp_sec"), (int, float)) else None,
    )
    _session_upsert_object_track(
        object_name=target_clean,
        qualifier=qualifier,
        object_x_m=object_x,
        object_y_m=object_y,
        distance_m=lidar_distance_m,
        heading_deg=object_heading_deg,
        confidence=confidence,
        source="tbot_estimate_object_pose",
    )

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

    map_usage: dict[str, Any] = {
        "object_1_source": "estimated_live",
        "object_2_source": "estimated_live",
        "used_session_map": False,
    }

    pose_1_track = _session_get_fresh_object_track(object_1_clean, qualifier_1, NAV_MAP_OBJECT_TTL_S)
    pose_1 = _track_to_pose_estimate(pose_1_track) if pose_1_track is not None else None
    if pose_1 is not None:
        map_usage["object_1_source"] = "session_map"
        map_usage["used_session_map"] = True
    else:
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
            "map_usage": map_usage,
            "used_bypass": False,
            "bypass_result": {},
            "nav_result": None,
        }

    pose_2_track = _session_get_fresh_object_track(object_2_clean, qualifier_2, NAV_MAP_OBJECT_TTL_S)
    pose_2 = _track_to_pose_estimate(pose_2_track) if pose_2_track is not None else None
    if pose_2 is not None:
        map_usage["object_2_source"] = "session_map"
        map_usage["used_session_map"] = True
    else:
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
            "map_usage": map_usage,
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
            "map_usage": map_usage,
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
                "map_usage": map_usage,
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
                "map_usage": map_usage,
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
        "map_usage": map_usage,
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

    await _session_capture_current_pose()
    final_pose = await tbot_nav_get_pose()
    _session_upsert_current_pose_from_nav_pose(final_pose)
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
                    _session_upsert_current_pose_from_nav_pose(final_pose)
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
                if bool(initial_bbox_view.get("visible")):
                    await _recenter_to_visible_target_if_needed(motion, initial_bbox_view)

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
                        visible_for_stop = await _scene_confirms_target_visible(
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
                            _session_upsert_current_pose_from_nav_pose(final_pose)
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
                            _session_upsert_current_pose_from_nav_pose(final_pose)
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

                    near_target = distance_for_limit is not None and distance_for_limit <= stop_distance_f
                    blocked = allowed_step <= 0.0
                    if near_target or blocked:
                        visible_for_stop = await _scene_confirms_target_visible(
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
                            _session_upsert_current_pose_from_nav_pose(final_pose)
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
                    move_duration_s = min(
                        NAV_CONTINUOUS_SEGMENT_S,
                        allowed_step / max(move_speed, 1e-6),
                    )
                    await motion.call_tool(
                        "tbot_motion_move_forward_continuous",
                        {"duration_seconds": move_duration_s, "speed": move_speed},
                    )
                    approach_steps += 1

                    post_move_visible = await _scene_confirms_target_visible(
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
                        confirm_visible = await _scene_confirms_target_visible(
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
                        _session_upsert_current_pose_from_nav_pose(final_pose)
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
                    _session_upsert_current_pose_from_nav_pose(final_pose)
                    return {
                        "success": False,
                        "final_pose": final_pose,
                        "object_in_frame": False,
                        "scene_description": "",
                        "stopped_reason": "max_retries",
                    }

                if not object_in_frame:
                    object_in_frame = await _scene_confirms_target_visible(
                        vision=vision,
                        target=target_clean,
                        qualifier=qualifier,
                    )
                # Refresh object map entry at arrival so planners can reuse it.
                if _session_has_active_state():
                    try:
                        await tbot_estimate_object_pose(target=target_clean, qualifier=qualifier)
                    except Exception:
                        pass
                if confirm_in_frame:
                    scene_raw = await vision.call_tool(
                        "tbot_vision_describe_scene",
                        {"prompt": f"Confirm the {target_clean} and nearby context."},
                    )
                    scene = _extract_tool_result_dict(scene_raw)
                    scene_description = str(scene.get("description", ""))

    final_pose = await tbot_nav_get_pose()
    _session_upsert_current_pose_from_nav_pose(final_pose)
    return {
        "success": (stopped_reason == "reached_target" and (not confirm_in_frame or object_in_frame)),
        "final_pose": final_pose,
        "object_in_frame": object_in_frame,
        "scene_description": scene_description,
        "stopped_reason": stopped_reason,
    }


@mcp_nav_v3.tool()
async def tbot_nav_plan_around_object(
    object_name: str,
    qualifier: str | None = None,
    preferred_side: str = "auto",
) -> dict[str, Any]:
    """
    Plan and execute a bypass around a mapped object while updating session pose.
    """
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")
    preferred_clean = preferred_side.strip().lower() if isinstance(preferred_side, str) else "auto"
    if preferred_clean not in ("left", "right", "auto"):
        raise ValueError("preferred_side must be 'left', 'right', or 'auto'")

    track = _session_get_fresh_object_track(object_name_clean, qualifier, NAV_MAP_OBJECT_TTL_S)
    pose_source = "session_map"
    pose_estimate = _track_to_pose_estimate(track) if track is not None else None
    if pose_estimate is None:
        pose_source = "live_estimate"
        pose_estimate = await tbot_estimate_object_pose(target=object_name_clean, qualifier=qualifier)
        if not bool(pose_estimate.get("success")):
            return {
                "success": False,
                "status": "object_not_localized",
                "object_name": object_name_clean,
                "qualifier": qualifier,
                "pose_source": pose_source,
                "object_pose": pose_estimate,
            }

    left_distance: float | None = None
    right_distance: float | None = None
    lidar_error: str | None = None
    try:
        async with Client(LIDAR_MCP_URL_V3) as lidar:
            left_raw = await lidar.call_tool("tbot_lidar_get_obstacle_distances", {"sector": "left"})
            right_raw = await lidar.call_tool("tbot_lidar_get_obstacle_distances", {"sector": "right"})
        left_result = _extract_tool_result_dict(left_raw)
        right_result = _extract_tool_result_dict(right_raw)
        left_distance = _extract_sector_distance(left_result, "left")
        right_distance = _extract_sector_distance(right_result, "right")
    except Exception as e:
        lidar_error = str(e)

    chosen_side = preferred_clean
    if chosen_side == "auto":
        left_cmp = left_distance if isinstance(left_distance, (int, float)) else -1.0
        right_cmp = right_distance if isinstance(right_distance, (int, float)) else -1.0
        chosen_side = "left" if left_cmp >= right_cmp else "right"

    bypass_result: dict[str, Any] = {}
    try:
        async with Client(MOTION_MCP_URL_V3) as motion:
            raw = await motion.call_tool("tbot_motion_bypass_obstacle", {"preferred_side": chosen_side})
            bypass_result = _extract_tool_result_dict(raw)
    except Exception as e:
        bypass_result = {"status": "error", "error": str(e)}

    final_pose = await _session_capture_current_pose()
    return {
        "success": bypass_result.get("status") == "completed",
        "status": bypass_result.get("status", "error"),
        "object_name": object_name_clean,
        "qualifier": qualifier,
        "pose_source": pose_source,
        "object_pose": pose_estimate,
        "chosen_side": chosen_side,
        "left_distance_m": left_distance,
        "right_distance_m": right_distance,
        "lidar_error": lidar_error,
        "bypass_result": bypass_result,
        "final_pose": final_pose,
        "session": _session_snapshot(include_history=False),
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
