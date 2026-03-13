"""
TurtleBot Motion MCP Server V3.

Duration-based motion commands via the HTTP API exposed by the motion server.
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
from typing import Any

import httpx
from fastmcp import Client, FastMCP

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


BASE_URL = os.getenv("MOTION_SERVER_BASE_URL", "http://10.158.38.26:5001").rstrip("/")
HTTP_TIMEOUT_S = _env_float("MOTION_TIMEOUT_S", 5.0)
MAX_LINEAR = abs(_env_float("MOTION_MAX_LINEAR", 0.2))
MAX_ANGULAR = abs(_env_float("MOTION_MAX_ANGULAR", 1.0))
ANGULAR_SIGN = _env_float("MOTION_ANGULAR_SIGN", -1.0)

MOTION_COMMAND_REFRESH_S_RAW = _env_float("MOTION_COMMAND_REFRESH_S", 0.05)
MOTION_COMMAND_REFRESH_S = (
    MOTION_COMMAND_REFRESH_S_RAW if MOTION_COMMAND_REFRESH_S_RAW > 0 else 0.05
)

MOVE_PATH = "/move"
STOP_PATH = "/stop"
HEALTH_PATH = "/health"
LIDAR_MCP_URL = os.getenv("TBOT_LIDAR_MCP_URL_V3", "http://127.0.0.1:18212/turtlebot-lidar-v3")

mcp_motion_v3 = FastMCP("TurtleBot Motion MCP Server V3")
_continuous_motion_thread: threading.Thread | None = None
_continuous_motion_stop_event: threading.Event | None = None
_continuous_motion_lock = threading.Lock()


def _validate_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _clamp(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


async def _post_move_for_duration(
    linear: float,
    angular: float,
    duration_s: float,
) -> tuple[dict[str, Any] | None, int]:
    """
    Keep reposting /move for duration_s so motion backends that require
    periodic updates keep executing the command.
    """
    start_mono = time.monotonic()
    posts = 0
    last_result: dict[str, Any] | None = None

    # Guarantee at least one post even for very short durations.
    effective_duration_s = max(duration_s, MOTION_COMMAND_REFRESH_S)
    deadline = start_mono + effective_duration_s

    while True:
        last_result = await _post_json(MOVE_PATH, {"linear": linear, "angular": angular})
        posts += 1

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(MOTION_COMMAND_REFRESH_S, remaining))

    return last_result, posts


def _post_json_sync(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_S) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Unexpected response from {url}: expected JSON object")
            return data
    except httpx.HTTPStatusError as e:
        body = e.response.text
        raise RuntimeError(f"Motion server error {e.response.status_code} for {url}: {body[:4000]}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to reach motion server at {url}: {e}") from e


def _continuous_motion_loop(linear: float, angular: float, stop_event: threading.Event) -> None:
    while True:
        if stop_event.is_set():
            return
        try:
            _post_json_sync(MOVE_PATH, {"linear": linear, "angular": angular})
        except Exception as e:
            logger.warning("Continuous motion update failed: %s", e)

        # Use event.wait so stop requests interrupt the sleep immediately.
        if stop_event.wait(MOTION_COMMAND_REFRESH_S):
            return


async def _set_continuous_motion(linear: float | None, angular: float = 0.0) -> bool:
    """
    Start or stop background /move keepalive updates.
    Returns True if a previous background stream existed and was replaced/stopped.
    """
    global _continuous_motion_thread, _continuous_motion_stop_event

    with _continuous_motion_lock:
        old_thread = _continuous_motion_thread
        old_stop_event = _continuous_motion_stop_event
        had_existing_stream = old_thread is not None
        _continuous_motion_thread = None
        _continuous_motion_stop_event = None

    if old_stop_event is not None:
        old_stop_event.set()
    if old_thread is not None:
        try:
            await asyncio.to_thread(old_thread.join, 1.0)
        except Exception as e:
            logger.warning("Continuous motion thread stop failed: %s", e)

    if linear is None:
        return had_existing_stream

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_continuous_motion_loop,
        args=(linear, angular, stop_event),
        daemon=True,
    )
    with _continuous_motion_lock:
        _continuous_motion_thread = thread
        _continuous_motion_stop_event = stop_event
    thread.start()
    return had_existing_stream


async def _post_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Unexpected response from {url}: expected JSON object")
            return data
    except httpx.HTTPStatusError as e:
        body = e.response.text
        raise RuntimeError(f"Motion server error {e.response.status_code} for {url}: {body[:4000]}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to reach motion server at {url}: {e}") from e


async def _get_json(path: str) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Unexpected response from {url}: expected JSON object")
            return data
    except httpx.HTTPStatusError as e:
        body = e.response.text
        raise RuntimeError(f"Motion server error {e.response.status_code} for {url}: {body[:4000]}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to reach motion server at {url}: {e}") from e


async def _try_get_health() -> tuple[dict[str, Any] | None, str | None]:
    try:
        return await _get_json(HEALTH_PATH), None
    except Exception as e:
        return None, str(e)


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


async def _get_collision_snapshot(lidar: Client | None = None) -> dict[str, Any]:
    if lidar is None:
        async with Client(LIDAR_MCP_URL) as lidar_client:
            raw = await lidar_client.call_tool("tbot_lidar_check_collision", {})
    else:
        raw = await lidar.call_tool("tbot_lidar_check_collision", {})
    return _extract_tool_result_dict(raw)


def _extract_front_distance(collision: dict[str, Any]) -> float | None:
    value = collision.get("min_forward_distance_m")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    distances = collision.get("distances")
    if isinstance(distances, dict):
        front = distances.get("front")
        if isinstance(front, (int, float)) and math.isfinite(float(front)):
            return float(front)
    return None


async def _execute_forward_distance(distance_m: float, speed: float) -> dict[str, Any]:
    """Move forward by a precomputed distance using duration-based execution."""
    distance_f = _validate_finite("distance_m", distance_m)
    speed_f = _validate_finite("speed", speed)
    if distance_f <= 0:
        raise ValueError("distance_m must be > 0")

    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    if clamped_speed == 0.0:
        raise ValueError(
            f"Effective speed is 0 (speed={speed_f!r}, MAX_LINEAR={MAX_LINEAR!r}). "
            "Pass a non-zero speed within the allowed range."
        )

    duration_s = distance_f / clamped_speed
    preempted_stream = await _set_continuous_motion(None)
    move_result, posts = await _post_move_for_duration(clamped_speed, 0.0, duration_s)
    stop_result = await _post_json(STOP_PATH)
    return {
        **stop_result,
        "status": "completed",
        "distance_m": distance_f,
        "speed": clamped_speed,
        "requested_speed": speed_f,
        "duration_s": duration_s,
        "move_posts": posts,
        "command_refresh_s": MOTION_COMMAND_REFRESH_S,
        "move_result": move_result,
        "was_clamped": abs(speed_f) != clamped_speed,
        "preempted_continuous_stream": preempted_stream,
    }


@mcp_motion_v3.tool()
async def tbot_motion_stop() -> dict[str, Any]:
    """Stop the robot immediately by setting linear and angular targets to zero."""
    had_stream = await _set_continuous_motion(None)
    result = await _post_json(STOP_PATH)
    return {
        **result,
        "base_url": BASE_URL,
        "endpoint": STOP_PATH,
        "had_continuous_stream": had_stream,
    }


@mcp_motion_v3.tool()
async def tbot_motion_move_forward_distance(
    distance_m: float,
    speed: float = 0.1,
) -> dict[str, Any]:
    """
    Move forward by a precomputed travel distance.

    This tool does not perform LiDAR checks while moving. Planners should
    precompute distance using LiDAR before calling this tool.
    """
    return await _execute_forward_distance(distance_m=distance_m, speed=speed)


@mcp_motion_v3.tool()
async def tbot_motion_move_forward_continuous(
    duration_seconds: float,
    speed: float = 0.1,
) -> dict[str, Any]:
    """
    Move forward continuously for duration_seconds, then stop.

    This tool does not perform LiDAR checks while moving.
    """
    duration_f = _validate_finite("duration_seconds", duration_seconds)
    if duration_f <= 0:
        raise ValueError("duration_seconds must be > 0")

    speed_f = _validate_finite("speed", speed)
    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    if clamped_speed == 0.0:
        raise ValueError(
            f"Effective speed is 0 (speed={speed_f!r}, MAX_LINEAR={MAX_LINEAR!r}). "
            "Pass a non-zero speed within the allowed range."
        )

    preempted_stream = await _set_continuous_motion(clamped_speed, 0.0)
    started = time.monotonic()
    try:
        await asyncio.sleep(duration_f)
    finally:
        had_stream_on_stop = await _set_continuous_motion(None)
        stop_result = await _post_json(STOP_PATH)
    executed_duration = time.monotonic() - started

    return {
        **stop_result,
        "status": "completed",
        "duration_seconds": duration_f,
        "executed_duration_seconds": executed_duration,
        "speed": clamped_speed,
        "requested_speed": speed_f,
        "was_clamped": abs(speed_f) != clamped_speed,
        "preempted_continuous_stream": preempted_stream,
        "had_continuous_stream_on_stop": had_stream_on_stop,
        "command_refresh_s": MOTION_COMMAND_REFRESH_S,
    }


@mcp_motion_v3.tool()
async def tbot_motion_turn(
    direction: str,
    speed: float,
    duration_seconds: float,
) -> dict[str, Any]:
    """
    Turn the robot left or right at the given angular speed for duration_seconds, then stop.

    direction must be "left" or "right".
    speed is in rad/s (clamped to MOTION_MAX_ANGULAR, default 1.0).
    """
    direction_clean = direction.strip().lower() if isinstance(direction, str) else ""
    if direction_clean not in ("left", "right"):
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

    speed_f = _validate_finite("speed", speed)
    duration_f = _validate_finite("duration_seconds", duration_seconds)
    if duration_f <= 0:
        raise ValueError("duration_seconds must be > 0")

    clamped_speed = _clamp(abs(speed_f), MAX_ANGULAR)
    if clamped_speed == 0.0:
        raise ValueError(
            f"Effective angular speed is 0 (speed={speed_f!r}, MAX_ANGULAR={MAX_ANGULAR!r}). "
            "Pass a non-zero speed within the allowed range."
        )

    # Keep turn(direction=...) aligned with the robot angular convention
    # (+angular = left/CCW, -angular = right/CW).
    # MOTION_ANGULAR_SIGN remains the hardware/frame adapter.
    input_frame_sign = 1.0 if direction_clean == "right" else -1.0
    angular_cmd = input_frame_sign * clamped_speed * ANGULAR_SIGN

    if angular_cmd == 0.0:
        raise ValueError(
            f"angular_cmd is 0 — check MOTION_ANGULAR_SIGN env var "
            f"(current value: {ANGULAR_SIGN!r}). Must be non-zero."
        )

    preempted_stream = await _set_continuous_motion(None)
    move_result, posts = await _post_move_for_duration(0.0, angular_cmd, duration_f)

    # Verify the motion server actually received the angular command.
    health, _ = await _try_get_health()
    health_angular: float | None = None
    if health is not None:
        raw = health.get("angular")
        try:
            health_angular = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            pass

    command_received = (
        health_angular is not None and abs(health_angular - angular_cmd) < 1e-6
    )

    stop_result = await _post_json(STOP_PATH)
    return {
        **stop_result,
        "status": "completed",
        "direction": direction_clean,
        "speed": clamped_speed,
        "angular_cmd": angular_cmd,
        "angular_sign": ANGULAR_SIGN,
        "duration_seconds": duration_f,
        "was_clamped": abs(speed_f) != clamped_speed,
        "move_result": move_result,
        "move_posts": posts,
        "command_refresh_s": MOTION_COMMAND_REFRESH_S,
        "preempted_continuous_stream": preempted_stream,
        "health_angular_after_move": health_angular,
        "command_received_by_server": command_received,
    }


def _pick_bypass_side(preferred_side: str, left_distance_m: float | None, right_distance_m: float | None) -> str:
    side_clean = preferred_side.strip().lower() if isinstance(preferred_side, str) else "auto"
    if side_clean in ("left", "right"):
        return side_clean

    left = left_distance_m if isinstance(left_distance_m, (int, float)) else -1.0
    right = right_distance_m if isinstance(right_distance_m, (int, float)) else -1.0
    return "left" if left >= right else "right"


def _compute_bypass_turn_angle_deg(
    front_distance_m: float | None,
    side_distance_m: float | None,
    lateral_clearance_m: float,
    min_turn_angle_deg: float,
    max_turn_angle_deg: float,
) -> tuple[float, float, float]:
    front = float(front_distance_m) if isinstance(front_distance_m, (int, float)) else lateral_clearance_m
    side = float(side_distance_m) if isinstance(side_distance_m, (int, float)) else 0.0
    lateral_shift_m = max(0.0, lateral_clearance_m - side)
    raw_angle_deg = math.degrees(math.atan2(lateral_shift_m, max(front, 1e-6)))
    angle_deg = max(min_turn_angle_deg, min(max_turn_angle_deg, raw_angle_deg))
    return angle_deg, raw_angle_deg, lateral_shift_m


async def _sample_front_distance(lidar: Client | None = None) -> tuple[float | None, dict[str, Any]]:
    if lidar is None:
        async with Client(LIDAR_MCP_URL) as lidar_client:
            raw = await lidar_client.call_tool("tbot_lidar_get_obstacle_distances", {"sector": "all"})
    else:
        raw = await lidar.call_tool("tbot_lidar_get_obstacle_distances", {"sector": "all"})
    result = _extract_tool_result_dict(raw)
    distances = result.get("distances")
    front: float | None = None
    if isinstance(distances, dict):
        value = distances.get("front")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            front = float(value)
    return front, result


@mcp_motion_v3.tool()
async def tbot_motion_approach_until_close(
    target_distance_m: float = 0.1,
    stop_distance_m: float = 0.1,
    speed: float = 0.1,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    """
    Safely approach until the front LiDAR distance is near target_distance_m.
    Uses collision checks before and after a single planned forward segment.
    """
    target_f = _validate_finite("target_distance_m", target_distance_m)
    stop_f = _validate_finite("stop_distance_m", stop_distance_m)
    timeout_f = _validate_finite("timeout_s", timeout_s)
    speed_f = _validate_finite("speed", speed)
    if target_f <= 0:
        raise ValueError("target_distance_m must be > 0")
    if stop_f < 0:
        raise ValueError("stop_distance_m must be >= 0")
    if timeout_f <= 0:
        raise ValueError("timeout_s must be > 0")

    started = time.monotonic()
    try:
        async with Client(LIDAR_MCP_URL) as lidar:
            initial_collision = await _get_collision_snapshot(lidar)
            initial_front = _extract_front_distance(initial_collision)
            if initial_front is None:
                return {
                    "status": "collision_risk",
                    "initial_front_distance_m": None,
                    "front_distance": None,
                    "required_move_distance_m": None,
                    "requested_move_duration_s": None,
                    "executed_move_duration_s": 0.0,
                    "collision": initial_collision,
                }

            required_distance = max(0.0, initial_front - target_f)
            clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
            if clamped_speed <= 0:
                raise ValueError("speed must produce a non-zero command after clamping")
            requested_duration = required_distance / clamped_speed if required_distance > 0 else 0.0
            executed_duration = min(requested_duration, timeout_f)

            if required_distance <= 0.0:
                await _post_json(STOP_PATH)
                status = "completed" if target_f <= stop_f else "reached"
                return {
                    "status": status,
                    "target_distance_m": target_f,
                    "stop_distance_m": stop_f,
                    "initial_front_distance_m": initial_front,
                    "front_distance": initial_front,
                    "required_move_distance_m": 0.0,
                    "requested_move_duration_s": 0.0,
                    "executed_move_duration_s": 0.0,
                    "elapsed_s": time.monotonic() - started,
                    "move_posts": 0,
                }

            move_result = await _execute_forward_distance(distance_m=required_distance, speed=clamped_speed)
            elapsed = time.monotonic() - started
            timeout_hit = requested_duration > timeout_f or elapsed >= timeout_f

            final_collision = await _get_collision_snapshot(lidar)
            final_front = _extract_front_distance(final_collision)
    except Exception as e:
        return {
            "status": "lidar_unavailable",
            "initial_front_distance_m": None,
            "front_distance": None,
            "required_move_distance_m": None,
            "requested_move_duration_s": None,
            "executed_move_duration_s": 0.0,
            "error": str(e),
        }

    if timeout_hit:
        status = "timeout"
    elif isinstance(final_collision, dict) and final_collision.get("risk_level") == "stop":
        status = "collision_risk"
    else:
        status = "completed" if target_f <= stop_f else "reached"

    return {
        "status": status,
        "target_distance_m": target_f,
        "stop_distance_m": stop_f,
        "initial_front_distance_m": initial_front,
        "front_distance": final_front if final_front is not None else initial_front,
        "required_move_distance_m": required_distance,
        "requested_move_duration_s": requested_duration,
        "executed_move_duration_s": executed_duration,
        "elapsed_s": elapsed,
        "move_posts": move_result.get("move_posts"),
        "move_result": move_result,
        "collision": final_collision,
    }


@mcp_motion_v3.tool()
async def tbot_motion_bypass_obstacle(
    preferred_side: str = "auto",
    speed: float = 0.1,
    turn_speed: float = 0.4,
    lateral_clearance_m: float = 0.45,
    min_turn_angle_deg: float = 15.0,
    max_turn_angle_deg: float = 60.0,
    parallel_front_clearance_m: float = 0.6,
    final_front_clearance_m: float = 0.9,
    max_first_leg_s: float = 3.0,
    max_second_leg_s: float = 4.0,
) -> dict[str, Any]:
    """
    Perform a two-leg obstacle bypass maneuver using LiDAR front/side clearances.
    """
    speed_f = _clamp(abs(_validate_finite("speed", speed)), MAX_LINEAR)
    turn_speed_f = _clamp(abs(_validate_finite("turn_speed", turn_speed)), MAX_ANGULAR)
    lateral_clearance = _validate_finite("lateral_clearance_m", lateral_clearance_m)
    min_angle = _validate_finite("min_turn_angle_deg", min_turn_angle_deg)
    max_angle = _validate_finite("max_turn_angle_deg", max_turn_angle_deg)
    parallel_clearance = _validate_finite("parallel_front_clearance_m", parallel_front_clearance_m)
    final_clearance = _validate_finite("final_front_clearance_m", final_front_clearance_m)
    max_first_leg = _validate_finite("max_first_leg_s", max_first_leg_s)
    max_second_leg = _validate_finite("max_second_leg_s", max_second_leg_s)
    if speed_f <= 0:
        raise ValueError("speed must be > 0 after clamping")
    if turn_speed_f <= 0:
        raise ValueError("turn_speed must be > 0 after clamping")

    async with Client(LIDAR_MCP_URL) as lidar:
        front0, first_scan = await _sample_front_distance(lidar)
        distances0 = first_scan.get("distances") if isinstance(first_scan.get("distances"), dict) else {}
        left0 = distances0.get("left") if isinstance(distances0, dict) else None
        right0 = distances0.get("right") if isinstance(distances0, dict) else None
        if isinstance(front0, (int, float)) and front0 >= final_clearance:
            await _post_json(STOP_PATH)
            return {
                "status": "no_obstacle",
                "initial_distances": distances0,
                "front_distance_m": front0,
            }

        chosen_side = _pick_bypass_side(preferred_side, left0 if isinstance(left0, (int, float)) else None, right0 if isinstance(right0, (int, float)) else None)
        side_distance = left0 if chosen_side == "left" else right0
        turn_angle_deg, raw_turn_angle_deg, lateral_shift_m = _compute_bypass_turn_angle_deg(
            front_distance_m=front0,
            side_distance_m=side_distance if isinstance(side_distance, (int, float)) else None,
            lateral_clearance_m=lateral_clearance,
            min_turn_angle_deg=min_angle,
            max_turn_angle_deg=max_angle,
        )

        turn_duration = math.radians(turn_angle_deg) / max(turn_speed_f, 1e-6)
        opposite_side = "right" if chosen_side == "left" else "left"

        await tbot_motion_turn(direction=chosen_side, speed=turn_speed_f, duration_seconds=turn_duration)
        await _set_continuous_motion(None)

        first_leg_status = "timeout"
        first_leg_started = time.monotonic()
        while (time.monotonic() - first_leg_started) < max_first_leg:
            await _post_json(MOVE_PATH, {"linear": speed_f, "angular": 0.0})
            front, _scan = await _sample_front_distance(lidar)
            if isinstance(front, (int, float)) and front >= parallel_clearance:
                first_leg_status = "parallel_reached"
                break
            await asyncio.sleep(MOTION_COMMAND_REFRESH_S)
        await _post_json(STOP_PATH)

        await tbot_motion_turn(direction=opposite_side, speed=turn_speed_f, duration_seconds=turn_duration)
        await _set_continuous_motion(None)

        second_leg_status = "timeout"
        second_leg_started = time.monotonic()
        front_end: float | None = None
        while (time.monotonic() - second_leg_started) < max_second_leg:
            await _post_json(MOVE_PATH, {"linear": speed_f, "angular": 0.0})
            front_end, _scan = await _sample_front_distance(lidar)
            if isinstance(front_end, (int, float)) and front_end >= final_clearance:
                second_leg_status = "clear_path_reached"
                break
            await asyncio.sleep(MOTION_COMMAND_REFRESH_S)
        await _post_json(STOP_PATH)

    status = "completed" if first_leg_status == "parallel_reached" and second_leg_status == "clear_path_reached" else "timeout"
    return {
        "status": status,
        "chosen_side": chosen_side,
        "turn_angle_deg": turn_angle_deg,
        "raw_turn_angle_deg": raw_turn_angle_deg,
        "lateral_shift_m": lateral_shift_m,
        "initial_distances": distances0,
        "first_leg_status": first_leg_status,
        "second_leg_status": second_leg_status,
        "final_front_distance_m": front_end,
    }


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18210,
    path: str = "/turtlebot-motion-v3",
    options: dict = {},
) -> None:
    """Run the TurtleBot Motion MCP Server V3."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting TurtleBot Motion MCP V3 base_url=%s timeout_s=%.2f at %s:%s%s",
        BASE_URL,
        HTTP_TIMEOUT_S,
        host,
        port,
        path,
    )
    mcp_motion_v3.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
