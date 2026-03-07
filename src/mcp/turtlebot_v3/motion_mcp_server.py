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

LIDAR_MCP_URL_V3 = os.getenv("TBOT_LIDAR_MCP_URL_V3", "http://127.0.0.1:18212/turtlebot-lidar-v3")

WALL_FOLLOW_KP = _env_float("WALL_FOLLOW_KP", 2.0)
WALL_FOLLOW_MAX_ANGULAR = _env_float("WALL_FOLLOW_MAX_ANGULAR", 0.5)
MOTION_COMMAND_REFRESH_S_RAW = _env_float("MOTION_COMMAND_REFRESH_S", 0.05)
MOTION_COMMAND_REFRESH_S = (
    MOTION_COMMAND_REFRESH_S_RAW if MOTION_COMMAND_REFRESH_S_RAW > 0 else 0.05
)
MOTION_COLLISION_POLL_S_RAW = _env_float("MOTION_COLLISION_POLL_S", 0.2)
MOTION_COLLISION_POLL_S = (
    MOTION_COLLISION_POLL_S_RAW if MOTION_COLLISION_POLL_S_RAW > 0 else 0.2
)
FORWARD_COLLISION_INTERRUPT_M = max(
    0.01,
    _env_float("MOTION_FORWARD_COLLISION_INTERRUPT_M", 0.10),
)
MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES = max(
    1,
    int(_env_float("MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES", 3.0)),
)
BYPASS_DEFAULT_LINEAR = abs(_env_float("MOTION_BYPASS_DEFAULT_LINEAR", 0.12))
BYPASS_DEFAULT_TURN_SPEED = abs(_env_float("MOTION_BYPASS_DEFAULT_TURN_SPEED", 0.5))
BYPASS_LATERAL_CLEARANCE_M = max(0.05, _env_float("MOTION_BYPASS_LATERAL_CLEARANCE_M", 0.45))
BYPASS_PARALLEL_FRONT_CLEARANCE_M = max(0.05, _env_float("MOTION_BYPASS_PARALLEL_FRONT_CLEARANCE_M", 0.6))
BYPASS_FINAL_FRONT_CLEARANCE_M = max(0.05, _env_float("MOTION_BYPASS_FINAL_FRONT_CLEARANCE_M", 0.9))
BYPASS_MIN_TURN_ANGLE_DEG = max(1.0, _env_float("MOTION_BYPASS_MIN_TURN_ANGLE_DEG", 15.0))
BYPASS_MAX_TURN_ANGLE_DEG = max(BYPASS_MIN_TURN_ANGLE_DEG, _env_float("MOTION_BYPASS_MAX_TURN_ANGLE_DEG", 60.0))
BYPASS_MIN_PARALLEL_TIME_S = max(0.1, _env_float("MOTION_BYPASS_MIN_PARALLEL_TIME_S", 0.6))
BYPASS_MIN_SECOND_LEG_TIME_S = max(0.1, _env_float("MOTION_BYPASS_MIN_SECOND_LEG_TIME_S", 0.6))

MOVE_PATH = "/move"
STOP_PATH = "/stop"
HEALTH_PATH = "/health"

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


def _extract_tool_dict(tool_result: Any) -> dict[str, Any]:
    """Extract a dict from a FastMCP Client.call_tool() ToolResult."""
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


def _to_distance_m(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


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


def _pick_bypass_side(
    preferred_side: str,
    left_m: float | None,
    right_m: float | None,
) -> str:
    side_clean = preferred_side.strip().lower() if isinstance(preferred_side, str) else ""
    if side_clean in ("left", "right"):
        return side_clean
    if left_m is None and right_m is None:
        return "left"
    if left_m is None:
        return "right"
    if right_m is None:
        return "left"
    return "left" if left_m >= right_m else "right"


def _compute_bypass_turn_angle_deg(
    front_distance_m: float | None,
    side_distance_m: float | None,
    lateral_clearance_m: float,
    min_turn_angle_deg: float,
    max_turn_angle_deg: float,
) -> tuple[float, float, float]:
    front_ref_m = max(0.05, front_distance_m if front_distance_m is not None else lateral_clearance_m)
    required_lateral_shift_m = (
        lateral_clearance_m if side_distance_m is None else max(0.0, lateral_clearance_m - side_distance_m)
    )
    raw_angle_deg = math.degrees(math.atan2(required_lateral_shift_m, front_ref_m))
    clamped_angle_deg = max(min_turn_angle_deg, min(max_turn_angle_deg, raw_angle_deg))
    return clamped_angle_deg, raw_angle_deg, required_lateral_shift_m


async def _read_lidar_distances(lidar: Client) -> tuple[dict[str, float | None], str | None]:
    try:
        raw = await lidar.call_tool("tbot_lidar_get_obstacle_distances", {"sector": "all"})
        parsed = _extract_tool_dict(raw)
    except Exception as e:
        return {"front": None, "left": None, "right": None, "rear": None}, str(e)

    distances_raw = parsed.get("distances")
    if not isinstance(distances_raw, dict):
        return {"front": None, "left": None, "right": None, "rear": None}, "missing_distances"

    return {
        "front": _to_distance_m(distances_raw.get("front")),
        "left": _to_distance_m(distances_raw.get("left")),
        "right": _to_distance_m(distances_raw.get("right")),
        "rear": _to_distance_m(distances_raw.get("rear")),
    }, None


@mcp_motion_v3.tool()
async def tbot_motion_health() -> dict[str, Any]:
    """Check whether the TurtleBot motion server is online."""
    health, health_error = await _try_get_health()
    if health is not None:
        return {
            **health,
            "status": "online",
            "base_url": BASE_URL,
            "endpoints": {"move": MOVE_PATH, "stop": STOP_PATH, "health": HEALTH_PATH},
            "reachable": True,
        }
    return {
        "status": "offline",
        "base_url": BASE_URL,
        "reachable": False,
        "endpoints": {"move": MOVE_PATH, "stop": STOP_PATH, "health": HEALTH_PATH},
        "health_error": health_error,
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
async def tbot_motion_bypass_obstacle(
    preferred_side: str = "auto",
    speed: float = BYPASS_DEFAULT_LINEAR,
    turn_speed: float = BYPASS_DEFAULT_TURN_SPEED,
    lateral_clearance_m: float = BYPASS_LATERAL_CLEARANCE_M,
    parallel_front_clearance_m: float = BYPASS_PARALLEL_FRONT_CLEARANCE_M,
    final_front_clearance_m: float = BYPASS_FINAL_FRONT_CLEARANCE_M,
    max_first_leg_s: float = 6.0,
    max_second_leg_s: float = 6.0,
    min_turn_angle_deg: float = BYPASS_MIN_TURN_ANGLE_DEG,
    max_turn_angle_deg: float = BYPASS_MAX_TURN_ANGLE_DEG,
) -> dict[str, Any]:
    """
    Bypass a blocking object using a two-rotation / two-forward maneuver.

    Sequence:
    1) LiDAR measures front and side distances.
    2) Compute turn angle to create lateral clearance.
    3) Rotate toward bypass side, then move forward until object is parallel.
    4) Rotate back, then move forward until front path is clear.
    """
    speed_f = _validate_finite("speed", speed)
    turn_speed_f = _validate_finite("turn_speed", turn_speed)
    lateral_clearance_f = _validate_finite("lateral_clearance_m", lateral_clearance_m)
    parallel_front_clearance_f = _validate_finite("parallel_front_clearance_m", parallel_front_clearance_m)
    final_front_clearance_f = _validate_finite("final_front_clearance_m", final_front_clearance_m)
    max_first_leg_f = _validate_finite("max_first_leg_s", max_first_leg_s)
    max_second_leg_f = _validate_finite("max_second_leg_s", max_second_leg_s)
    min_turn_deg_f = _validate_finite("min_turn_angle_deg", min_turn_angle_deg)
    max_turn_deg_f = _validate_finite("max_turn_angle_deg", max_turn_angle_deg)

    side_clean = preferred_side.strip().lower() if isinstance(preferred_side, str) else ""
    if side_clean not in ("auto", "left", "right"):
        raise ValueError("preferred_side must be 'auto', 'left', or 'right'")
    if lateral_clearance_f <= 0:
        raise ValueError("lateral_clearance_m must be > 0")
    if parallel_front_clearance_f <= 0:
        raise ValueError("parallel_front_clearance_m must be > 0")
    if final_front_clearance_f <= 0:
        raise ValueError("final_front_clearance_m must be > 0")
    if max_first_leg_f <= 0:
        raise ValueError("max_first_leg_s must be > 0")
    if max_second_leg_f <= 0:
        raise ValueError("max_second_leg_s must be > 0")
    if min_turn_deg_f <= 0:
        raise ValueError("min_turn_angle_deg must be > 0")
    if max_turn_deg_f < min_turn_deg_f:
        raise ValueError("max_turn_angle_deg must be >= min_turn_angle_deg")

    linear_cmd = _clamp(abs(speed_f), MAX_LINEAR)
    angular_speed_cmd = _clamp(abs(turn_speed_f), MAX_ANGULAR)
    if linear_cmd == 0.0:
        raise ValueError("speed is too small after clamping; pass a larger non-zero speed")
    if angular_speed_cmd == 0.0:
        raise ValueError("turn_speed is too small after clamping; pass a larger non-zero turn_speed")

    await _set_continuous_motion(None)

    first_leg_status = "not_started"
    second_leg_status = "not_started"
    first_leg_elapsed_s = 0.0
    second_leg_elapsed_s = 0.0
    first_leg_parallel_confirmations = 0
    second_leg_clear_confirmations = 0

    initial_distances: dict[str, float | None] = {"front": None, "left": None, "right": None, "rear": None}
    final_distances: dict[str, float | None] = {"front": None, "left": None, "right": None, "rear": None}
    chosen_side = "left"
    required_lateral_shift_m = 0.0
    raw_turn_angle_deg = 0.0
    turn_angle_deg = min_turn_deg_f

    try:
        async with Client(LIDAR_MCP_URL_V3) as lidar:
            initial_distances, lidar_error = await _read_lidar_distances(lidar)
            if lidar_error is not None:
                stop_result = await _post_json(STOP_PATH)
                return {
                    **stop_result,
                    "status": "lidar_unavailable",
                    "phase": "initial_scan",
                    "error": lidar_error,
                }

            front_start = initial_distances.get("front")
            left_start = initial_distances.get("left")
            right_start = initial_distances.get("right")

            chosen_side = _pick_bypass_side(side_clean, left_start, right_start)
            side_start = left_start if chosen_side == "left" else right_start

            turn_angle_deg, raw_turn_angle_deg, required_lateral_shift_m = _compute_bypass_turn_angle_deg(
                front_distance_m=front_start,
                side_distance_m=side_start,
                lateral_clearance_m=lateral_clearance_f,
                min_turn_angle_deg=min_turn_deg_f,
                max_turn_angle_deg=max_turn_deg_f,
            )

            # If the path is already clear, avoid unnecessary sidestep.
            if front_start is not None and front_start >= final_front_clearance_f:
                stop_result = await _post_json(STOP_PATH)
                return {
                    **stop_result,
                    "status": "no_obstacle",
                    "chosen_side": chosen_side,
                    "turn_angle_deg": turn_angle_deg,
                    "turn_angle_raw_deg": raw_turn_angle_deg,
                    "required_lateral_shift_m": required_lateral_shift_m,
                    "initial_distances": initial_distances,
                    "final_distances": initial_distances,
                }

            turn_duration_s = math.radians(turn_angle_deg) / angular_speed_cmd
            turn_out_direction = "left" if chosen_side == "left" else "right"
            turn_back_direction = "right" if chosen_side == "left" else "left"

            await tbot_motion_turn(
                direction=turn_out_direction,
                speed=angular_speed_cmd,
                duration_seconds=turn_duration_s,
            )

            first_leg_start = time.monotonic()
            first_leg_status = "timeout"
            first_leg_consecutive_lidar_failures = 0

            await _set_continuous_motion(linear_cmd, 0.0)
            try:
                while True:
                    first_leg_elapsed_s = time.monotonic() - first_leg_start
                    if first_leg_elapsed_s >= max_first_leg_f:
                        first_leg_status = "timeout"
                        break

                    distances, lidar_read_error = await _read_lidar_distances(lidar)
                    if lidar_read_error is not None:
                        first_leg_consecutive_lidar_failures += 1
                        if first_leg_consecutive_lidar_failures >= MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES:
                            first_leg_status = "lidar_unavailable"
                            break
                    else:
                        first_leg_consecutive_lidar_failures = 0
                        final_distances = distances
                        front_dist = distances.get("front")
                        side_dist = distances.get(chosen_side)

                        if front_dist is not None and front_dist <= FORWARD_COLLISION_INTERRUPT_M:
                            first_leg_status = "collision_risk"
                            break

                        front_clear = front_dist is not None and front_dist >= parallel_front_clearance_f
                        has_signal = front_dist is not None or side_dist is not None
                        parallel_detected = (
                            has_signal
                            and front_clear
                            and (side_dist is not None or first_leg_elapsed_s >= BYPASS_MIN_PARALLEL_TIME_S)
                        )
                        if parallel_detected:
                            first_leg_parallel_confirmations += 1
                        else:
                            first_leg_parallel_confirmations = 0

                        if first_leg_parallel_confirmations >= 2:
                            first_leg_status = "parallel_reached"
                            break

                    await asyncio.sleep(MOTION_COLLISION_POLL_S)
            finally:
                await _set_continuous_motion(None)

            if first_leg_status != "parallel_reached":
                stop_result = await _post_json(STOP_PATH)
                return {
                    **stop_result,
                    "status": first_leg_status,
                    "phase": "first_leg",
                    "chosen_side": chosen_side,
                    "turn_angle_deg": turn_angle_deg,
                    "turn_angle_raw_deg": raw_turn_angle_deg,
                    "required_lateral_shift_m": required_lateral_shift_m,
                    "initial_distances": initial_distances,
                    "final_distances": final_distances,
                    "first_leg_elapsed_s": first_leg_elapsed_s,
                    "first_leg_parallel_confirmations": first_leg_parallel_confirmations,
                }

            await tbot_motion_turn(
                direction=turn_back_direction,
                speed=angular_speed_cmd,
                duration_seconds=turn_duration_s,
            )

            second_leg_start = time.monotonic()
            second_leg_status = "timeout"
            second_leg_consecutive_lidar_failures = 0

            await _set_continuous_motion(linear_cmd, 0.0)
            try:
                while True:
                    second_leg_elapsed_s = time.monotonic() - second_leg_start
                    if second_leg_elapsed_s >= max_second_leg_f:
                        second_leg_status = "timeout"
                        break

                    distances, lidar_read_error = await _read_lidar_distances(lidar)
                    if lidar_read_error is not None:
                        second_leg_consecutive_lidar_failures += 1
                        if second_leg_consecutive_lidar_failures >= MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES:
                            second_leg_status = "lidar_unavailable"
                            break
                    else:
                        second_leg_consecutive_lidar_failures = 0
                        final_distances = distances
                        front_dist = distances.get("front")
                        if front_dist is not None and front_dist <= FORWARD_COLLISION_INTERRUPT_M:
                            second_leg_status = "collision_risk"
                            break
                        if (
                            front_dist is not None
                            and front_dist >= final_front_clearance_f
                            and second_leg_elapsed_s >= BYPASS_MIN_SECOND_LEG_TIME_S
                        ):
                            second_leg_clear_confirmations += 1
                        else:
                            second_leg_clear_confirmations = 0

                        if second_leg_clear_confirmations >= 2:
                            second_leg_status = "clear_path_reached"
                            break

                    await asyncio.sleep(MOTION_COLLISION_POLL_S)
            finally:
                await _set_continuous_motion(None)

            stop_result = await _post_json(STOP_PATH)
            final_status = "completed" if second_leg_status == "clear_path_reached" else second_leg_status
            return {
                **stop_result,
                "status": final_status,
                "chosen_side": chosen_side,
                "turn_angle_deg": turn_angle_deg,
                "turn_angle_raw_deg": raw_turn_angle_deg,
                "required_lateral_shift_m": required_lateral_shift_m,
                "turn_duration_s": turn_duration_s,
                "turn_speed": angular_speed_cmd,
                "forward_speed": linear_cmd,
                "initial_distances": initial_distances,
                "final_distances": final_distances,
                "first_leg_status": first_leg_status,
                "first_leg_elapsed_s": first_leg_elapsed_s,
                "first_leg_parallel_confirmations": first_leg_parallel_confirmations,
                "first_leg_move_posts": max(1, int(math.ceil(first_leg_elapsed_s / MOTION_COMMAND_REFRESH_S))),
                "second_leg_status": second_leg_status,
                "second_leg_elapsed_s": second_leg_elapsed_s,
                "second_leg_clear_confirmations": second_leg_clear_confirmations,
                "second_leg_move_posts": max(1, int(math.ceil(second_leg_elapsed_s / MOTION_COMMAND_REFRESH_S))),
                "interrupt_distance_m": FORWARD_COLLISION_INTERRUPT_M,
                "parallel_front_clearance_m": parallel_front_clearance_f,
                "final_front_clearance_m": final_front_clearance_f,
            }
    finally:
        try:
            await _set_continuous_motion(None)
            await _post_json(STOP_PATH)
        except Exception:
            pass


@mcp_motion_v3.tool()
async def tbot_motion_move_timed(
    linear: float,
    angular: float,
    duration_s: float,
) -> dict[str, Any]:
    """
    Send a linear+angular velocity command for a fixed duration, then stop.
    No LiDAR collision guarding. Intended for pure rotation, lateral correction,
    and short combined ticks (e.g., 0.2 s line-following steps where the caller
    already checks LiDAR before each tick).
    For planned forward travel use tbot_motion_move_forward_distance.
    """
    linear_f = _validate_finite("linear", linear)
    angular_f = _validate_finite("angular", angular)
    duration_f = _validate_finite("duration_s", duration_s)
    if duration_f <= 0:
        raise ValueError("duration_s must be > 0")

    linear_cmd = _clamp(linear_f, MAX_LINEAR)
    angular_cmd = _clamp(angular_f, MAX_ANGULAR)

    preempted_stream = await _set_continuous_motion(None)
    move_result, posts = await _post_move_for_duration(linear_cmd, angular_cmd, duration_f)
    stop_result = await _post_json(STOP_PATH)
    return {
        **stop_result,
        "status": "completed",
        "linear_cmd": linear_cmd,
        "angular_cmd": angular_cmd,
        "duration_s": duration_f,
        "move_posts": posts,
        "command_refresh_s": MOTION_COMMAND_REFRESH_S,
        "move_result": move_result,
        "preempted_continuous_stream": preempted_stream,
    }


@mcp_motion_v3.tool()
async def tbot_motion_move_along_wall(
    direction: str,
    target_distance_m: float,
    speed: float = 0.15,
    timeout_s: float = 30.0,
    stop_distance_m: float = 0.4,
) -> dict[str, Any]:
    """
    Move forward while maintaining a fixed lateral distance from a wall.

    direction: "left" or "right" — which wall to follow.
    target_distance_m: desired lateral distance from the wall in metres.
    speed: forward speed in m/s (clamped to MAX_LINEAR).
    timeout_s: stop and return after this many seconds.
    stop_distance_m: retained for API compatibility; forward collision interrupt
                     uses a fixed 10 cm threshold.

    Returns {"status": "obstacle_reached"|"timeout", "distance_traveled_ticks": int}.
    """
    direction_clean = direction.strip().lower() if isinstance(direction, str) else ""
    if direction_clean not in ("left", "right"):
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

    target_dist = _validate_finite("target_distance_m", target_distance_m)
    speed_f = _validate_finite("speed", speed)
    timeout_f = _validate_finite("timeout_s", timeout_s)
    stop_dist = _validate_finite("stop_distance_m", stop_distance_m)

    if target_dist <= 0:
        raise ValueError("target_distance_m must be > 0")
    if timeout_f <= 0:
        raise ValueError("timeout_s must be > 0")
    if stop_dist <= 0:
        raise ValueError("stop_distance_m must be > 0")

    await _set_continuous_motion(None)
    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    wall_sign = 1.0 if direction_clean == "left" else -1.0
    start_mono = time.monotonic()
    ticks = 0
    move_posts = 0
    interrupt_distance_m = FORWARD_COLLISION_INTERRUPT_M

    async with Client(LIDAR_MCP_URL_V3) as lidar:
        while True:
            if time.monotonic() - start_mono >= timeout_f:
                await _post_json(STOP_PATH)
                return {
                    "status": "timeout",
                    "distance_traveled_ticks": ticks,
                    "move_posts": move_posts,
                    "command_refresh_s": MOTION_COMMAND_REFRESH_S,
                    "requested_stop_distance_m": stop_dist,
                    "interrupt_distance_m": interrupt_distance_m,
                }

            try:
                raw = await lidar.call_tool("tbot_lidar_get_obstacle_distances", {"sector": "all"})
                distances = _extract_tool_dict(raw).get("distances", {})
            except Exception:
                distances = {}

            front_dist = distances.get("front")
            lateral_dist = distances.get(direction_clean)

            if front_dist is not None and front_dist <= interrupt_distance_m:
                await _post_json(STOP_PATH)
                return {
                    "status": "obstacle_reached",
                    "distance_traveled_ticks": ticks,
                    "move_posts": move_posts,
                    "command_refresh_s": MOTION_COMMAND_REFRESH_S,
                    "requested_stop_distance_m": stop_dist,
                    "interrupt_distance_m": interrupt_distance_m,
                }

            if lateral_dist is not None:
                error = lateral_dist - target_dist
                angular_correction = _clamp(wall_sign * WALL_FOLLOW_KP * error, WALL_FOLLOW_MAX_ANGULAR)
            else:
                angular_correction = 0.0

            await _post_json(MOVE_PATH, {"linear": clamped_speed, "angular": angular_correction})
            ticks += 1
            move_posts += 1
            await asyncio.sleep(MOTION_COMMAND_REFRESH_S)


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

    # ROS convention: angular.z > 0 = CCW = left, angular.z < 0 = CW = right.
    # ANGULAR_SIGN flips the mapping when the robot's frame differs (default -1.0).
    # "right" → positive input frame → +1 * ANGULAR_SIGN
    # "left"  → negative input frame → -1 * ANGULAR_SIGN
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


@mcp_motion_v3.tool()
async def tbot_motion_get_robot_status() -> dict[str, Any]:
    """Get the current robot motion status — whether it is moving, and current linear/angular values."""
    health, health_error = await _try_get_health()
    if health is None:
        return {
            "moving": None,
            "linear": None,
            "angular": None,
            "base_url": BASE_URL,
            "error": health_error,
        }

    linear_raw = health.get("linear", 0.0)
    angular_raw = health.get("angular", 0.0)
    try:
        linear_f = float(linear_raw) if linear_raw is not None else 0.0
        angular_f = float(angular_raw) if angular_raw is not None else 0.0
        moving = abs(linear_f) > 1e-6 or abs(angular_f) > 1e-6
    except (TypeError, ValueError):
        linear_f = 0.0
        angular_f = 0.0
        moving = False

    return {
        "moving": moving,
        "linear": linear_f,
        "angular": angular_f,
        "base_url": BASE_URL,
    }


@mcp_motion_v3.tool()
async def tbot_motion_approach_until_close(
    target_distance_m: float,
    stop_distance_m: float,
    speed: float,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """
    INTERNAL: Called by tbot_vision_search_and_approach_object.
    Do not call this directly — use tbot_vision_search_and_approach_object instead.

    Approach an object using precomputed LiDAR distance before moving.

    Flow:
    1) Read current front LiDAR distance.
    2) Compute required move distance: current_front_distance - target_distance_m.
    3) Convert to duration using speed and execute one forward-distance move.
    4) Re-check front LiDAR distance and return final status.

    stop_distance_m is retained for API compatibility. Collision interrupt uses
    a fixed 10 cm threshold to avoid over-conservative stops.

    Returns status: "reached" | "completed" | "collision_risk" | "timeout" | "error".
    """
    target_dist = _validate_finite("target_distance_m", target_distance_m)
    stop_dist = _validate_finite("stop_distance_m", stop_distance_m)
    speed_f = _validate_finite("speed", speed)
    timeout_f = _validate_finite("timeout_s", timeout_s)

    if target_dist <= 0:
        raise ValueError("target_distance_m must be > 0")
    if stop_dist <= 0:
        raise ValueError("stop_distance_m must be > 0")
    if timeout_f <= 0:
        raise ValueError("timeout_s must be > 0")

    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    if clamped_speed == 0.0:
        clamped_speed = 0.05

    await _set_continuous_motion(None)

    start_mono = time.monotonic()
    initial_front_distance_m: float | None = None
    final_distance_m: float | None = None
    required_move_distance_m = 0.0
    requested_move_duration_s = 0.0
    executed_move_duration_s = 0.0
    move_posts = 0
    collision_checks = 0
    consecutive_lidar_failures = 0
    final_status = "error"
    interrupt_distance_m = FORWARD_COLLISION_INTERRUPT_M

    def _extract_front_distance(collision_data: dict[str, Any]) -> float | None:
        min_fwd = collision_data.get("min_forward_distance_m")
        if isinstance(min_fwd, (int, float)) and math.isfinite(float(min_fwd)):
            return float(min_fwd)
        distances = collision_data.get("distances")
        if isinstance(distances, dict):
            front_raw = distances.get("front")
            if isinstance(front_raw, (int, float)) and math.isfinite(float(front_raw)):
                return float(front_raw)
        return None

    try:
        async with Client(LIDAR_MCP_URL_V3) as lidar:
            try:
                initial_raw = await lidar.call_tool(
                    "tbot_lidar_check_collision",
                    {"front_threshold_m": interrupt_distance_m},
                )
                collision_checks += 1
                initial_check = _extract_tool_dict(initial_raw)
            except Exception as e:
                return {
                    "status": "error",
                    "front_distance": None,
                    "initial_front_distance_m": None,
                    "required_move_distance_m": 0.0,
                    "requested_move_duration_s": 0.0,
                    "executed_move_duration_s": 0.0,
                    "requested_stop_distance_m": stop_dist,
                    "interrupt_distance_m": interrupt_distance_m,
                    "move_posts": 0,
                    "collision_checks": collision_checks,
                    "consecutive_lidar_failures": 1,
                    "command_refresh_s": MOTION_COMMAND_REFRESH_S,
                    "error": f"Initial LiDAR check failed: {e}",
                }

            initial_risk = str(initial_check.get("risk_level") or "clear")
            initial_front_distance_m = _extract_front_distance(initial_check)
            final_distance_m = initial_front_distance_m

            if initial_front_distance_m is None:
                return {
                    "status": "error",
                    "front_distance": None,
                    "initial_front_distance_m": None,
                    "required_move_distance_m": 0.0,
                    "requested_move_duration_s": 0.0,
                    "executed_move_duration_s": 0.0,
                    "requested_stop_distance_m": stop_dist,
                    "interrupt_distance_m": interrupt_distance_m,
                    "move_posts": 0,
                    "collision_checks": collision_checks,
                    "consecutive_lidar_failures": consecutive_lidar_failures,
                    "command_refresh_s": MOTION_COMMAND_REFRESH_S,
                    "error": "LiDAR front distance unavailable during initial check",
                }

            if initial_risk == "stop" or initial_front_distance_m <= interrupt_distance_m:
                final_status = "collision_risk"
            elif initial_front_distance_m <= target_dist:
                final_status = "reached"
            else:
                required_move_distance_m = max(0.0, initial_front_distance_m - target_dist)
                requested_move_duration_s = required_move_distance_m / clamped_speed
                elapsed_s = max(0.0, time.monotonic() - start_mono)
                remaining_timeout_s = max(0.0, timeout_f - elapsed_s)
                executed_move_duration_s = min(requested_move_duration_s, remaining_timeout_s)

                if executed_move_duration_s <= 0:
                    final_status = "timeout"
                else:
                    # Closed-loop approach: poll LiDAR every tick and stop early
                    # when the target distance is reached, preventing overshoot.
                    approach_start = time.monotonic()
                    loop_exit = "duration_elapsed"

                    while True:
                        elapsed_loop = time.monotonic() - approach_start
                        if elapsed_loop >= executed_move_duration_s:
                            break

                        try:
                            poll_raw = await lidar.call_tool(
                                "tbot_lidar_check_collision",
                                {"front_threshold_m": interrupt_distance_m},
                            )
                            collision_checks += 1
                            poll_check = _extract_tool_dict(poll_raw)
                            poll_front = _extract_front_distance(poll_check)
                            if poll_front is not None:
                                final_distance_m = poll_front
                            consecutive_lidar_failures = 0

                            if poll_front is not None and poll_front <= interrupt_distance_m:
                                loop_exit = "collision_risk"
                                break
                            if poll_front is not None and poll_front <= target_dist:
                                loop_exit = "reached"
                                break
                        except Exception:
                            consecutive_lidar_failures += 1
                            if consecutive_lidar_failures >= MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES:
                                loop_exit = "lidar_unavailable"
                                break

                        remaining_loop = executed_move_duration_s - (time.monotonic() - approach_start)
                        if remaining_loop <= 0:
                            break
                        await _post_json(MOVE_PATH, {"linear": clamped_speed, "angular": 0.0})
                        move_posts += 1
                        await asyncio.sleep(min(MOTION_COMMAND_REFRESH_S, remaining_loop))

                    if loop_exit == "collision_risk":
                        final_status = "collision_risk"
                    elif loop_exit == "reached":
                        final_status = "reached"
                    else:
                        final_status = "timeout" if executed_move_duration_s + 1e-6 < requested_move_duration_s else "completed"
    except Exception as e:
        try:
            await _set_continuous_motion(None)
            await _post_json(STOP_PATH)
        except Exception:
            pass
        return {
            "status": "error",
            "front_distance": final_distance_m,
            "initial_front_distance_m": initial_front_distance_m,
            "required_move_distance_m": required_move_distance_m,
            "requested_move_duration_s": requested_move_duration_s,
            "executed_move_duration_s": executed_move_duration_s,
            "requested_stop_distance_m": stop_dist,
            "interrupt_distance_m": interrupt_distance_m,
            "move_posts": move_posts,
            "collision_checks": collision_checks,
            "consecutive_lidar_failures": consecutive_lidar_failures,
            "command_refresh_s": MOTION_COMMAND_REFRESH_S,
            "error": str(e),
        }
    finally:
        try:
            await _set_continuous_motion(None)
        except Exception:
            pass

    stop_result = await _post_json(STOP_PATH)
    if move_posts == 0 and executed_move_duration_s > 0:
        move_posts = max(1, int(math.ceil(executed_move_duration_s / MOTION_COMMAND_REFRESH_S)))

    return {
        **stop_result,
        "status": final_status,
        "front_distance": final_distance_m,
        "initial_front_distance_m": initial_front_distance_m,
        "required_move_distance_m": required_move_distance_m,
        "requested_move_duration_s": requested_move_duration_s,
        "executed_move_duration_s": executed_move_duration_s,
        "speed": clamped_speed,
        "requested_stop_distance_m": stop_dist,
        "interrupt_distance_m": interrupt_distance_m,
        "move_posts": move_posts,
        "collision_checks": collision_checks,
        "consecutive_lidar_failures": consecutive_lidar_failures,
        "command_refresh_s": MOTION_COMMAND_REFRESH_S,
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
