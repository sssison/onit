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
MOTION_STOP_BUFFER_M = max(0.0, _env_float("MOTION_STOP_BUFFER_M", 0.03))
MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES = max(
    1,
    int(_env_float("MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES", 3.0)),
)

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
async def tbot_motion_move_forward(
    speed: float,
    duration_seconds: float,
    stop_distance_m: float = 0.1,
) -> dict[str, Any]:
    """
    Move forward smoothly for duration_seconds with LiDAR stop guard.

    Motion is streamed continuously in the background so forward movement does
    not become pulse-like while waiting for LiDAR checks.
    """
    speed_f = _validate_finite("speed", speed)
    duration_f = _validate_finite("duration_seconds", duration_seconds)
    stop_distance_f = _validate_finite("stop_distance_m", stop_distance_m)
    if duration_f <= 0:
        raise ValueError("duration_seconds must be > 0")
    if stop_distance_f <= 0:
        raise ValueError("stop_distance_m must be > 0")

    await _set_continuous_motion(None)
    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    start_mono = time.monotonic()
    collision_checks = 0
    consecutive_lidar_failures = 0
    last_front_dist: float | None = None
    last_dynamic_stop_trigger_m = stop_distance_f + MOTION_STOP_BUFFER_M
    final_status = "completed"

    async with Client(LIDAR_MCP_URL_V3) as lidar:
        await _set_continuous_motion(clamped_speed, 0.0)
        try:
            while True:
                elapsed = time.monotonic() - start_mono
                if elapsed >= duration_f:
                    break

                poll_started = time.monotonic()
                try:
                    raw = await lidar.call_tool(
                        "tbot_lidar_check_collision",
                        {"front_threshold_m": stop_distance_f},
                    )
                    collision = _extract_tool_dict(raw)
                    collision_checks += 1
                except Exception:
                    collision = {}
                    consecutive_lidar_failures += 1
                    if consecutive_lidar_failures >= MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES:
                        final_status = "lidar_unavailable"
                        break

                distances = collision.get("distances")
                if isinstance(distances, dict):
                    front_raw = distances.get("front")
                    if isinstance(front_raw, (int, float)):
                        last_front_dist = float(front_raw)
                        consecutive_lidar_failures = 0
                    else:
                        consecutive_lidar_failures += 1
                        if consecutive_lidar_failures >= MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES:
                            final_status = "lidar_unavailable"
                            break
                else:
                    consecutive_lidar_failures += 1
                    if consecutive_lidar_failures >= MOTION_LIDAR_MAX_CONSECUTIVE_FAILURES:
                        final_status = "lidar_unavailable"
                        break

                poll_elapsed_s = max(0.0, time.monotonic() - poll_started)
                dynamic_margin_m = (
                    MOTION_STOP_BUFFER_M
                    + (clamped_speed * (poll_elapsed_s + MOTION_COMMAND_REFRESH_S))
                )
                last_dynamic_stop_trigger_m = stop_distance_f + dynamic_margin_m

                if (
                    last_front_dist is not None
                    and last_front_dist <= last_dynamic_stop_trigger_m
                ):
                    final_status = "collision_risk"
                    break
                if collision.get("risk_level") == "stop":
                    final_status = "collision_risk"
                    break

                remaining = duration_f - (time.monotonic() - start_mono)
                if remaining <= 0:
                    break
                await asyncio.sleep(min(MOTION_COLLISION_POLL_S, remaining))
        finally:
            await _set_continuous_motion(None)

    stop_result = await _post_json(STOP_PATH)
    elapsed_total = min(duration_f, max(0.0, time.monotonic() - start_mono))
    estimated_move_posts = max(1, int(math.ceil(elapsed_total / MOTION_COMMAND_REFRESH_S)))

    if final_status == "collision_risk":
        return {
            **stop_result,
            "status": "collision_risk",
            "front_distance": last_front_dist,
            "stop_distance_m": stop_distance_f,
            "move_posts": estimated_move_posts,
            "collision_checks": collision_checks,
            "consecutive_lidar_failures": consecutive_lidar_failures,
            "command_refresh_s": MOTION_COMMAND_REFRESH_S,
            "collision_poll_s": MOTION_COLLISION_POLL_S,
            "dynamic_stop_trigger_m": last_dynamic_stop_trigger_m,
        }

    if final_status == "lidar_unavailable":
        return {
            **stop_result,
            "status": "lidar_unavailable",
            "front_distance": last_front_dist,
            "stop_distance_m": stop_distance_f,
            "move_posts": estimated_move_posts,
            "collision_checks": collision_checks,
            "consecutive_lidar_failures": consecutive_lidar_failures,
            "command_refresh_s": MOTION_COMMAND_REFRESH_S,
            "collision_poll_s": MOTION_COLLISION_POLL_S,
            "dynamic_stop_trigger_m": last_dynamic_stop_trigger_m,
        }

    return {
        **stop_result,
        "status": "completed",
        "speed": clamped_speed,
        "duration_seconds": duration_f,
        "stop_distance_m": stop_distance_f,
        "was_clamped": abs(speed_f) != clamped_speed,
        "move_posts": estimated_move_posts,
        "collision_checks": collision_checks,
        "consecutive_lidar_failures": consecutive_lidar_failures,
        "command_refresh_s": MOTION_COMMAND_REFRESH_S,
        "collision_poll_s": MOTION_COLLISION_POLL_S,
        "dynamic_stop_trigger_m": last_dynamic_stop_trigger_m,
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
    stop_distance_m: stop when a front obstacle is closer than this distance.

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

    async with Client(LIDAR_MCP_URL_V3) as lidar:
        while True:
            if time.monotonic() - start_mono >= timeout_f:
                await _post_json(STOP_PATH)
                return {
                    "status": "timeout",
                    "distance_traveled_ticks": ticks,
                    "move_posts": move_posts,
                    "command_refresh_s": MOTION_COMMAND_REFRESH_S,
                }

            try:
                raw = await lidar.call_tool("tbot_lidar_get_obstacle_distances", {"sector": "all"})
                distances = _extract_tool_dict(raw).get("distances", {})
            except Exception:
                distances = {}

            front_dist = distances.get("front")
            lateral_dist = distances.get(direction_clean)

            if front_dist is not None and front_dist <= stop_dist:
                await _post_json(STOP_PATH)
                return {
                    "status": "obstacle_reached",
                    "distance_traveled_ticks": ticks,
                    "move_posts": move_posts,
                    "command_refresh_s": MOTION_COMMAND_REFRESH_S,
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
async def tbot_motion_move(
    linear: float,
    angular: float,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """
    Send a direct linear + angular velocity command to the robot.

    linear:   forward speed m/s, clamped to ±MOTION_MAX_LINEAR (default 0.2).
    angular:  angular speed rad/s sent directly; positive = left/CCW, negative = right/CW.
              Clamped to ±MOTION_MAX_ANGULAR (default 1.0) but sign is NOT remapped — use
              positive to turn left, negative to turn right (standard ROS2 convention).
    duration_s: if provided and > 0, automatically stops after this many seconds.
                If omitted, the command runs until tbot_motion_stop is called.
    """
    linear_f = _validate_finite("linear", linear)
    angular_f = _validate_finite("angular", angular)

    linear_cmd = _clamp(linear_f, MAX_LINEAR)
    angular_cmd = _clamp(angular_f, MAX_ANGULAR)

    if duration_s is not None:
        duration_f = _validate_finite("duration_s", duration_s)
        if duration_f > 0:
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

    result = await _post_json(MOVE_PATH, {"linear": linear_cmd, "angular": angular_cmd})
    replaced_stream = await _set_continuous_motion(linear_cmd, angular_cmd)
    return {
        **result,
        "status": "streaming",
        "linear_cmd": linear_cmd,
        "angular_cmd": angular_cmd,
        "command_refresh_s": MOTION_COMMAND_REFRESH_S,
        "preempted_continuous_stream": replaced_stream,
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
    Move forward continuously until LiDAR reports target distance reached.

    This uses continuous cmd_vel streaming (not discrete forward re-posts), and
    LiDAR is used only to decide when to stop.

    stop_distance_m is passed as front_threshold_m to tbot_lidar_check_collision.
    If risk_level is "stop", halts and returns collision_risk.
    Returns status: "reached" | "collision_risk" | "timeout".
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
    final_distance_m: float | None = None
    collision_checks = 0
    consecutive_lidar_failures = 0
    final_status = "timeout"

    try:
        async with Client(LIDAR_MCP_URL_V3) as lidar:
            await _set_continuous_motion(clamped_speed, 0.0)
            while True:
                elapsed = time.monotonic() - start_mono
                if elapsed >= timeout_f:
                    final_status = "timeout"
                    break

                # Check collision risk
                try:
                    collision_result = await lidar.call_tool(
                        "tbot_lidar_check_collision",
                        {"front_threshold_m": stop_dist},
                    )
                    collision_data = _extract_tool_dict(collision_result)
                    collision_checks += 1
                    risk_level = collision_data.get("risk_level", "clear")
                    distances = collision_data.get("distances", {})
                    if isinstance(distances, dict):
                        front_raw = distances.get("front")
                        if front_raw is not None:
                            try:
                                final_distance_m = float(front_raw)
                                consecutive_lidar_failures = 0
                            except (TypeError, ValueError):
                                pass

                    if risk_level == "stop":
                        final_status = "collision_risk"
                        break
                except Exception:
                    consecutive_lidar_failures += 1

                # Check target distance
                try:
                    dist_result = await lidar.call_tool(
                        "tbot_lidar_get_obstacle_distances",
                        {"sector": "front"},
                    )
                    dist_data = _extract_tool_dict(dist_result)
                    dist_raw = dist_data.get("distance_m")
                    if dist_raw is not None:
                        try:
                            front_dist = float(dist_raw)
                            final_distance_m = front_dist
                            consecutive_lidar_failures = 0
                            if front_dist <= target_dist:
                                final_status = "reached"
                                break
                        except (TypeError, ValueError):
                            pass
                except Exception:
                    consecutive_lidar_failures += 1

                await asyncio.sleep(MOTION_COMMAND_REFRESH_S)
    except Exception as e:
        try:
            await _set_continuous_motion(None)
            await _post_json(STOP_PATH)
        except Exception:
            pass
        return {
            "status": "error",
            "front_distance": final_distance_m,
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
    elapsed_total = min(timeout_f, max(0.0, time.monotonic() - start_mono))
    estimated_move_posts = max(1, int(math.ceil(elapsed_total / MOTION_COMMAND_REFRESH_S)))

    return {
        **stop_result,
        "status": final_status,
        "front_distance": final_distance_m,
        "speed": clamped_speed,
        "move_posts": estimated_move_posts,
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
