"""
TurtleBot Motion MCP Server V3.

Duration-based motion commands via the HTTP API exposed by the motion server.
"""

import asyncio
import logging
import math
import os
import threading
import time
from typing import Any

import httpx
from fastmcp import FastMCP

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

    # Keep turn(direction=...) aligned with signed angular convention used by
    # tbot_motion_move_timed on this robot (+angular = left/CCW, -angular = right/CW).
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
