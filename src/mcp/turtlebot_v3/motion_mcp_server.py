"""
TurtleBot Motion MCP Server V3.

Duration-based motion commands via the HTTP API exposed by the motion server.
"""

import asyncio
import json
import logging
import math
import os
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

MOVE_PATH = "/move"
STOP_PATH = "/stop"
HEALTH_PATH = "/health"

mcp_motion_v3 = FastMCP("TurtleBot Motion MCP Server V3")


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
    result = await _post_json(STOP_PATH)
    return {**result, "base_url": BASE_URL, "endpoint": STOP_PATH}


@mcp_motion_v3.tool()
async def tbot_motion_move_forward(
    speed: float,
    duration_seconds: float,
) -> dict[str, Any]:
    """Move the robot forward at the given speed for duration_seconds, then stop automatically."""
    speed_f = _validate_finite("speed", speed)
    duration_f = _validate_finite("duration_seconds", duration_seconds)
    if duration_f <= 0:
        raise ValueError("duration_seconds must be > 0")

    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    await _post_json(MOVE_PATH, {"linear": clamped_speed, "angular": 0.0})
    await asyncio.sleep(duration_f)
    stop_result = await _post_json(STOP_PATH)
    return {
        **stop_result,
        "status": "completed",
        "speed": clamped_speed,
        "duration_seconds": duration_f,
        "was_clamped": abs(speed_f) != clamped_speed,
    }


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

    result = await _post_json(MOVE_PATH, {"linear": linear_cmd, "angular": angular_cmd})

    if duration_s is not None:
        duration_f = _validate_finite("duration_s", duration_s)
        if duration_f > 0:
            await asyncio.sleep(duration_f)
            stop_result = await _post_json(STOP_PATH)
            return {
                **stop_result,
                "status": "completed",
                "linear_cmd": linear_cmd,
                "angular_cmd": angular_cmd,
                "duration_s": duration_f,
            }

    return {
        **result,
        "linear_cmd": linear_cmd,
        "angular_cmd": angular_cmd,
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

    move_result = await _post_json(MOVE_PATH, {"linear": 0.0, "angular": angular_cmd})

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

    await asyncio.sleep(duration_f)
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
    Move forward until the front obstacle is within target_distance_m.

    stop_distance_m is passed as front_threshold_m to tbot_lidar_check_collision.
    If risk_level is "stop", halts and returns collision_risk.
    If risk_level is "caution", stops, nudges ~10 degrees, then resumes.
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

    # Nudge parameters: ~10 degrees at a low angular speed
    nudge_speed = min(0.5, MAX_ANGULAR)
    nudge_angular_cmd = nudge_speed * ANGULAR_SIGN
    nudge_duration_s = (10.0 * math.pi / 180.0) / max(0.01, nudge_speed)

    start_mono = time.monotonic()
    final_distance_m: float | None = None

    try:
        await _post_json(MOVE_PATH, {"linear": clamped_speed, "angular": 0.0})

        async with Client(LIDAR_MCP_URL_V3) as lidar:
            while True:
                elapsed = time.monotonic() - start_mono
                if elapsed >= timeout_f:
                    await _post_json(STOP_PATH)
                    return {"status": "timeout", "front_distance": final_distance_m}

                # Check collision risk
                try:
                    collision_result = await lidar.call_tool(
                        "tbot_lidar_check_collision",
                        {"front_threshold_m": stop_dist},
                    )
                    collision_data = _extract_tool_dict(collision_result)
                    risk_level = collision_data.get("risk_level", "clear")
                    distances = collision_data.get("distances", {})
                    if isinstance(distances, dict):
                        front_raw = distances.get("front")
                        if front_raw is not None:
                            try:
                                final_distance_m = float(front_raw)
                            except (TypeError, ValueError):
                                pass

                    if risk_level == "stop":
                        await _post_json(STOP_PATH)
                        return {"status": "collision_risk", "front_distance": final_distance_m}

                    if risk_level == "caution":
                        await _post_json(STOP_PATH)
                        # Nudge: small rotation to attempt repositioning
                        await _post_json(MOVE_PATH, {"linear": 0.0, "angular": nudge_angular_cmd})
                        await asyncio.sleep(nudge_duration_s)
                        await _post_json(STOP_PATH)
                        # Resume forward motion
                        await _post_json(MOVE_PATH, {"linear": clamped_speed, "angular": 0.0})
                except Exception:
                    pass

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
                            if front_dist <= target_dist:
                                await _post_json(STOP_PATH)
                                return {"status": "reached", "front_distance": final_distance_m}
                        except (TypeError, ValueError):
                            pass
                except Exception:
                    pass

                await asyncio.sleep(0.2)

    except Exception:
        try:
            await _post_json(STOP_PATH)
        except Exception:
            pass
        raise


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
