"""
TurtleBot Motion MCP Server V2.

Controls TurtleBot motion via the HTTP API exposed by motion_server_tbot.py.
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
from typing import Any, Optional

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
HTTP_TIMEOUT_S = _env_float("MOTION_TIMEOUT_S", 2.0)
MAX_LINEAR = abs(_env_float("MOTION_MAX_LINEAR", 0.2))
MAX_ANGULAR = abs(_env_float("MOTION_MAX_ANGULAR", 1.0))
ANGULAR_SIGN = _env_float("MOTION_ANGULAR_SIGN", -1.0)

LIDAR_MCP_URL = os.getenv("TBOT_LIDAR_MCP_URL", "http://127.0.0.1:18208/turtlebot-lidar-v2")

MOVE_PATH = "/move"
STOP_PATH = "/stop"
HEALTH_PATH = "/health"

mcp_motion_v2 = FastMCP("TurtleBot Motion MCP Server V2")

_motion_command_lock = threading.Lock()
_motion_command_version = 0


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


def _reserve_motion_command() -> int:
    global _motion_command_version
    with _motion_command_lock:
        _motion_command_version += 1
        return _motion_command_version


def _is_latest_motion_command(version: int) -> bool:
    with _motion_command_lock:
        return version == _motion_command_version


def _invalidate_motion_commands() -> None:
    global _motion_command_version
    with _motion_command_lock:
        _motion_command_version += 1


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


async def _probe_motion_api() -> dict[str, Any]:
    url = f"{BASE_URL}{MOVE_PATH}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.options(url)
    except httpx.RequestError as e:
        return {
            "status": "offline",
            "base_url": BASE_URL,
            "reachable": False,
            "error": str(e),
            "endpoints": {"move": MOVE_PATH, "stop": STOP_PATH, "health": HEALTH_PATH},
        }

    return {
        "status": "online" if response.status_code != 404 else "offline",
        "base_url": BASE_URL,
        "reachable": response.status_code != 404,
        "status_code": response.status_code,
        "endpoints": {"move": MOVE_PATH, "stop": STOP_PATH, "health": HEALTH_PATH},
    }


@mcp_motion_v2.tool()
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

    probe = await _probe_motion_api()
    return {**probe, "health_error": health_error}


@mcp_motion_v2.tool()
async def tbot_motion_stop() -> dict[str, Any]:
    """Stop the robot by setting linear and angular targets to zero."""
    _invalidate_motion_commands()
    result = await _post_json(STOP_PATH)
    health, health_error = await _try_get_health()

    verification: dict[str, Any] = {
        "health_supported": health_error is None,
        "health_error": health_error,
        "command_matches_health": None,
    }
    if health is not None:
        health_linear = health.get("linear")
        health_angular = health.get("angular")
        verification["health"] = health
        verification["command_matches_health"] = (
            isinstance(health_linear, (int, float))
            and isinstance(health_angular, (int, float))
            and abs(float(health_linear)) <= 1e-6
            and abs(float(health_angular)) <= 1e-6
        )

    return {
        **result,
        "base_url": BASE_URL,
        "endpoint": STOP_PATH,
        "verification": verification,
        "physical_motion_confirmed": False,
    }


@mcp_motion_v2.tool()
async def tbot_motion_move(
    linear: float,
    angular: float,
    duration_s: Optional[float] = None,
) -> dict[str, Any]:
    """
    Command robot motion via the motion server.

    Inputs are clamped to +/- MOTION_MAX_LINEAR and +/- MOTION_MAX_ANGULAR for safety.
    If duration_s is provided and > 0, an automatic stop is issued after the interval.
    """
    linear_f = _validate_finite("linear", linear)
    angular_f = _validate_finite("angular", angular)

    linear_cmd = _clamp(linear_f, MAX_LINEAR)
    angular_cmd_unmapped = _clamp(angular_f, MAX_ANGULAR)
    angular_cmd = angular_cmd_unmapped * ANGULAR_SIGN

    command_version = _reserve_motion_command()
    result = await _post_json(MOVE_PATH, {"linear": linear_cmd, "angular": angular_cmd})

    if duration_s is not None:
        duration_f = _validate_finite("duration_s", duration_s)
        if duration_f > 0:
            await asyncio.sleep(duration_f)
            if _is_latest_motion_command(command_version):
                await _post_json(STOP_PATH)
                result = {**result, "auto_stopped": True, "duration_s": duration_f}
            else:
                result = {
                    **result,
                    "auto_stopped": False,
                    "duration_s": duration_f,
                    "auto_stop_skipped_stale": True,
                }

    health, health_error = await _try_get_health()
    verification: dict[str, Any] = {
        "health_supported": health_error is None,
        "health_error": health_error,
        "command_matches_health": None,
    }
    if health is not None:
        health_linear = health.get("linear")
        health_angular = health.get("angular")
        verification["health"] = health
        verification["command_matches_health"] = (
            isinstance(health_linear, (int, float))
            and isinstance(health_angular, (int, float))
            and abs(float(health_linear) - linear_cmd) <= 1e-6
            and abs(float(health_angular) - angular_cmd) <= 1e-6
        )

    return {
        **result,
        "requested": {"linear": linear_f, "angular": angular_f},
        "commanded": {"linear": linear_cmd, "angular": angular_cmd},
        "commanded_input_frame": {"linear": linear_cmd, "angular": angular_cmd_unmapped},
        "angular_sign": ANGULAR_SIGN,
        "was_clamped": (linear_cmd != linear_f) or (angular_cmd_unmapped != angular_f),
        "limits": {"max_linear": MAX_LINEAR, "max_angular": MAX_ANGULAR},
        "base_url": BASE_URL,
        "endpoint": MOVE_PATH,
        "verification": verification,
        "physical_motion_confirmed": False,
    }


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


@mcp_motion_v2.tool()
async def tbot_motion_scan_rotate(
    degrees: float,
    speed: float = 0.5,
) -> dict[str, Any]:
    """Rotate the robot by a fixed angle in degrees. Positive=right/CW, negative=left/CCW. Blocks until rotation is complete or preempted."""
    degrees_f = _validate_finite("degrees", degrees)
    speed_f = _validate_finite("speed", speed)

    clamped_speed = abs(_clamp(abs(speed_f), MAX_ANGULAR))
    if clamped_speed == 0.0 or degrees_f == 0.0:
        return {"status": "completed", "degrees": degrees_f, "duration_s": 0.0, "angular_cmd": 0.0}

    duration_s = abs(degrees_f * math.pi / 180.0) / clamped_speed
    angular_cmd = math.copysign(clamped_speed, degrees_f) * ANGULAR_SIGN

    command_version = _reserve_motion_command()
    await _post_json(MOVE_PATH, {"linear": 0.0, "angular": angular_cmd})
    await asyncio.sleep(duration_s)

    if _is_latest_motion_command(command_version):
        await _post_json(STOP_PATH)
        return {"status": "completed", "degrees": degrees_f, "duration_s": duration_s, "angular_cmd": angular_cmd}

    return {"status": "preempted", "degrees": degrees_f, "duration_s": duration_s, "angular_cmd": angular_cmd}


@mcp_motion_v2.tool()
async def tbot_motion_forward_until_close(
    target_distance_m: float,
    speed: float = 0.1,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Move forward until the nearest obstacle in front is within target_distance_m, or until timeout_s is reached."""
    target_dist = _validate_finite("target_distance_m", target_distance_m)
    speed_f = _validate_finite("speed", speed)
    timeout_f = _validate_finite("timeout_s", timeout_s)

    if target_dist <= 0:
        raise ValueError("target_distance_m must be > 0")
    if timeout_f <= 0:
        raise ValueError("timeout_s must be > 0")

    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    if clamped_speed == 0.0:
        clamped_speed = 0.05

    command_version = _reserve_motion_command()
    await _post_json(MOVE_PATH, {"linear": clamped_speed, "angular": 0.0})

    start_mono = time.monotonic()
    stop_reason = "timeout"
    final_distance_m: float | None = None

    try:
        async with Client(LIDAR_MCP_URL) as lidar:
            while True:
                elapsed = time.monotonic() - start_mono
                if elapsed >= timeout_f:
                    stop_reason = "timeout"
                    break
                if not _is_latest_motion_command(command_version):
                    stop_reason = "preempted"
                    break
                try:
                    result = await lidar.call_tool("tbot_lidar_nearest_obstacle", {"sector": "front"})
                    data = _extract_tool_dict(result)
                    dist = data.get("distance_m")
                    if dist is not None:
                        final_distance_m = float(dist)
                        if final_distance_m <= target_dist:
                            stop_reason = "obstacle_reached"
                            break
                except Exception:
                    pass
                await asyncio.sleep(0.2)
    finally:
        if _is_latest_motion_command(command_version):
            try:
                await _post_json(STOP_PATH)
            except Exception:
                pass

    elapsed_s = time.monotonic() - start_mono
    return {
        "status": stop_reason,
        "elapsed_s": elapsed_s,
        "final_distance_m": final_distance_m,
    }


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18205,
    path: str = "/turtlebot-motion-v2",
    options: dict = {},
) -> None:
    """Run the TurtleBot Motion MCP Server V2."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting TurtleBot Motion MCP V2 base_url=%s timeout_s=%.2f at %s:%s%s",
        BASE_URL,
        HTTP_TIMEOUT_S,
        host,
        port,
        path,
    )
    mcp_motion_v2.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
