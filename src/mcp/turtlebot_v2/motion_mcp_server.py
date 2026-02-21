"""
TurtleBot Motion MCP Server V2.

Controls TurtleBot motion via the HTTP API exposed by motion_server_tbot.py.
"""

import asyncio
import logging
import math
import os
import threading
from typing import Any, Optional

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
HTTP_TIMEOUT_S = _env_float("MOTION_TIMEOUT_S", 2.0)
MAX_LINEAR = abs(_env_float("MOTION_MAX_LINEAR", 0.2))
MAX_ANGULAR = abs(_env_float("MOTION_MAX_ANGULAR", 1.0))

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
    angular_cmd = _clamp(angular_f, MAX_ANGULAR)

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
        "was_clamped": (linear_cmd != linear_f) or (angular_cmd != angular_f),
        "limits": {"max_linear": MAX_LINEAR, "max_angular": MAX_ANGULAR},
        "base_url": BASE_URL,
        "endpoint": MOVE_PATH,
        "verification": verification,
        "physical_motion_confirmed": False,
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

