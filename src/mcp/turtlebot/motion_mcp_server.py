"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


mcp_motion = FastMCP("Motion MCP")
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
    logger.debug("POST %s payload_keys=%s", url, sorted(payload.keys()) if payload else [])

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                raise RuntimeError(f"Unexpected response from {url}: expected JSON object")

            logger.debug("POST %s -> status=%s keys=%s", url, response.status_code, sorted(data.keys()))
            return data
    except httpx.HTTPStatusError as e:
        body = e.response.text
        logger.error("POST %s failed status=%s body=%s", url, e.response.status_code, body[:500])
        raise RuntimeError(f"Motion server error {e.response.status_code} for {url}: {body[:4000]}") from e
    except httpx.RequestError as e:
        logger.error("POST %s request error: %s", url, e)
        raise RuntimeError(f"Failed to reach motion server at {url}: {e}") from e


async def _get_json(path: str) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    logger.debug("GET %s", url)

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                raise RuntimeError(f"Unexpected response from {url}: expected JSON object")

            logger.debug("GET %s -> status=%s keys=%s", url, response.status_code, sorted(data.keys()))
            return data
    except httpx.HTTPStatusError as e:
        body = e.response.text
        logger.error("GET %s failed status=%s body=%s", url, e.response.status_code, body[:500])
        raise RuntimeError(f"Motion server error {e.response.status_code} for {url}: {body[:4000]}") from e
    except httpx.RequestError as e:
        logger.error("GET %s request error: %s", url, e)
        raise RuntimeError(f"Failed to reach motion server at {url}: {e}") from e


async def _probe_motion_api() -> dict[str, Any]:
    """
    Probe motion_server.py API reachability.

    motion_server.py exposes POST /move and POST /stop and may expose GET /health.
    """
    url = f"{BASE_URL}{MOVE_PATH}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.options(url)
    except httpx.RequestError as e:
        return {
            "status": "offline",
            "base_url": BASE_URL,
            "api": "motion_server.py",
            "endpoints": {"move": MOVE_PATH, "stop": STOP_PATH},
            "reachable": False,
            "probe_method": "OPTIONS",
            "probe_url": url,
            "error": str(e),
        }

    if response.status_code == 404:
        return {
            "status": "offline",
            "base_url": BASE_URL,
            "api": "motion_server.py",
            "endpoints": {"move": MOVE_PATH, "stop": STOP_PATH},
            "reachable": False,
            "probe_method": "OPTIONS",
            "probe_url": url,
            "status_code": response.status_code,
            "error": "move endpoint not found",
        }

    return {
        "status": "online",
        "base_url": BASE_URL,
        "api": "motion_server.py",
        "endpoints": {"move": MOVE_PATH, "stop": STOP_PATH, "health": "/health"},
        "reachable": True,
        "probe_method": "OPTIONS",
        "probe_url": url,
        "status_code": response.status_code,
    }


async def _try_get_health() -> tuple[dict[str, Any] | None, str | None]:
    try:
        return await _get_json("/health"), None
    except Exception as e:
        return None, str(e)


@mcp_motion.tool()
async def motion_health() -> dict[str, Any]:
    """Check whether the motion server is online."""
    logger.debug("Tool motion_health called")
    health, health_error = await _try_get_health()
    if health is not None:
        return {
            **health,
            "base_url": BASE_URL,
            "api": "motion_server.py",
            "endpoints": {"move": MOVE_PATH, "stop": STOP_PATH, "health": "/health"},
            "reachable": True,
        }

    probe = await _probe_motion_api()
    return {
        **probe,
        "health_error": health_error,
    }


@mcp_motion.tool()
async def motion_stop() -> dict[str, Any]:
    """Stop the robot (sets target linear/angular to 0)."""
    logger.info("Tool motion_stop called")
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


@mcp_motion.tool()
async def motion_move(
    linear: float,
    angular: float,
    duration_s: Optional[float] = None,
) -> dict[str, Any]:
    """
    Command robot motion via the motion server.

    Inputs are clamped to +/- MOTION_MAX_LINEAR and +/- MOTION_MAX_ANGULAR for safety.
    If duration_s is provided and > 0, the server will automatically stop after that many seconds.
    Note: this confirms command acceptance by the HTTP API, not physical displacement.
    """
    linear_f = _validate_finite("linear", linear)
    angular_f = _validate_finite("angular", angular)

    linear_cmd = _clamp(linear_f, MAX_LINEAR)
    angular_cmd = _clamp(angular_f, MAX_ANGULAR)
    logger.info(
        "Tool motion_move requested linear=%.5f angular=%.5f -> commanded linear=%.5f angular=%.5f",
        linear_f,
        angular_f,
        linear_cmd,
        angular_cmd,
    )

    command_version = _reserve_motion_command()
    result = await _post_json(MOVE_PATH, {"linear": linear_cmd, "angular": angular_cmd})

    if duration_s is not None:
        duration_f = _validate_finite("duration_s", duration_s)
        if duration_f > 0:
            logger.debug("motion_move sleeping for duration_s=%.3f before auto-stop", duration_f)
            await asyncio.sleep(duration_f)
            if _is_latest_motion_command(command_version):
                await _post_json(STOP_PATH)
                logger.info("motion_move auto-stop completed after duration_s=%.3f", duration_f)
                result = {**result, "auto_stopped": True, "duration_s": duration_f}
            else:
                logger.info("motion_move skipped stale auto-stop for superseded command")
                result = {**result, "auto_stopped": False, "duration_s": duration_f, "auto_stop_skipped_stale": True}

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
        "base_url": BASE_URL,
        "endpoint": MOVE_PATH,
        "limits": {"max_linear": MAX_LINEAR, "max_angular": MAX_ANGULAR},
        "verification": verification,
        "physical_motion_confirmed": False,
    }


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18201,
    path: str = "/motion",
    options: dict = {},
) -> None:
    """Run the Motion MCP server."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting Motion MCP server base_url=%s timeout_s=%.2f max_linear=%.4f max_angular=%.4f at %s:%s%s",
        BASE_URL,
        HTTP_TIMEOUT_S,
        MAX_LINEAR,
        MAX_ANGULAR,
        host,
        port,
        path,
    )
    mcp_motion.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    mcp_motion.run()
