"""
TurtleBot Motion MCP Server V3.

Duration-based motion commands via direct ROS2 /cmd_vel publishing.

Angular sign convention
=======================
ROS2 standard: angular.z > 0 = CCW (left), angular.z < 0 = CW (right).

MOTION_ANGULAR_SIGN (env var, default -1.0) corrects for robots whose physical
yaw frame differs from the ROS2 convention.  With the default value (-1.0) and
the input_frame_sign logic in tbot_motion_turn:

  "left"  → input_frame_sign = -1 → angular_cmd = -1 × speed × (-1) = +speed  (CCW = left  ✓)
  "right" → input_frame_sign = +1 → angular_cmd = +1 × speed × (-1) = -speed  (CW  = right ✓)

tbot_motion_move() sends angular.z directly without ANGULAR_SIGN remapping.
Use positive values to turn left (CCW) and negative values to turn right (CW).

If the robot turns the wrong way, set MOTION_ANGULAR_SIGN=1.0.
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
from typing import Any

import rclpy
from fastmcp import Client, FastMCP
from geometry_msgs.msg import Twist
from rclpy.node import Node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CMD_VEL_TOPIC = os.getenv("TBOT_CMD_VEL_TOPIC", "/cmd_vel")
ROS_NODE_NAME  = os.getenv("TBOT_ROS_NODE_NAME", "tbot_motion_mcp_v3")
CMD_VEL_QOS    = int(os.getenv("TBOT_CMD_VEL_QOS", "10"))

MAX_LINEAR   = abs(_env_float("MOTION_MAX_LINEAR", 0.2))
MAX_ANGULAR  = abs(_env_float("MOTION_MAX_ANGULAR", 1.0))
ANGULAR_SIGN = _env_float("MOTION_ANGULAR_SIGN", -1.0)

LIDAR_MCP_URL_V3       = os.getenv("TBOT_LIDAR_MCP_URL_V3", "http://127.0.0.1:18212/turtlebot-lidar-v3")
# Must exceed the LiDAR server's LIDAR_SCAN_TIMEOUT_S (default 3.0 s) plus HTTP overhead.
LIDAR_CALL_TIMEOUT_S   = _env_float("LIDAR_CALL_TIMEOUT_S", 5.0)
# Consecutive ticks with no LiDAR response before a forward motion loop stops.
LIDAR_MAX_CONSEC_FAILS = int(os.getenv("LIDAR_MAX_CONSEC_FAILS", "3"))

WALL_FOLLOW_KP          = _env_float("WALL_FOLLOW_KP", 2.0)
WALL_FOLLOW_MAX_ANGULAR = _env_float("WALL_FOLLOW_MAX_ANGULAR", 0.5)

# Control loop tick rates
CONTROL_HZ = 10   # general motion loops (forward, wall-follow, move) → 0.1 s/tick
TURN_HZ    = 25   # turn-in-place loop                                 → 0.04 s/tick

mcp_motion_v3 = FastMCP("TurtleBot Motion MCP Server V3")

# ---------------------------------------------------------------------------
# ROS2 module-level state
# ---------------------------------------------------------------------------

_ros_lock: threading.Lock = threading.Lock()    # guards publisher calls and velocity reads/writes
_async_ros_lock: asyncio.Lock = asyncio.Lock()  # guards async initialisation
_ros_initialized: bool = False
_ros_node: Node | None = None
_ros_publisher: Any | None = None
_last_linear: float = 0.0
_last_angular: float = 0.0

# ---------------------------------------------------------------------------
# LiDAR client state
# ---------------------------------------------------------------------------

_lidar_client: Client | None = None
_lidar_client_lock: asyncio.Lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


def _get_last_velocities() -> tuple[float, float]:
    """Return (linear, angular) thread-safely."""
    with _ros_lock:
        return _last_linear, _last_angular


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


# ---------------------------------------------------------------------------
# ROS2 initialisation
# ---------------------------------------------------------------------------

async def _ensure_ros() -> tuple[Node, Any]:
    """Async-safe, idempotent ROS2 initialiser (double-checked locking)."""
    global _ros_initialized, _ros_node, _ros_publisher
    if _ros_initialized:
        return _ros_node, _ros_publisher
    async with _async_ros_lock:
        if not _ros_initialized:
            if not rclpy.ok():
                rclpy.init()
            _ros_node = Node(ROS_NODE_NAME)
            _ros_publisher = _ros_node.create_publisher(Twist, CMD_VEL_TOPIC, CMD_VEL_QOS)
            _ros_initialized = True
    return _ros_node, _ros_publisher


def _publish_twist_sync(linear: float, angular: float) -> None:
    """Publish a Twist and record last velocities under the threading lock.

    Caller must have already called ``await _ensure_ros()`` to guarantee
    _ros_publisher is initialised.
    """
    global _last_linear, _last_angular
    with _ros_lock:
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        _ros_publisher.publish(msg)
        _last_linear = linear
        _last_angular = angular


async def _publish_twist(linear: float, angular: float) -> dict[str, Any]:
    await _ensure_ros()
    _publish_twist_sync(linear, angular)
    return {"linear": linear, "angular": angular, "topic": CMD_VEL_TOPIC}


async def _stop_robot() -> dict[str, Any]:
    return await _publish_twist(0.0, 0.0)


# ---------------------------------------------------------------------------
# LiDAR client helpers
# ---------------------------------------------------------------------------

async def _get_lidar_client() -> Client:
    """Return the cached LiDAR client, creating and connecting it if needed."""
    global _lidar_client
    async with _lidar_client_lock:
        if _lidar_client is None:
            client = Client(LIDAR_MCP_URL_V3)
            await client.__aenter__()
            _lidar_client = client
    return _lidar_client


async def _call_lidar_tool(tool_name: str, args: dict) -> dict[str, Any]:
    """Call a LiDAR tool with LIDAR_CALL_TIMEOUT_S timeout, falling back to a per-call client on failure."""
    global _lidar_client
    try:
        client = await _get_lidar_client()
        raw = await asyncio.wait_for(client.call_tool(tool_name, args), timeout=LIDAR_CALL_TIMEOUT_S)
        return _extract_tool_dict(raw)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("LiDAR call failed (cached client): %s", e)
        async with _lidar_client_lock:
            _lidar_client = None
    # Fallback: fresh per-call client
    async with Client(LIDAR_MCP_URL_V3) as fresh:
        raw = await asyncio.wait_for(fresh.call_tool(tool_name, args), timeout=LIDAR_CALL_TIMEOUT_S)
        return _extract_tool_dict(raw)


# ---------------------------------------------------------------------------
# Concurrent forward-motion helper
# ---------------------------------------------------------------------------

async def _run_forward_with_collision_guard(
    linear: float,
    angular: float,
    duration_f: float,
) -> dict[str, Any]:
    """
    Publish cmd_vel at CONTROL_HZ for up to duration_f seconds.

    LiDAR collision checks run concurrently in a background asyncio.Task so
    the /cmd_vel stream is never blocked waiting for a slow LiDAR round-trip
    (~3-5 s). The last-known risk level is applied on every publish tick.

    Returns dict with 'status' key: "completed" | "collision_risk" | "lidar_unavailable".
    """
    tick = 1.0 / CONTROL_HZ
    start_mono = time.monotonic()
    last_risk: str = "clear"
    last_front_dist: float | None = None
    consec_fails: int = 0
    lidar_task: asyncio.Task | None = None

    try:
        while True:
            elapsed = time.monotonic() - start_mono
            if elapsed >= duration_f:
                break

            # Harvest a completed LiDAR background task
            if lidar_task is not None and lidar_task.done():
                try:
                    col = lidar_task.result()
                    last_risk = col.get("risk_level", "clear")
                    fd = (col.get("distances") or {}).get("front")
                    if fd is not None:
                        last_front_dist = float(fd)
                    consec_fails = 0
                except Exception as exc:
                    logger.warning("LiDAR background task failed: %s", exc)
                    consec_fails += 1
                lidar_task = None

            # Fire a new LiDAR task if none is in flight
            if lidar_task is None:
                if consec_fails > LIDAR_MAX_CONSEC_FAILS:
                    await _stop_robot()
                    return {"status": "lidar_unavailable", "front_distance": last_front_dist}
                lidar_task = asyncio.create_task(
                    _call_lidar_tool("tbot_lidar_check_collision", {})
                )

            # Act on last known risk — publish every tick regardless of LiDAR state
            if last_risk == "stop":
                lidar_task.cancel()
                await _stop_robot()
                return {"status": "collision_risk", "front_distance": last_front_dist}

            effective_linear = linear * 0.5 if last_risk == "caution" else linear
            await _publish_twist(effective_linear, angular)
            await asyncio.sleep(tick)

    finally:
        if lidar_task is not None:
            lidar_task.cancel()

    stop_result = await _stop_robot()
    return {**stop_result, "status": "completed", "front_distance": last_front_dist}


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp_motion_v3.tool()
async def tbot_motion_stop() -> dict[str, Any]:
    """Stop the robot immediately by setting linear and angular targets to zero."""
    return await _stop_robot()


@mcp_motion_v3.tool()
async def tbot_motion_disconnect_lidar() -> dict[str, Any]:
    """Close and reset the cached LiDAR client connection."""
    global _lidar_client
    async with _lidar_client_lock:
        if _lidar_client is not None:
            try:
                await _lidar_client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error closing LiDAR client: %s", e)
            _lidar_client = None
            return {"status": "disconnected"}
    return {"status": "not_connected"}


@mcp_motion_v3.tool()
async def tbot_motion_move_forward(
    speed: float,
    duration_seconds: float,
) -> dict[str, Any]:
    """
    Move the robot forward at speed m/s for duration_seconds, then stop.

    COLLISION GUARD IS BUILT IN — do NOT call tbot_lidar_check_collision or
    tbot_lidar_get_obstacle_distances before this tool. The loop already polls
    the LiDAR server internally on every tick and halts on risk_level "stop".
    Pre-calling LiDAR queues an extra slow scan (~3 s) and delays the move.

    Returns {"status": "completed"|"collision_risk"|"lidar_unavailable", ...}.
    """
    speed_f    = _validate_finite("speed", speed)
    duration_f = _validate_finite("duration_seconds", duration_seconds)
    if duration_f <= 0:
        raise ValueError("duration_seconds must be > 0")

    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    result = await _run_forward_with_collision_guard(clamped_speed, 0.0, duration_f)
    return {
        **result,
        "speed": clamped_speed,
        "duration_seconds": duration_f,
        "was_clamped": abs(speed_f) != clamped_speed,
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

    LiDAR IS POLLED INTERNALLY on every tick — do NOT call tbot_lidar_check_collision
    or tbot_lidar_get_obstacle_distances before this tool. Pre-calling adds ~3 s delay.

    Returns {"status": "obstacle_reached"|"timeout", "distance_traveled_ticks": int}.
    """
    direction_clean = direction.strip().lower() if isinstance(direction, str) else ""
    if direction_clean not in ("left", "right"):
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

    target_dist = _validate_finite("target_distance_m", target_distance_m)
    speed_f     = _validate_finite("speed", speed)
    timeout_f   = _validate_finite("timeout_s", timeout_s)
    stop_dist   = _validate_finite("stop_distance_m", stop_distance_m)

    if target_dist <= 0:
        raise ValueError("target_distance_m must be > 0")
    if timeout_f <= 0:
        raise ValueError("timeout_s must be > 0")
    if stop_dist <= 0:
        raise ValueError("stop_distance_m must be > 0")

    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    wall_sign     = 1.0 if direction_clean == "left" else -1.0
    start_mono    = time.monotonic()
    ticks         = 0
    tick          = 1.0 / CONTROL_HZ

    last_distances: dict[str, Any] = {}
    lidar_task: asyncio.Task | None = None

    try:
        while True:
            if time.monotonic() - start_mono >= timeout_f:
                await _stop_robot()
                return {"status": "timeout", "distance_traveled_ticks": ticks}

            # Harvest completed LiDAR background task
            if lidar_task is not None and lidar_task.done():
                try:
                    last_distances = lidar_task.result().get("distances", {})
                except Exception as exc:
                    logger.warning("LiDAR background task failed: %s", exc)
                lidar_task = None

            # Fire new LiDAR task if none in flight
            if lidar_task is None:
                lidar_task = asyncio.create_task(
                    _call_lidar_tool("tbot_lidar_get_obstacle_distances", {"sector": "all"})
                )

            front_dist   = last_distances.get("front")
            lateral_dist = last_distances.get(direction_clean)

            if front_dist is not None and front_dist <= stop_dist:
                lidar_task.cancel()
                await _stop_robot()
                return {"status": "obstacle_reached", "distance_traveled_ticks": ticks}

            if lateral_dist is not None:
                error              = lateral_dist - target_dist
                angular_correction = _clamp(wall_sign * WALL_FOLLOW_KP * error, WALL_FOLLOW_MAX_ANGULAR)
            else:
                angular_correction = 0.0

            await _publish_twist(clamped_speed, angular_correction)
            ticks += 1
            await asyncio.sleep(tick)

    finally:
        if lidar_task is not None:
            lidar_task.cancel()


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

    COLLISION GUARD IS BUILT IN for timed forward motion — do NOT call
    tbot_lidar_check_collision or tbot_lidar_get_obstacle_distances before this
    tool when using duration_s. The loop polls LiDAR every tick internally.
    Pre-calling LiDAR adds ~3 s of delay before the robot starts moving.
    Pure-rotation commands (linear=0) skip LiDAR and turn immediately.
    """
    linear_f  = _validate_finite("linear", linear)
    angular_f = _validate_finite("angular", angular)

    linear_cmd  = _clamp(linear_f, MAX_LINEAR)
    angular_cmd = _clamp(angular_f, MAX_ANGULAR)

    result = await _publish_twist(linear_cmd, angular_cmd)

    if duration_s is not None:
        duration_f = _validate_finite("duration_s", duration_s)
        if duration_f > 0:
            tick          = 1.0 / CONTROL_HZ
            pure_rotation = (linear_cmd == 0.0 and angular_cmd != 0.0)
            start_mono    = time.monotonic()

            if pure_rotation:
                # Pure rotation: skip LiDAR overhead, republish continuously
                while time.monotonic() - start_mono < duration_f:
                    await _publish_twist(linear_cmd, angular_cmd)
                    await asyncio.sleep(tick)
                stop_result = await _stop_robot()
                return {
                    **stop_result,
                    "status": "completed",
                    "linear_cmd": linear_cmd,
                    "angular_cmd": angular_cmd,
                    "duration_s": duration_f,
                }
            else:
                # Forward/combined: concurrent LiDAR guard, unblocked publishing
                guard = await _run_forward_with_collision_guard(linear_cmd, angular_cmd, duration_f)
                return {**guard, "linear_cmd": linear_cmd, "angular_cmd": angular_cmd, "duration_s": duration_f}

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

    speed_f    = _validate_finite("speed", speed)
    duration_f = _validate_finite("duration_seconds", duration_seconds)
    if duration_f <= 0:
        raise ValueError("duration_seconds must be > 0")

    clamped_speed = _clamp(abs(speed_f), MAX_ANGULAR)
    if clamped_speed == 0.0:
        raise ValueError(
            f"Effective angular speed is 0 (speed={speed_f!r}, MAX_ANGULAR={MAX_ANGULAR!r}). "
            "Pass a non-zero speed within the allowed range."
        )

    # ROS2: angular.z > 0 = CCW = left, angular.z < 0 = CW = right.
    # With ANGULAR_SIGN = -1.0 (default):
    #   "left"  → input_frame_sign = -1 → angular_cmd = -1 × speed × -1 = +speed  (CCW = left  ✓)
    #   "right" → input_frame_sign = +1 → angular_cmd = +1 × speed × -1 = -speed  (CW  = right ✓)
    # If the robot turns backward, flip MOTION_ANGULAR_SIGN to +1.0.
    input_frame_sign = 1.0 if direction_clean == "right" else -1.0
    angular_cmd      = input_frame_sign * clamped_speed * ANGULAR_SIGN

    if angular_cmd == 0.0:
        raise ValueError(
            f"angular_cmd is 0 — check MOTION_ANGULAR_SIGN env var "
            f"(current value: {ANGULAR_SIGN!r}). Must be non-zero."
        )

    tick       = 1.0 / TURN_HZ
    start_mono = time.monotonic()
    while time.monotonic() - start_mono < duration_f:
        await _publish_twist(0.0, angular_cmd)
        await asyncio.sleep(tick)

    stop_result = await _stop_robot()
    return {
        **stop_result,
        "status": "completed",
        "direction": direction_clean,
        "speed": clamped_speed,
        "angular_cmd": angular_cmd,
        "angular_sign": ANGULAR_SIGN,
        "duration_seconds": duration_f,
        "was_clamped": abs(speed_f) != clamped_speed,
    }


@mcp_motion_v3.tool()
async def tbot_motion_get_robot_status() -> dict[str, Any]:
    """Get the current robot motion status — whether it is moving, and current linear/angular values."""
    linear, angular = _get_last_velocities()
    return {
        "moving": abs(linear) > 1e-6 or abs(angular) > 1e-6,
        "linear": linear,
        "angular": angular,
        "topic": CMD_VEL_TOPIC,
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
    Returns status: "reached" | "collision_risk" | "timeout" | "lidar_unavailable".

    LiDAR IS POLLED INTERNALLY on every tick — do NOT call tbot_lidar_check_collision
    or tbot_lidar_get_obstacle_distances before this tool. Pre-calling adds ~3 s delay.
    """
    target_dist = _validate_finite("target_distance_m", target_distance_m)
    stop_dist   = _validate_finite("stop_distance_m", stop_distance_m)
    speed_f     = _validate_finite("speed", speed)
    timeout_f   = _validate_finite("timeout_s", timeout_s)

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
    nudge_speed       = min(0.5, MAX_ANGULAR)
    nudge_angular_cmd = nudge_speed * ANGULAR_SIGN
    nudge_duration_s  = (10.0 * math.pi / 180.0) / max(0.01, nudge_speed)

    start_mono       = time.monotonic()
    final_dist_m: float | None = None
    consec_fails: int = 0
    tick = 1.0 / CONTROL_HZ
    lidar_task: asyncio.Task | None = None

    # Seed last-known values so the first tick doesn't stop prematurely
    last_risk: str = "clear"
    last_front: float | None = None

    try:
        await _publish_twist(clamped_speed, 0.0)

        while True:
            elapsed = time.monotonic() - start_mono
            if elapsed >= timeout_f:
                await _stop_robot()
                return {"status": "timeout", "front_distance": final_dist_m}

            # Harvest completed LiDAR task
            # One call covers both safety and distance — collision check returns distances.front
            if lidar_task is not None and lidar_task.done():
                try:
                    col = lidar_task.result()
                    last_risk = col.get("risk_level", "clear")
                    fd = (col.get("distances") or {}).get("front")
                    if fd is not None:
                        last_front = float(fd)
                        final_dist_m = last_front
                    consec_fails = 0
                except Exception as exc:
                    logger.warning("LiDAR background task failed: %s", exc)
                    consec_fails += 1
                lidar_task = None

            # Fire new LiDAR task if none in flight
            if lidar_task is None:
                if consec_fails > LIDAR_MAX_CONSEC_FAILS:
                    await _stop_robot()
                    return {"status": "lidar_unavailable", "front_distance": final_dist_m}
                lidar_task = asyncio.create_task(
                    _call_lidar_tool("tbot_lidar_check_collision", {"front_threshold_m": stop_dist})
                )

            # Act on last known state
            if last_risk == "stop":
                lidar_task.cancel()
                await _stop_robot()
                return {"status": "collision_risk", "front_distance": final_dist_m}

            if last_risk == "caution":
                await _stop_robot()
                await _publish_twist(0.0, nudge_angular_cmd)
                await asyncio.sleep(nudge_duration_s)
                await _stop_robot()
                last_risk = "clear"
                await _publish_twist(clamped_speed, 0.0)
            else:
                await _publish_twist(clamped_speed, 0.0)

            # Target-distance reached check (uses front dist from collision result)
            if last_front is not None and last_front <= target_dist:
                lidar_task.cancel()
                await _stop_robot()
                return {"status": "reached", "front_distance": final_dist_m}

            await asyncio.sleep(tick)

    except Exception:
        try:
            await _stop_robot()
        except Exception:
            pass
        raise
    finally:
        if lidar_task is not None:
            lidar_task.cancel()


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
        "Starting TurtleBot Motion MCP V3 topic=%s node=%s at %s:%s%s",
        CMD_VEL_TOPIC,
        ROS_NODE_NAME,
        host,
        port,
        path,
    )
    mcp_motion_v3.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
