"""
TurtleBot Motion MCP Server V3.

Duration-based motion commands via direct ROS2 /cmd_vel publishing.
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


CMD_VEL_TOPIC = os.getenv("TBOT_CMD_VEL_TOPIC", "/cmd_vel")
ROS_NODE_NAME  = os.getenv("TBOT_ROS_NODE_NAME", "tbot_motion_mcp_v3")
CMD_VEL_QOS    = int(os.getenv("TBOT_CMD_VEL_QOS", "10"))

MAX_LINEAR = abs(_env_float("MOTION_MAX_LINEAR", 0.2))
MAX_ANGULAR = abs(_env_float("MOTION_MAX_ANGULAR", 1.0))
ANGULAR_SIGN = _env_float("MOTION_ANGULAR_SIGN", -1.0)

LIDAR_MCP_URL_V3 = os.getenv("TBOT_LIDAR_MCP_URL_V3", "http://127.0.0.1:18212/turtlebot-lidar-v3")

WALL_FOLLOW_KP = _env_float("WALL_FOLLOW_KP", 2.0)
WALL_FOLLOW_MAX_ANGULAR = _env_float("WALL_FOLLOW_MAX_ANGULAR", 0.5)

mcp_motion_v3 = FastMCP("TurtleBot Motion MCP Server V3")

# Module-level ROS2 state
_ros_lock: threading.Lock = threading.Lock()
_ros_node: Node | None = None
_ros_publisher: Any | None = None
_last_linear: float = 0.0
_last_angular: float = 0.0


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


def _ensure_ros() -> tuple[Node, Any]:
    global _ros_node, _ros_publisher
    with _ros_lock:
        if _ros_node is None:
            if not rclpy.ok():
                rclpy.init()
            _ros_node = Node(ROS_NODE_NAME)
            _ros_publisher = _ros_node.create_publisher(Twist, CMD_VEL_TOPIC, CMD_VEL_QOS)
    return _ros_node, _ros_publisher


def _publish_twist_sync(linear: float, angular: float) -> None:
    global _last_linear, _last_angular
    _, pub = _ensure_ros()
    msg = Twist()
    msg.linear.x = linear
    msg.angular.z = angular
    pub.publish(msg)
    _last_linear = linear
    _last_angular = angular


async def _publish_twist(linear: float, angular: float) -> dict[str, Any]:
    _publish_twist_sync(linear, angular)
    return {"linear": linear, "angular": angular, "topic": CMD_VEL_TOPIC}


async def _stop_robot() -> dict[str, Any]:
    return await _publish_twist(0.0, 0.0)


async def _try_get_health() -> tuple[dict[str, Any] | None, str | None]:
    try:
        _, pub = _ensure_ros()
        return {
            "linear": _last_linear,
            "angular": _last_angular,
            "subscriber_count": pub.get_subscription_count(),
        }, None
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
            "status": "online",
            "topic": CMD_VEL_TOPIC,
            "node": ROS_NODE_NAME,
            "subscriber_count": health.get("subscriber_count", 0),
        }
    return {
        "status": "offline",
        "topic": CMD_VEL_TOPIC,
        "node": ROS_NODE_NAME,
        "health_error": health_error,
    }


@mcp_motion_v3.tool()
async def tbot_motion_stop() -> dict[str, Any]:
    """Stop the robot immediately by setting linear and angular targets to zero."""
    return await _stop_robot()


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
    start_mono = time.monotonic()

    async with Client(LIDAR_MCP_URL_V3) as lidar:
        while True:
            elapsed = time.monotonic() - start_mono
            if elapsed >= duration_f:
                break

            try:
                raw = await lidar.call_tool("tbot_lidar_check_collision", {})
                collision = _extract_tool_dict(raw)
            except Exception:
                collision = {}

            risk_level = collision.get("risk_level", "clear")
            front_dist = (collision.get("distances") or {}).get("front")

            if risk_level == "stop":
                await _stop_robot()
                return {"status": "collision_risk", "front_distance": front_dist}

            effective_speed = clamped_speed * 0.5 if risk_level == "caution" else clamped_speed
            await _publish_twist(effective_speed, 0.0)
            await asyncio.sleep(0.2)

    stop_result = await _stop_robot()
    return {
        **stop_result,
        "status": "completed",
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

    clamped_speed = _clamp(abs(speed_f), MAX_LINEAR)
    wall_sign = 1.0 if direction_clean == "left" else -1.0
    start_mono = time.monotonic()
    ticks = 0

    async with Client(LIDAR_MCP_URL_V3) as lidar:
        while True:
            if time.monotonic() - start_mono >= timeout_f:
                await _stop_robot()
                return {"status": "timeout", "distance_traveled_ticks": ticks}

            try:
                raw = await lidar.call_tool("tbot_lidar_get_obstacle_distances", {"sector": "all"})
                distances = _extract_tool_dict(raw).get("distances", {})
            except Exception:
                distances = {}

            front_dist = distances.get("front")
            lateral_dist = distances.get(direction_clean)

            if front_dist is not None and front_dist <= stop_dist:
                await _stop_robot()
                return {"status": "obstacle_reached", "distance_traveled_ticks": ticks}

            if lateral_dist is not None:
                error = lateral_dist - target_dist
                angular_correction = _clamp(wall_sign * WALL_FOLLOW_KP * error, WALL_FOLLOW_MAX_ANGULAR)
            else:
                angular_correction = 0.0

            await _publish_twist(clamped_speed, angular_correction)
            ticks += 1
            await asyncio.sleep(0.2)


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

    result = await _publish_twist(linear_cmd, angular_cmd)

    if duration_s is not None:
        duration_f = _validate_finite("duration_s", duration_s)
        if duration_f > 0:
            if linear_cmd > 0:
                start_mono = time.monotonic()
                async with Client(LIDAR_MCP_URL_V3) as lidar:
                    while time.monotonic() - start_mono < duration_f:
                        try:
                            raw = await lidar.call_tool("tbot_lidar_check_collision", {})
                            collision = _extract_tool_dict(raw)
                            if collision.get("risk_level") == "stop":
                                front_dist = (collision.get("distances") or {}).get("front")
                                await _stop_robot()
                                return {
                                    "status": "collision_risk",
                                    "front_distance": front_dist,
                                    "linear_cmd": linear_cmd,
                                    "angular_cmd": angular_cmd,
                                }
                        except Exception:
                            pass
                        await asyncio.sleep(0.2)
            else:
                start_mono = time.monotonic()
                while time.monotonic() - start_mono < duration_f:
                    await _publish_twist(linear_cmd, angular_cmd)
                    await asyncio.sleep(0.02)
            stop_result = await _stop_robot()
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

    start_mono = time.monotonic()
    while time.monotonic() - start_mono < duration_f:
        await _publish_twist(0.0, angular_cmd)
        await asyncio.sleep(0.02)
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
    return {
        "moving": abs(_last_linear) > 1e-6 or abs(_last_angular) > 1e-6,
        "linear": _last_linear,
        "angular": _last_angular,
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
        await _publish_twist(clamped_speed, 0.0)

        async with Client(LIDAR_MCP_URL_V3) as lidar:
            while True:
                elapsed = time.monotonic() - start_mono
                if elapsed >= timeout_f:
                    await _stop_robot()
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
                        await _stop_robot()
                        return {"status": "collision_risk", "front_distance": final_distance_m}

                    if risk_level == "caution":
                        await _stop_robot()
                        # Nudge: small rotation to attempt repositioning
                        await _publish_twist(0.0, nudge_angular_cmd)
                        await asyncio.sleep(nudge_duration_s)
                        await _stop_robot()
                        # Resume forward motion
                        await _publish_twist(clamped_speed, 0.0)
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
                                await _stop_robot()
                                return {"status": "reached", "front_distance": final_distance_m}
                        except (TypeError, ValueError):
                            pass
                except Exception:
                    pass

                await asyncio.sleep(0.2)

    except Exception:
        try:
            await _stop_robot()
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
