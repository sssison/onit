"""
TurtleBot LiDAR MCP Server V2.

Subscribes to /scan and exposes general-purpose LiDAR analysis tools.
"""

import asyncio
import atexit
import logging
import math
import os
import threading
import time
from typing import Any, Optional

from fastmcp import FastMCP

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan

    RCLPY_AVAILABLE = True
    RCLPY_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment]
    qos_profile_sensor_data = None  # type: ignore[assignment]
    LaserScan = object  # type: ignore[assignment]
    RCLPY_AVAILABLE = False
    RCLPY_IMPORT_ERROR = e

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


LIDAR_TOPIC = os.getenv("LIDAR_TOPIC", "/scan")
NODE_NAME = os.getenv("LIDAR_NODE_NAME", "lidar_mcp_server_node_v2")
FRAME_LOG_EVERY = max(1, int(_env_float("LIDAR_FRAME_LOG_EVERY", 30)))

mcp_lidar_v2 = FastMCP("TurtleBot LiDAR MCP Server V2")


def _ensure_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _ensure_non_negative(name: str, value: float) -> float:
    parsed = _ensure_finite(name, value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _is_valid_range(value: Any, range_min: float, range_max: float) -> bool:
    if not isinstance(value, (int, float)):
        return False
    parsed = float(value)
    return math.isfinite(parsed) and range_min <= parsed <= range_max


def _angle_for_index_deg(scan: dict[str, Any], index: int) -> float:
    angle_rad = float(scan["angle_min"]) + (float(index) * float(scan["angle_increment"]))
    return math.degrees(angle_rad)


def _sector_indices(
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    num_points: int,
    center_rad: float,
    half_width_rad: float,
) -> tuple[list[int], str]:
    if num_points <= 0:
        return [], "no_data"
    if angle_increment <= 0:
        return [], "no_data"
    if half_width_rad < 0:
        return [], "no_data"

    sector_min = center_rad - half_width_rad
    sector_max = center_rad + half_width_rad

    if sector_max < angle_min or sector_min > angle_max:
        return [], "angle_out_of_range"

    overlap_min = max(sector_min, angle_min)
    overlap_max = min(sector_max, angle_max)

    start_idx = int(math.ceil((overlap_min - angle_min) / angle_increment))
    end_idx = int(math.floor((overlap_max - angle_min) / angle_increment))

    start_idx = max(0, min(num_points - 1, start_idx))
    end_idx = max(0, min(num_points - 1, end_idx))

    if start_idx > end_idx:
        return [], "angle_out_of_range"

    return list(range(start_idx, end_idx + 1)), "ok"


def _sector_valid_ranges(scan: dict[str, Any], center_deg: float, half_width_deg: float) -> dict[str, Any]:
    ranges = scan["ranges"]
    num_points = len(ranges)
    center_rad = math.radians(center_deg)
    half_width_rad = math.radians(half_width_deg)

    indices, status = _sector_indices(
        angle_min=float(scan["angle_min"]),
        angle_max=float(scan["angle_max"]),
        angle_increment=float(scan["angle_increment"]),
        num_points=num_points,
        center_rad=center_rad,
        half_width_rad=half_width_rad,
    )
    if status != "ok":
        return {
            "status": status,
            "indices": [],
            "valid_values": [],
            "valid_angles_deg": [],
            "total_count": 0,
            "valid_count": 0,
        }

    valid_values: list[float] = []
    valid_angles_deg: list[float] = []
    for index in indices:
        value = ranges[index]
        if _is_valid_range(value, float(scan["range_min"]), float(scan["range_max"])):
            valid_values.append(float(value))
            valid_angles_deg.append(_angle_for_index_deg(scan, index))

    return {
        "status": "ok" if valid_values else "no_data",
        "indices": indices,
        "valid_values": valid_values,
        "valid_angles_deg": valid_angles_deg,
        "total_count": len(indices),
        "valid_count": len(valid_values),
    }


def _percentiles(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}

    sorted_values = sorted(values)
    length = len(sorted_values)

    def _pick(p: float) -> float:
        index = int((p / 100.0) * (length - 1))
        return sorted_values[index]

    return {"p10": _pick(10.0), "p50": _pick(50.0), "p90": _pick(90.0)}


def _compute_sector_stats(scan: dict[str, Any], center_deg: float, half_width_deg: float) -> dict[str, Any]:
    extracted = _sector_valid_ranges(scan, center_deg=center_deg, half_width_deg=half_width_deg)

    if extracted["status"] == "angle_out_of_range":
        return {
            "status": "angle_out_of_range",
            "valid_count": 0,
            "total_count": 0,
            "min_m": None,
            "mean_m": None,
            "p10_m": None,
            "p50_m": None,
            "p90_m": None,
            "closest_angle_deg": None,
            "closest_range_m": None,
        }

    if extracted["status"] == "no_data":
        return {
            "status": "no_data",
            "valid_count": extracted["valid_count"],
            "total_count": extracted["total_count"],
            "min_m": None,
            "mean_m": None,
            "p10_m": None,
            "p50_m": None,
            "p90_m": None,
            "closest_angle_deg": None,
            "closest_range_m": None,
        }

    valid_values = extracted["valid_values"]
    valid_angles_deg = extracted["valid_angles_deg"]
    min_value = min(valid_values)
    min_index = valid_values.index(min_value)
    percentile_values = _percentiles(valid_values)

    return {
        "status": "ok",
        "valid_count": extracted["valid_count"],
        "total_count": extracted["total_count"],
        "min_m": min_value,
        "mean_m": sum(valid_values) / len(valid_values),
        "p10_m": percentile_values["p10"],
        "p50_m": percentile_values["p50"],
        "p90_m": percentile_values["p90"],
        "closest_angle_deg": valid_angles_deg[min_index],
        "closest_range_m": min_value,
    }


def _distance_at_angle_from_scan(scan: dict[str, Any], angle_deg: float, window_deg: float, method: str) -> dict[str, Any]:
    extracted = _sector_valid_ranges(scan, center_deg=angle_deg, half_width_deg=max(0.0, window_deg / 2.0))

    if extracted["status"] == "angle_out_of_range":
        return {
            "status": "angle_out_of_range",
            "distance_m": None,
            "closest_angle_deg": None,
            "samples_considered": 0,
            "samples_valid": 0,
        }

    if extracted["status"] == "no_data":
        return {
            "status": "no_data",
            "distance_m": None,
            "closest_angle_deg": None,
            "samples_considered": extracted["total_count"],
            "samples_valid": extracted["valid_count"],
        }

    valid_values = extracted["valid_values"]
    valid_angles = extracted["valid_angles_deg"]

    if method == "min":
        distance = min(valid_values)
    elif method == "median":
        sorted_values = sorted(valid_values)
        length = len(sorted_values)
        middle = length // 2
        if length % 2 == 1:
            distance = sorted_values[middle]
        else:
            distance = (sorted_values[middle - 1] + sorted_values[middle]) / 2.0
    elif method == "mean":
        distance = sum(valid_values) / len(valid_values)
    else:
        raise ValueError("method must be one of: min, median, mean")

    min_value = min(valid_values)
    min_index = valid_values.index(min_value)

    return {
        "status": "ok",
        "distance_m": distance,
        "closest_angle_deg": valid_angles[min_index],
        "samples_considered": extracted["total_count"],
        "samples_valid": extracted["valid_count"],
    }


def _collision_from_scan(
    scan: dict[str, Any],
    front_threshold_m: float,
    side_threshold_m: float,
    back_threshold_m: float,
    sector_half_width_deg: float,
    max_scan_age_s: float,
) -> dict[str, Any]:
    age_s = scan.get("age_s")
    frame_id = scan.get("frame_id")
    topic = scan.get("topic")

    sectors = {
        "front": (0.0, front_threshold_m),
        "left": (90.0, side_threshold_m),
        "right": (-90.0, side_threshold_m),
        "back": (180.0, back_threshold_m),
    }

    directions: dict[str, Any] = {}
    has_unknown = False
    has_collision = False
    has_caution = False

    for name, (center_deg, threshold) in sectors.items():
        stats = _compute_sector_stats(scan, center_deg=center_deg, half_width_deg=sector_half_width_deg)
        collision_state: bool | str

        if age_s is None or age_s > max_scan_age_s:
            collision_state = "unknown"
            has_unknown = True
        elif stats["status"] != "ok":
            collision_state = "unknown"
            has_unknown = True
        else:
            min_m = float(stats["min_m"])
            if min_m <= threshold:
                collision_state = True
                has_collision = True
            else:
                collision_state = False
                if min_m <= (threshold * 1.5):
                    has_caution = True

        directions[name] = {
            "min_m": stats["min_m"],
            "threshold_m": threshold,
            "collision": collision_state,
            "closest_angle_deg": stats["closest_angle_deg"],
            "status": stats["status"],
        }

    if has_collision:
        risk_level = "stop"
        recommended_action = "stop"
    elif has_unknown:
        risk_level = "unknown"
        recommended_action = "recheck"
    elif has_caution:
        risk_level = "caution"
        recommended_action = "slow_down"
    else:
        risk_level = "clear"
        recommended_action = "proceed"

    return {
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "directions": directions,
        "scan_age_s": age_s,
        "frame_id": frame_id,
        "topic": topic,
    }


def _find_clear_heading_from_scan(
    scan: dict[str, Any],
    search_min_deg: float,
    search_max_deg: float,
    step_deg: float,
    sector_half_width_deg: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    heading = search_min_deg
    while heading <= search_max_deg + 1e-9:
        stats = _compute_sector_stats(scan, center_deg=heading, half_width_deg=sector_half_width_deg)
        candidates.append(
            {
                "heading_deg": round(heading, 6),
                "status": stats["status"],
                "clearance_m": stats["min_m"],
            }
        )
        heading += step_deg

    valid_candidates = [item for item in candidates if item["status"] == "ok" and item["clearance_m"] is not None]
    valid_sorted = sorted(valid_candidates, key=lambda item: item["clearance_m"], reverse=True)

    best_heading_deg = valid_sorted[0]["heading_deg"] if valid_sorted else None
    best_clearance_m = valid_sorted[0]["clearance_m"] if valid_sorted else None
    top_headings = valid_sorted[:5]

    return {
        "status": "ok" if valid_sorted else "no_data",
        "best_heading_deg": best_heading_deg,
        "best_clearance_m": best_clearance_m,
        "candidates_evaluated": len(candidates),
        "top_headings": top_headings,
    }


if RCLPY_AVAILABLE:

    class LidarSubscriber(Node):
        def __init__(self, topic_name: str) -> None:
            super().__init__(NODE_NAME)
            self._topic_name = topic_name
            self._lock = threading.Lock()
            self._scan_condition = threading.Condition(self._lock)
            self._scan_count = 0
            self._latest_scan: Optional[dict[str, Any]] = None

            self.create_subscription(
                LaserScan,
                self._topic_name,
                self._on_message,
                qos_profile_sensor_data,
            )
            self.get_logger().info(f"Subscribed to topic: {self._topic_name}")

        def _on_message(self, msg: LaserScan) -> None:
            now_wall = time.time()
            now_mono = time.monotonic()
            stamp_s = None
            if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
                stamp_s = float(msg.header.stamp.sec) + (float(msg.header.stamp.nanosec) / 1_000_000_000.0)

            with self._scan_condition:
                self._scan_count += 1
                self._latest_scan = {
                    "topic": self._topic_name,
                    "frame_id": msg.header.frame_id,
                    "ranges": [float(value) for value in msg.ranges],
                    "intensities": [float(value) for value in msg.intensities],
                    "angle_min": float(msg.angle_min),
                    "angle_max": float(msg.angle_max),
                    "angle_increment": float(msg.angle_increment),
                    "range_min": float(msg.range_min),
                    "range_max": float(msg.range_max),
                    "stamp_s": stamp_s,
                    "received_unix_s": now_wall,
                    "received_mono_s": now_mono,
                    "scan_count": self._scan_count,
                }
                self._scan_condition.notify_all()
                if self._scan_count == 1 or self._scan_count % FRAME_LOG_EVERY == 0:
                    logger.debug(
                        "Received lidar scan count=%d points=%d frame_id=%s",
                        self._scan_count,
                        len(msg.ranges),
                        msg.header.frame_id,
                    )

        def snapshot(self) -> dict[str, Any]:
            with self._scan_condition:
                scan_present = self._latest_scan is not None
                latest_age_s = None
                frame_id = None
                num_points = 0
                if self._latest_scan is not None:
                    latest_age_s = max(0.0, time.monotonic() - float(self._latest_scan["received_mono_s"]))
                    frame_id = self._latest_scan["frame_id"]
                    num_points = len(self._latest_scan["ranges"])
                return {
                    "topic": self._topic_name,
                    "scan_present": scan_present,
                    "scan_count": self._scan_count,
                    "latest_age_s": latest_age_s,
                    "frame_id": frame_id,
                    "num_points": num_points,
                }

        def latest_scan(self) -> dict[str, Any]:
            with self._scan_condition:
                if self._latest_scan is None:
                    raise RuntimeError(
                        f"No scan received yet on topic '{self._topic_name}'. "
                        "Use tbot_lidar_health() or confirm the publisher is active."
                    )
                age_s = max(0.0, time.monotonic() - float(self._latest_scan["received_mono_s"]))
                return {
                    **self._latest_scan,
                    "age_s": age_s,
                }

        def wait_for_scan(self, timeout_s: float, min_scan_count: int = 0) -> bool:
            deadline = time.monotonic() + timeout_s
            with self._scan_condition:
                while self._scan_count <= min_scan_count:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._scan_condition.wait(timeout=remaining)
                return True

else:

    class LidarSubscriber:  # pragma: no cover - fallback stub
        def __init__(self, topic_name: str) -> None:
            raise RuntimeError(f"rclpy is required for lidar MCP server but is unavailable: {RCLPY_IMPORT_ERROR}")


_node_ready = threading.Event()
_ros_thread: Optional[threading.Thread] = None
_lidar_node: Optional[LidarSubscriber] = None
_spin_error: Optional[BaseException] = None


def _ros_spin_worker() -> None:
    global _lidar_node, _spin_error
    if not RCLPY_AVAILABLE:
        _spin_error = RuntimeError(f"rclpy unavailable: {RCLPY_IMPORT_ERROR}")
        _node_ready.set()
        return
    try:
        if not rclpy.ok():
            rclpy.init()
        _lidar_node = LidarSubscriber(LIDAR_TOPIC)
        _node_ready.set()
        rclpy.spin(_lidar_node)
    except BaseException as exc:
        _spin_error = exc
        logger.exception("ROS spin worker failed: %s", exc)
        _node_ready.set()
    finally:
        if _lidar_node is not None:
            _lidar_node.destroy_node()
            _lidar_node = None
        if rclpy.ok():
            rclpy.shutdown()


def _start_ros_thread() -> None:
    global _ros_thread, _spin_error
    if _ros_thread is not None and _ros_thread.is_alive():
        return
    _spin_error = None
    _node_ready.clear()
    _ros_thread = threading.Thread(target=_ros_spin_worker, name="lidar-mcp-v2-ros-spin", daemon=True)
    _ros_thread.start()


def _get_lidar_node() -> LidarSubscriber:
    _start_ros_thread()
    if not _node_ready.wait(timeout=3.0):
        raise RuntimeError("LiDAR ROS node did not start within 3 seconds.")
    if _spin_error is not None:
        raise RuntimeError(f"LiDAR ROS node failed to start: {_spin_error}") from _spin_error
    if _lidar_node is None:
        raise RuntimeError("LiDAR ROS node unavailable.")
    return _lidar_node


def _shutdown_ros() -> None:
    if RCLPY_AVAILABLE and rclpy.ok():
        rclpy.shutdown()


atexit.register(_shutdown_ros)


@mcp_lidar_v2.tool()
async def tbot_lidar_health() -> dict[str, Any]:
    """Check if the LiDAR sensor is online and receiving data."""
    try:
        node = _get_lidar_node()
    except Exception as e:
        return {
            "status": "offline",
            "ros_available": RCLPY_AVAILABLE,
            "topic": LIDAR_TOPIC,
            "scan_present": False,
            "scan_count": 0,
            "latest_age_s": None,
            "frame_id": None,
            "error": str(e),
        }

    snapshot = node.snapshot()
    status = "online" if snapshot["scan_present"] else "waiting_for_scans"
    return {"status": status, "ros_available": RCLPY_AVAILABLE, **snapshot}


@mcp_lidar_v2.tool()
async def tbot_lidar_distance_at_angle(
    angle_deg: float = 0.0,
    window_deg: float = 2.0,
    method: str = "min",
) -> dict[str, Any]:
    """Measure the distance to an object in a given direction (0=front, 90=left, -90=right, 180=back). Use this to estimate how far away a detected object is before approaching it."""
    angle_value = _ensure_finite("angle_deg", angle_deg)
    window_value = _ensure_non_negative("window_deg", window_deg)
    method_value = method.strip().lower()

    result = _distance_at_angle_from_scan(
        scan=_get_lidar_node().latest_scan(),
        angle_deg=angle_value,
        window_deg=window_value,
        method=method_value,
    )
    return {
        **result,
        "angle_deg": angle_value,
        "window_deg": window_value,
        "method": method_value,
    }


@mcp_lidar_v2.tool()
async def tbot_lidar_check_collision(
    front_threshold_m: float = 0.25,
    side_threshold_m: float = 0.20,
    back_threshold_m: float = 0.25,
    sector_half_width_deg: float = 20.0,
    max_scan_age_s: float = 0.5,
) -> dict[str, Any]:
    """Check for collision risk before moving forward or backward toward an object. Returns risk_level: 'clear' (safe to proceed), 'caution' (slow down), or 'stop' (do not move). Only needed before linear motion — do NOT call this for rotation."""
    front_threshold = _ensure_non_negative("front_threshold_m", front_threshold_m)
    side_threshold = _ensure_non_negative("side_threshold_m", side_threshold_m)
    back_threshold = _ensure_non_negative("back_threshold_m", back_threshold_m)
    sector_half_width = _ensure_non_negative("sector_half_width_deg", sector_half_width_deg)
    max_scan_age = _ensure_non_negative("max_scan_age_s", max_scan_age_s)

    return _collision_from_scan(
        scan=_get_lidar_node().latest_scan(),
        front_threshold_m=front_threshold,
        side_threshold_m=side_threshold,
        back_threshold_m=back_threshold,
        sector_half_width_deg=sector_half_width,
        max_scan_age_s=max_scan_age,
    )


def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18208,
    path: str = "/turtlebot-lidar-v2",
    options: dict = {},
) -> None:
    """Run the TurtleBot LiDAR MCP Server V2."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting TurtleBot LiDAR MCP V2 topic=%s node_name=%s at %s:%s%s",
        LIDAR_TOPIC,
        NODE_NAME,
        host,
        port,
        path,
    )
    _start_ros_thread()
    mcp_lidar_v2.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
