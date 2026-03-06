"""
TurtleBot LiDAR MCP Server V3.

Uses short-lived per-call rclpy nodes — no persistent background thread.
Each tool call spins a new node, waits for one scan, then destroys it.
"""

import asyncio
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
SCAN_TIMEOUT_S = _env_float("LIDAR_SCAN_TIMEOUT_S", 3.0)
NODE_NAME_PREFIX = os.getenv("LIDAR_NODE_NAME_PREFIX", "lidar_mcp_v3_node")

mcp_lidar_v3 = FastMCP("TurtleBot LiDAR MCP Server V3")

_rclpy_init_lock = threading.Lock()

_SECTOR_CENTERS: dict[str, float] = {
    "front": 0.0,
    "left": 90.0,
    "right": -90.0,
    "rear": 180.0,
}


# --- Helper functions (reused from v2 pattern) ---


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


def _grid_index_from_xy(
    x_m: float,
    y_m: float,
    resolution_m: float,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    if resolution_m <= 0:
        return None
    cx = width // 2
    cy = height // 2
    gx = int(round(x_m / resolution_m)) + cx
    gy = cy - int(round(y_m / resolution_m))
    if gx < 0 or gx >= width or gy < 0 or gy >= height:
        return None
    return gx, gy


def _mark_ray_free_cells(
    grid: list[list[int]],
    angle_rad: float,
    distance_m: float,
    resolution_m: float,
) -> None:
    width = len(grid[0]) if grid else 0
    height = len(grid)
    if width == 0 or height == 0:
        return
    if distance_m <= 0:
        return

    step = 0.0
    while step < distance_m:
        x_m = step * math.cos(angle_rad)
        y_m = step * math.sin(angle_rad)
        index = _grid_index_from_xy(x_m, y_m, resolution_m, width, height)
        if index is None:
            step += resolution_m
            continue
        gx, gy = index
        if grid[gy][gx] != 100:
            grid[gy][gx] = 0
        step += resolution_m


# --- Short-lived rclpy node pattern ---


def _get_one_scan_sync(timeout_s: float) -> dict[str, Any]:
    """
    Spin a short-lived rclpy node to receive one LaserScan message.

    Initializes rclpy if not already running, creates a temporary subscriber node,
    spins until one scan arrives or timeout is reached, then destroys the node.
    Raises RuntimeError if no scan is received within timeout_s.
    """
    if not RCLPY_AVAILABLE:
        raise RuntimeError(f"rclpy is not available: {RCLPY_IMPORT_ERROR}")

    with _rclpy_init_lock:
        if not rclpy.ok():
            rclpy.init()

    import uuid

    node_name = f"{NODE_NAME_PREFIX}_{uuid.uuid4().hex[:8]}"
    scan_data: dict[str, Any] = {}
    received = threading.Event()

    class _OneShotNode(Node):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(node_name)
            self.create_subscription(
                LaserScan,
                LIDAR_TOPIC,
                self._callback,
                qos_profile_sensor_data,
            )

        def _callback(self, msg: Any) -> None:
            if received.is_set():
                return
            now_mono = time.monotonic()
            stamp_s: float | None = None
            if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
                stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1_000_000_000.0
            scan_data.update({
                "topic": LIDAR_TOPIC,
                "frame_id": msg.header.frame_id,
                "ranges": [float(v) for v in msg.ranges],
                "intensities": [float(v) for v in msg.intensities],
                "angle_min": float(msg.angle_min),
                "angle_max": float(msg.angle_max),
                "angle_increment": float(msg.angle_increment),
                "range_min": float(msg.range_min),
                "range_max": float(msg.range_max),
                "stamp_s": stamp_s,
                "received_mono_s": now_mono,
                "received_unix_s": time.time(),
            })
            received.set()

    node = _OneShotNode()
    try:
        deadline = time.monotonic() + timeout_s
        while not received.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(
                    f"No LiDAR scan received within {timeout_s}s on topic {LIDAR_TOPIC!r}"
                )
            rclpy.spin_once(node, timeout_sec=min(0.1, remaining))
    finally:
        node.destroy_node()

    scan_data["age_s"] = max(0.0, time.monotonic() - scan_data["received_mono_s"])
    return scan_data


async def _get_one_scan(timeout_s: float = SCAN_TIMEOUT_S) -> dict[str, Any]:
    """Async wrapper: run _get_one_scan_sync in a thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_one_scan_sync, timeout_s)


# --- Tools ---


@mcp_lidar_v3.tool()
async def tbot_lidar_get_obstacle_distances(sector: str = "all") -> dict[str, Any]:
    """
    Return the distance to the nearest obstacle in a named sector.

    sector: "front" | "left" | "right" | "rear" | "all"
    Returns {status, sector, distance_m, distances: {front, left, right, rear}}.

    NOTE: Motion planners may call this to precompute movement distances.
    For preplanned forward motion in V3, LiDAR is read before moving.
    """
    sector_clean = sector.strip().lower() if isinstance(sector, str) else "all"
    valid_sectors = set(_SECTOR_CENTERS.keys()) | {"all"}
    if sector_clean not in valid_sectors:
        raise ValueError(f"sector must be one of: front, left, right, rear, all. Got: {sector!r}")

    try:
        scan = await _get_one_scan()
    except Exception as e:
        return {
            "status": "no_scan",
            "sector": sector_clean,
            "distance_m": None,
            "distances": {s: None for s in _SECTOR_CENTERS},
            "error": str(e),
        }

    distances: dict[str, float | None] = {}
    for name, center_deg in _SECTOR_CENTERS.items():
        # Narrow front cone (±20°) so side obstacles don't register as forward hazards.
        half_width = 20.0 if name == "front" else 45.0
        stats = _compute_sector_stats(scan, center_deg=center_deg, half_width_deg=half_width)
        distances[name] = float(stats["min_m"]) if stats["status"] == "ok" and stats["min_m"] is not None else None

    if sector_clean == "all":
        min_dist: float | None = None
        for d in distances.values():
            if d is not None and (min_dist is None or d < min_dist):
                min_dist = d
        return {
            "status": "ok",
            "sector": sector_clean,
            "distance_m": min_dist,
            "distances": distances,
        }

    return {
        "status": "ok",
        "sector": sector_clean,
        "distance_m": distances.get(sector_clean),
        "distances": distances,
    }


@mcp_lidar_v3.tool()
async def tbot_lidar_get_distance_at_angle(
    angle_deg: float,
    half_width_deg: float = 2.0,
    statistic: str = "min",
) -> dict[str, Any]:
    """
    Return LiDAR distance around a specific heading angle.

    angle_deg: heading to sample (0=front, 90=left, -90=right).
    half_width_deg: half-width of the cone around angle_deg.
    statistic: "min" or "mean" distance within the cone.
    """
    if not isinstance(angle_deg, (int, float)) or not math.isfinite(float(angle_deg)):
        raise ValueError("angle_deg must be a finite number")
    if not isinstance(half_width_deg, (int, float)) or not math.isfinite(float(half_width_deg)):
        raise ValueError("half_width_deg must be a finite number")
    half_width_f = float(half_width_deg)
    if half_width_f <= 0:
        raise ValueError("half_width_deg must be > 0")

    statistic_clean = statistic.strip().lower() if isinstance(statistic, str) else "min"
    if statistic_clean not in ("min", "mean"):
        raise ValueError("statistic must be 'min' or 'mean'")

    try:
        scan = await _get_one_scan()
    except Exception as e:
        return {
            "status": "no_scan",
            "angle_deg": float(angle_deg),
            "half_width_deg": half_width_f,
            "statistic": statistic_clean,
            "distance_m": None,
            "error": str(e),
        }

    stats = _compute_sector_stats(
        scan,
        center_deg=float(angle_deg),
        half_width_deg=half_width_f,
    )
    if stats["status"] != "ok":
        return {
            "status": stats["status"],
            "angle_deg": float(angle_deg),
            "half_width_deg": half_width_f,
            "statistic": statistic_clean,
            "distance_m": None,
            "valid_count": stats["valid_count"],
            "total_count": stats["total_count"],
            "closest_angle_deg": stats["closest_angle_deg"],
        }

    distance_m = float(stats["min_m"]) if statistic_clean == "min" else float(stats["mean_m"])
    return {
        "status": "ok",
        "angle_deg": float(angle_deg),
        "half_width_deg": half_width_f,
        "statistic": statistic_clean,
        "distance_m": distance_m,
        "min_m": float(stats["min_m"]),
        "mean_m": float(stats["mean_m"]),
        "valid_count": stats["valid_count"],
        "total_count": stats["total_count"],
        "closest_angle_deg": stats["closest_angle_deg"],
    }


@mcp_lidar_v3.tool()
async def tbot_lidar_check_collision(
    front_threshold_m: float = 0.1,
    side_threshold_m: float = 0.2,
    back_threshold_m: float = 0.1,
) -> dict[str, Any]:
    """
    Check for collision risk.

    risk_level:
      "stop"    — front distance < front_threshold_m (default 0.1 m / 10 cm)
      "caution" — front distance between front_threshold_m and 2 × front_threshold_m
      "clear"   — front distance >= 2 × front_threshold_m

    Returns {risk_level, min_forward_distance_m, distances: {front, left, right, rear}}.

    NOTE: This tool is intended for pre-move assessment and stationary checks.
    """
    try:
        scan = await _get_one_scan()
    except Exception as e:
        return {
            "risk_level": "stop",
            "min_forward_distance_m": None,
            "distances": {"front": None, "left": None, "right": None, "rear": None},
            "error": str(e),
        }

    distances: dict[str, float | None] = {}
    for name, center_deg in _SECTOR_CENTERS.items():
        # Use a narrow 20° half-width for front to avoid picking up side obstacles;
        # wider 45° for side/rear sectors (informational only, not used for risk_level).
        half_width = 20.0 if name == "front" else 45.0
        stats = _compute_sector_stats(scan, center_deg=center_deg, half_width_deg=half_width)
        distances[name] = float(stats["min_m"]) if stats["status"] == "ok" and stats["min_m"] is not None else None

    front_dist = distances.get("front")
    if front_dist is None:
        risk_level = "stop"
    elif front_dist < front_threshold_m:
        risk_level = "stop"
    elif front_dist < 2.0 * front_threshold_m:
        risk_level = "caution"
    else:
        risk_level = "clear"

    return {"risk_level": risk_level, "min_forward_distance_m": distances.get("front"), "distances": distances}


@mcp_lidar_v3.tool()
async def tbot_lidar_find_clear_path(
    min_gap_width_m: float = 0.5,
    clearance_threshold_m: float = 1.0,
) -> dict[str, Any]:
    """
    Find navigable gaps in the current LiDAR scan.

    min_gap_width_m: only report gaps at least this wide (arc-length approximation).
    clearance_threshold_m: a range reading must be >= this value to be counted as clear.

    Returns {"gaps": [...], "best_gap": {...}|null} where each gap has
    {"heading_degrees": float, "width_m": float, "center_distance_m": float}.
    Heading convention matches _SECTOR_CENTERS: 0° = forward, 90° = left, -90° = right.
    """
    try:
        scan = await _get_one_scan()
    except Exception as e:
        return {"gaps": [], "best_gap": None, "error": str(e)}

    ranges = scan["ranges"]
    range_min = float(scan["range_min"])
    range_max = float(scan["range_max"])
    angle_increment_rad = float(scan["angle_increment"])
    n = len(ranges)

    # Mark each index as clear or occupied
    clear = [
        _is_valid_range(ranges[i], range_min, range_max) and ranges[i] >= clearance_threshold_m
        for i in range(n)
    ]

    # Group consecutive clear indices into gap runs
    runs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    for i in range(n):
        if clear[i]:
            if not in_run:
                run_start = i
                in_run = True
        else:
            if in_run:
                runs.append((run_start, i - 1))
                in_run = False
    if in_run:
        runs.append((run_start, n - 1))

    # Compute gap fields for each run
    gaps: list[dict[str, Any]] = []
    for start_idx, end_idx in runs:
        indices = range(start_idx, end_idx + 1)
        angles_deg = [_angle_for_index_deg(scan, i) for i in indices]
        clear_ranges = [float(ranges[i]) for i in indices]
        center_distance_m = sum(clear_ranges) / len(clear_ranges)
        heading_degrees = (angles_deg[0] + angles_deg[-1]) / 2.0
        angle_span_rad = len(clear_ranges) * angle_increment_rad
        width_m = angle_span_rad * center_distance_m

        if width_m >= min_gap_width_m:
            gaps.append({
                "heading_degrees": heading_degrees,
                "width_m": width_m,
                "center_distance_m": center_distance_m,
            })

    gaps.sort(key=lambda g: g["width_m"], reverse=True)
    best_gap = gaps[0] if gaps else None
    return {"gaps": gaps, "best_gap": best_gap}


@mcp_lidar_v3.tool()
async def tbot_lidar_export_free_space_map(
    radius_m: float = 2.0,
    resolution_m: float = 0.1,
    samples: int = 1,
) -> dict[str, Any]:
    """
    Export a local robot-centric free-space occupancy grid.

    Grid encoding:
      -1 = unknown
       0 = free
     100 = occupied
    """
    if not isinstance(radius_m, (int, float)) or not math.isfinite(float(radius_m)):
        raise ValueError("radius_m must be a finite number")
    if not isinstance(resolution_m, (int, float)) or not math.isfinite(float(resolution_m)):
        raise ValueError("resolution_m must be a finite number")
    if not isinstance(samples, int):
        raise ValueError("samples must be an integer")

    radius_f = float(radius_m)
    resolution_f = float(resolution_m)
    if radius_f <= 0:
        raise ValueError("radius_m must be > 0")
    if resolution_f <= 0:
        raise ValueError("resolution_m must be > 0")
    if samples < 1:
        raise ValueError("samples must be >= 1")

    scans: list[dict[str, Any]] = []
    errors: list[str] = []
    for _ in range(samples):
        try:
            scans.append(await _get_one_scan())
        except Exception as e:
            errors.append(str(e))
            if not scans:
                return {
                    "status": "no_scan",
                    "grid": [],
                    "best_effort_samples": 0,
                    "error": str(e),
                }
            break

    base_scan = scans[0]
    ranges = list(base_scan["ranges"])
    range_min = float(base_scan["range_min"])
    range_max = float(base_scan["range_max"])

    # Merge multiple scans conservatively by taking nearest valid range per ray.
    if len(scans) > 1:
        merged_ranges: list[float] = []
        for i in range(len(ranges)):
            candidates: list[float] = []
            for scan in scans:
                value = scan["ranges"][i]
                if _is_valid_range(value, range_min, range_max):
                    candidates.append(float(value))
            merged_ranges.append(min(candidates) if candidates else float("inf"))
        ranges = merged_ranges

    grid_size = int(math.ceil((2.0 * radius_f) / resolution_f)) + 1
    grid = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]

    for i, value in enumerate(ranges):
        angle_rad = float(base_scan["angle_min"]) + i * float(base_scan["angle_increment"])
        valid = _is_valid_range(value, range_min, range_max)
        ray_distance = min(float(value), radius_f) if valid else radius_f

        _mark_ray_free_cells(grid, angle_rad=angle_rad, distance_m=ray_distance, resolution_m=resolution_f)

        if valid and float(value) <= radius_f:
            obstacle_x = float(value) * math.cos(angle_rad)
            obstacle_y = float(value) * math.sin(angle_rad)
            obstacle_idx = _grid_index_from_xy(
                obstacle_x,
                obstacle_y,
                resolution_f,
                grid_size,
                grid_size,
            )
            if obstacle_idx is not None:
                gx, gy = obstacle_idx
                grid[gy][gx] = 100

    free_cells = sum(1 for row in grid for cell in row if cell == 0)
    occupied_cells = sum(1 for row in grid for cell in row if cell == 100)
    known_cells = free_cells + occupied_cells
    free_space_ratio = (free_cells / known_cells) if known_cells > 0 else 0.0

    response: dict[str, Any] = {
        "status": "ok",
        "frame_id": base_scan.get("frame_id"),
        "radius_m": radius_f,
        "resolution_m": resolution_f,
        "width": grid_size,
        "height": grid_size,
        "origin_cell": {"x": grid_size // 2, "y": grid_size // 2},
        "grid": grid,
        "free_cells": free_cells,
        "occupied_cells": occupied_cells,
        "unknown_cells": (grid_size * grid_size) - known_cells,
        "free_space_ratio": free_space_ratio,
        "best_effort_samples": len(scans),
    }
    if errors:
        response["errors"] = errors
    return response


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18212,
    path: str = "/turtlebot-lidar-v3",
    options: dict = {},
) -> None:
    """Run the TurtleBot LiDAR MCP Server V3."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting TurtleBot LiDAR MCP V3 topic=%s prefix=%s at %s:%s%s",
        LIDAR_TOPIC,
        NODE_NAME_PREFIX,
        host,
        port,
        path,
    )
    mcp_lidar_v3.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
