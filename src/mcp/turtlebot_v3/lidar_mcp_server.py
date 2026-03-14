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

# --- Persistent streaming subscriber state ---
_latest_scan: dict[str, Any] | None = None
_scan_version: int = 0
_scan_condition = threading.Condition()   # protects _latest_scan and _scan_version
_listener_node: Any = None                # _ScanListenerNode instance
_ros_thread: threading.Thread | None = None

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


def _ensure_non_negative(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


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
        "rear": (180.0, back_threshold_m),
    }

    directions: dict[str, Any] = {}
    distances: dict[str, float | None] = {}
    has_unknown = False
    has_collision = False
    has_caution = False

    for name, (center_deg, threshold) in sectors.items():
        stats = _compute_sector_stats(scan, center_deg=center_deg, half_width_deg=sector_half_width_deg)
        min_m: float | None = float(stats["min_m"]) if stats["status"] == "ok" and stats["min_m"] is not None else None
        distances[name] = min_m
        is_front = name == "front"

        collision_state: bool | str
        if age_s is None or (max_scan_age_s > 0 and float(age_s) > max_scan_age_s):
            collision_state = "unknown"
            if is_front:
                has_unknown = True
        elif min_m is None:
            collision_state = "unknown"
            if is_front:
                has_unknown = True
        elif min_m <= threshold:
            collision_state = True
            has_collision = True
        else:
            collision_state = False
            if min_m <= (threshold * 2.0):
                has_caution = True

        directions[name] = {
            "min_m": min_m,
            "threshold_m": threshold,
            "collision": collision_state,
            "closest_angle_deg": stats["closest_angle_deg"],
            "status": stats["status"],
        }

    # Front distance is mandatory for forward motion safety.
    if distances.get("front") is None:
        has_collision = True

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
        "distances": distances,
        "directions": directions,
        "scan_age_s": age_s,
        "frame_id": frame_id,
        "topic": topic,
        "min_forward_distance_m": distances.get("front"),
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


# --- Persistent background subscriber ---


class _ScanListenerNode(Node):  # type: ignore[misc]
    """Long-lived node that caches every incoming LaserScan."""

    def __init__(self) -> None:
        super().__init__(f"{NODE_NAME_PREFIX}_listener")
        self.create_subscription(
            LaserScan,
            LIDAR_TOPIC,
            self._callback,
            qos_profile_sensor_data,
        )

    def _callback(self, msg: Any) -> None:
        global _latest_scan, _scan_version
        now_mono = time.monotonic()
        stamp_s: float | None = None
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1_000_000_000.0
        scan_data: dict[str, Any] = {
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
            "age_s": 0.0,
        }
        with _scan_condition:
            _latest_scan = scan_data
            _scan_version += 1
            _scan_condition.notify_all()


def _ensure_listener_running() -> None:
    global _listener_node, _ros_thread
    if _ros_thread is not None and _ros_thread.is_alive():
        return
    if not RCLPY_AVAILABLE:
        raise RuntimeError(f"rclpy is not available: {RCLPY_IMPORT_ERROR}")
    with _rclpy_init_lock:
        if not rclpy.ok():
            rclpy.init()
    _listener_node = _ScanListenerNode()
    _ros_thread = threading.Thread(
        target=rclpy.spin, args=(_listener_node,), daemon=True, name="lidar_spin"
    )
    _ros_thread.start()


def _shutdown_lidar_ros() -> None:
    global _listener_node
    if _listener_node is not None:
        try:
            _listener_node.destroy_node()
        except Exception:
            pass
        _listener_node = None


import atexit
atexit.register(_shutdown_lidar_ros)


def _get_cached_scan_sync(timeout_s: float = SCAN_TIMEOUT_S) -> dict[str, Any]:
    """Return the latest cached scan; block until one arrives if cache is empty."""
    _ensure_listener_running()
    with _scan_condition:
        if _latest_scan is not None:
            scan = dict(_latest_scan)
            scan["age_s"] = max(0.0, time.monotonic() - scan["received_mono_s"])
            return scan
        arrived = _scan_condition.wait(timeout=timeout_s)
    if not arrived or _latest_scan is None:
        raise RuntimeError(f"No LiDAR scan received within {timeout_s}s on topic {LIDAR_TOPIC!r}")
    scan = dict(_latest_scan)
    scan["age_s"] = max(0.0, time.monotonic() - scan["received_mono_s"])
    return scan


def _get_next_scan_sync(current_version: int, timeout_s: float = SCAN_TIMEOUT_S) -> dict[str, Any]:
    """Block until a scan newer than current_version arrives; used by multi-sample export."""
    _ensure_listener_running()
    deadline = time.monotonic() + timeout_s
    with _scan_condition:
        while _scan_version <= current_version:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"Timed out waiting for new LiDAR scan after {timeout_s}s")
            _scan_condition.wait(timeout=min(0.1, remaining))
        scan = dict(_latest_scan)  # type: ignore[arg-type]
    scan["age_s"] = max(0.0, time.monotonic() - scan["received_mono_s"])
    return scan


async def _get_cached_scan(timeout_s: float = SCAN_TIMEOUT_S) -> dict[str, Any]:
    """Async wrapper for _get_cached_scan_sync."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_cached_scan_sync, timeout_s)


async def _get_next_scan(current_version: int, timeout_s: float = SCAN_TIMEOUT_S) -> dict[str, Any]:
    """Async wrapper for _get_next_scan_sync."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_next_scan_sync, current_version, timeout_s)


async def _get_one_scan(timeout_s: float = SCAN_TIMEOUT_S) -> dict[str, Any]:
    """Compatibility helper used by tests and higher-level tools."""
    return await _get_cached_scan(timeout_s)


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
async def tbot_lidar_check_collision(
    front_threshold_m: float = 0.25,
    side_threshold_m: float = 0.20,
    back_threshold_m: float = 0.25,
    sector_half_width_deg: float = 20.0,
    max_scan_age_s: float = 0.5,
) -> dict[str, Any]:
    """
    Check directional collision risk for short motion planning.
    Returns risk_level in: clear|caution|stop|unknown.
    Role: gate for all forward motion — always call via the COLLISION_GUARD sub-procedure.
    Do NOT call for pure rotations.
    """
    front_threshold = _ensure_non_negative("front_threshold_m", front_threshold_m)
    side_threshold = _ensure_non_negative("side_threshold_m", side_threshold_m)
    back_threshold = _ensure_non_negative("back_threshold_m", back_threshold_m)
    sector_half_width = _ensure_non_negative("sector_half_width_deg", sector_half_width_deg)
    max_scan_age = _ensure_non_negative("max_scan_age_s", max_scan_age_s)

    try:
        scan = await _get_one_scan()
    except Exception as e:
        return {
            "risk_level": "stop",
            "recommended_action": "stop",
            "distances": {"front": None, "left": None, "right": None, "rear": None},
            "directions": {},
            "scan_age_s": None,
            "frame_id": None,
            "topic": LIDAR_TOPIC,
            "min_forward_distance_m": None,
            "error": str(e),
        }

    return _collision_from_scan(
        scan=scan,
        front_threshold_m=front_threshold,
        side_threshold_m=side_threshold,
        back_threshold_m=back_threshold,
        sector_half_width_deg=sector_half_width,
        max_scan_age_s=max_scan_age,
    )


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
async def tbot_lidar_get_wall_profile(
    step_deg: float = 15.0,
) -> dict[str, Any]:
    """
    Return distance measurements sampled around the full 360° at fixed angular steps.

    Gives the agent a compact map of walls and open space in all directions.
    step_deg: angular spacing between samples (default 15° = 24 readings).
    Returns readings[], nearest obstacle, farthest open direction, and open_headings_deg (> 1.0 m).
    """
    if not isinstance(step_deg, (int, float)) or not math.isfinite(float(step_deg)):
        raise ValueError("step_deg must be a finite number")
    step_f = float(step_deg)
    if step_f <= 0 or step_f > 180:
        raise ValueError("step_deg must be in (0, 180]")

    try:
        scan = await _get_one_scan()
    except Exception as e:
        return {"status": "no_scan", "error": str(e), "readings": []}

    readings: list[dict[str, Any]] = []
    angle = -180.0
    while angle < 180.0:
        stats = _compute_sector_stats(scan, center_deg=angle, half_width_deg=step_f / 2.0)
        dist = float(stats["min_m"]) if stats["status"] == "ok" and stats["min_m"] is not None else None
        readings.append({"angle_deg": round(angle, 1), "distance_m": dist})
        angle += step_f

    valid = [(r["angle_deg"], r["distance_m"]) for r in readings if r["distance_m"] is not None]
    nearest = min(valid, key=lambda x: x[1]) if valid else None
    farthest = max(valid, key=lambda x: x[1]) if valid else None
    open_headings = [r["angle_deg"] for r in readings if r["distance_m"] is not None and r["distance_m"] > 1.0]

    return {
        "status": "ok",
        "step_deg": step_f,
        "readings": readings,
        "nearest": {"angle_deg": nearest[0], "distance_m": nearest[1]} if nearest else None,
        "farthest": {"angle_deg": farthest[0], "distance_m": farthest[1]} if farthest else None,
        "open_headings_deg": open_headings,
        "scan_age_s": scan.get("age_s"),
    }


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
    try:
        first = await _get_one_scan()
        scans.append(first)
    except Exception as e:
        return {
            "status": "no_scan",
            "grid": [],
            "best_effort_samples": 0,
            "error": str(e),
        }
    with _scan_condition:
        last_version = _scan_version
    for _ in range(samples - 1):
        try:
            scans.append(await _get_next_scan(last_version))
            with _scan_condition:
                last_version = _scan_version
        except Exception as e:
            errors.append(str(e))
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
