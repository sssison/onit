"""Tests for src/mcp/turtlebot_v2/lidar_mcp_server.py."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v2 import lidar_mcp_server as lidar_v2


def _make_scan() -> dict:
    num_points = 181
    angle_min = -math.pi / 2.0
    angle_max = math.pi / 2.0
    angle_increment = (angle_max - angle_min) / float(num_points - 1)
    return {
        "topic": "/scan",
        "frame_id": "base_scan",
        "ranges": [float("inf")] * num_points,
        "intensities": [0.0] * num_points,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "angle_increment": angle_increment,
        "range_min": 0.12,
        "range_max": 3.5,
        "stamp_s": 0.0,
        "received_unix_s": 0.0,
        "age_s": 0.1,
        "scan_count": 1,
    }


def _index_from_deg(scan: dict, angle_deg: float) -> int:
    angle_rad = math.radians(angle_deg)
    index = int(round((angle_rad - float(scan["angle_min"])) / float(scan["angle_increment"])))
    return max(0, min(len(scan["ranges"]) - 1, index))


def test_sector_valid_ranges_filters_invalid_values():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, -1.0)] = float("nan")
    scan["ranges"][_index_from_deg(scan, 0.0)] = 0.05
    scan["ranges"][_index_from_deg(scan, 1.0)] = 4.2
    scan["ranges"][_index_from_deg(scan, 2.0)] = 0.8

    extracted = lidar_v2._sector_valid_ranges(scan, center_deg=0.0, half_width_deg=3.0)
    assert extracted["status"] == "ok"
    assert extracted["valid_count"] == 1
    assert extracted["valid_values"] == [0.8]


def test_distance_at_angle_returns_no_data_when_invalid():
    scan = _make_scan()
    result = lidar_v2._distance_at_angle_from_scan(scan, angle_deg=0.0, window_deg=2.0, method="min")
    assert result["status"] == "no_data"
    assert result["distance_m"] is None


def test_collision_returns_stop_when_front_too_close():
    scan = _make_scan()
    for index in range(len(scan["ranges"])):
        scan["ranges"][index] = 1.0
    for angle in (-5.0, -2.5, 0.0, 2.5, 5.0):
        scan["ranges"][_index_from_deg(scan, angle)] = 0.15

    result = lidar_v2._collision_from_scan(
        scan=scan,
        front_threshold_m=0.25,
        side_threshold_m=0.20,
        back_threshold_m=0.25,
        sector_half_width_deg=20.0,
        max_scan_age_s=0.5,
    )

    assert result["risk_level"] == "stop"
    assert result["recommended_action"] == "stop"
    assert result["directions"]["front"]["collision"] is True


def test_find_clear_heading_selects_best_clearance():
    scan = _make_scan()
    for index in range(len(scan["ranges"])):
        scan["ranges"][index] = 0.5

    for angle in range(25, 36):
        scan["ranges"][_index_from_deg(scan, float(angle))] = 2.0

    result = lidar_v2._find_clear_heading_from_scan(
        scan=scan,
        search_min_deg=-60.0,
        search_max_deg=60.0,
        step_deg=10.0,
        sector_half_width_deg=5.0,
    )

    assert result["status"] == "ok"
    assert result["best_heading_deg"] == 30.0
    assert result["best_clearance_m"] == 2.0
