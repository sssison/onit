"""Tests for src/mcp/turtlebot_v3/lidar_mcp_server.py."""

import asyncio
import math
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import lidar_mcp_server as lidar_v3


def _make_scan() -> dict:
    """Create a 181-point scan spanning -90° to +90° with all ranges set to inf."""
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
        "received_mono_s": 0.0,
        "age_s": 0.05,
    }


def _index_from_deg(scan: dict, angle_deg: float) -> int:
    angle_rad = math.radians(angle_deg)
    index = int(round((angle_rad - float(scan["angle_min"])) / float(scan["angle_increment"])))
    return max(0, min(len(scan["ranges"]) - 1, index))


# --- Helper function unit tests ---


def test_sector_valid_ranges_filters_invalid():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, -1.0)] = float("nan")
    scan["ranges"][_index_from_deg(scan, 0.0)] = 0.05   # below range_min → invalid
    scan["ranges"][_index_from_deg(scan, 1.0)] = 4.2    # above range_max → invalid
    scan["ranges"][_index_from_deg(scan, 2.0)] = 0.8    # valid

    extracted = lidar_v3._sector_valid_ranges(scan, center_deg=0.0, half_width_deg=3.0)
    assert extracted["status"] == "ok"
    assert extracted["valid_count"] == 1
    assert extracted["valid_values"] == [0.8]


def test_compute_sector_stats_min_and_percentiles():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, -1.0)] = 1.0
    scan["ranges"][_index_from_deg(scan, 0.0)] = 2.0
    scan["ranges"][_index_from_deg(scan, 1.0)] = 3.0

    stats = lidar_v3._compute_sector_stats(scan, center_deg=0.0, half_width_deg=2.0)
    assert stats["status"] == "ok"
    assert stats["min_m"] == 1.0
    assert stats["p10_m"] == 1.0
    assert stats["p50_m"] == 2.0
    assert stats["p90_m"] == 2.0


def test_compute_sector_stats_no_data_when_all_inf():
    scan = _make_scan()
    stats = lidar_v3._compute_sector_stats(scan, center_deg=0.0, half_width_deg=45.0)
    assert stats["status"] == "no_data"
    assert stats["min_m"] is None


def test_percentiles_empty():
    result = lidar_v3._percentiles([])
    assert result == {"p10": None, "p50": None, "p90": None}


def test_percentiles_single_value():
    result = lidar_v3._percentiles([2.5])
    assert result["p10"] == pytest.approx(2.5)
    assert result["p50"] == pytest.approx(2.5)
    assert result["p90"] == pytest.approx(2.5)


# --- Tool unit tests (mocking _get_one_scan) ---


def test_lidar_health_online():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, 0.0)] = 1.5

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_health())

    assert result["status"] == "online"
    assert result["scan_present"] is True
    assert result["num_points"] == 181


def test_lidar_health_offline_on_error():
    async def fake_get_scan(timeout_s=3.0):
        raise RuntimeError("No scan received within 3s")

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_health())

    assert result["status"] == "offline"
    assert result["scan_present"] is False
    assert "error" in result


def test_lidar_get_obstacle_distances_front():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, 0.0)] = 1.2

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_get_obstacle_distances(sector="front"))

    assert result["status"] == "ok"
    assert result["sector"] == "front"
    assert result["distance_m"] == pytest.approx(1.2)
    assert "front" in result["distances"]


def test_lidar_get_obstacle_distances_all_returns_minimum():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, 0.0)] = 1.0   # front
    scan["ranges"][_index_from_deg(scan, 85.0)] = 0.4  # left

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_get_obstacle_distances(sector="all"))

    assert result["sector"] == "all"
    # minimum should be the smaller of the two valid distances
    assert result["distance_m"] is not None
    assert result["distance_m"] <= 1.0


def test_lidar_get_obstacle_distances_invalid_sector():
    with pytest.raises(ValueError, match="sector must be one of"):
        asyncio.run(lidar_v3.tbot_lidar_get_obstacle_distances(sector="diagonal"))


def test_lidar_is_path_clear_true():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, 0.0)] = 2.0

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_is_path_clear(threshold_m=0.5))

    assert result["clear"] is True
    assert result["min_forward_distance"] == pytest.approx(2.0)


def test_lidar_is_path_clear_false_when_obstacle_close():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, 0.0)] = 0.2

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_is_path_clear(threshold_m=0.5))

    assert result["clear"] is False
    assert result["min_forward_distance"] == pytest.approx(0.2)


def test_lidar_check_collision_stop():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, 0.0)] = 0.2  # valid (>= range_min=0.12), below front_threshold=0.3

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_check_collision(front_threshold_m=0.3))

    assert result["risk_level"] == "stop"
    assert "distances" in result
    assert result["distances"]["front"] == pytest.approx(0.2)


def test_lidar_check_collision_caution():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, 0.0)] = 0.45  # between 0.3 and 0.6

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_check_collision(front_threshold_m=0.3))

    assert result["risk_level"] == "caution"


def test_lidar_check_collision_clear():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, 0.0)] = 1.5  # well above 2 × 0.3

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_check_collision(front_threshold_m=0.3))

    assert result["risk_level"] == "clear"


def test_lidar_check_collision_stop_on_scan_error():
    async def fake_get_scan(timeout_s=3.0):
        raise RuntimeError("ROS unavailable")

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_check_collision())

    assert result["risk_level"] == "stop"
    assert result["distances"]["front"] is None


def test_lidar_check_collision_stop_when_no_front_data():
    scan = _make_scan()
    # all ranges remain inf → no valid front readings

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_check_collision(front_threshold_m=0.3))

    assert result["risk_level"] == "stop"
    assert result["distances"]["front"] is None
