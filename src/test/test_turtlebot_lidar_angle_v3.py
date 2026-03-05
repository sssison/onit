"""Tests for angle-based LiDAR distance tool (v3)."""

import asyncio
import math
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import lidar_mcp_server as lidar_v3


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
        "received_mono_s": 0.0,
        "age_s": 0.05,
    }


def _index_from_deg(scan: dict, angle_deg: float) -> int:
    angle_rad = math.radians(angle_deg)
    idx = int(round((angle_rad - float(scan["angle_min"])) / float(scan["angle_increment"])))
    return max(0, min(len(scan["ranges"]) - 1, idx))


def test_lidar_distance_at_angle_returns_min_statistic():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, -2.0)] = 1.0
    scan["ranges"][_index_from_deg(scan, 0.0)] = 0.7
    scan["ranges"][_index_from_deg(scan, 2.0)] = 0.9

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(
            lidar_v3.tbot_lidar_get_distance_at_angle(
                angle_deg=0.0,
                half_width_deg=3.0,
                statistic="min",
            )
        )

    assert result["status"] == "ok"
    assert result["distance_m"] == pytest.approx(0.7)
    assert result["min_m"] == pytest.approx(0.7)


def test_lidar_distance_at_angle_returns_mean_statistic():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, -2.0)] = 1.0
    scan["ranges"][_index_from_deg(scan, 0.0)] = 0.7
    scan["ranges"][_index_from_deg(scan, 2.0)] = 0.9

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(
            lidar_v3.tbot_lidar_get_distance_at_angle(
                angle_deg=0.0,
                half_width_deg=3.0,
                statistic="mean",
            )
        )

    assert result["status"] == "ok"
    assert result["distance_m"] == pytest.approx((1.0 + 0.7 + 0.9) / 3.0)


def test_lidar_distance_at_angle_no_data():
    scan = _make_scan()

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(
            lidar_v3.tbot_lidar_get_distance_at_angle(
                angle_deg=0.0,
                half_width_deg=3.0,
            )
        )

    assert result["status"] == "no_data"
    assert result["distance_m"] is None


def test_lidar_distance_at_angle_invalid_statistic():
    with pytest.raises(ValueError, match="statistic must be 'min' or 'mean'"):
        asyncio.run(
            lidar_v3.tbot_lidar_get_distance_at_angle(
                angle_deg=0.0,
                half_width_deg=2.0,
                statistic="median",
            )
        )
