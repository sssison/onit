"""Tests for TurtleBot v3 LiDAR free-space map export."""

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


def test_lidar_export_free_space_map_marks_free_and_occupied_cells():
    scan = _make_scan()
    scan["ranges"][_index_from_deg(scan, 0.0)] = 0.8

    async def fake_get_scan(timeout_s=3.0):
        return scan

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(
            lidar_v3.tbot_lidar_export_free_space_map(
                radius_m=1.0,
                resolution_m=0.1,
                samples=1,
            )
        )

    assert result["status"] == "ok"
    assert result["width"] > 0
    assert result["height"] > 0
    assert result["occupied_cells"] >= 1
    assert result["free_cells"] >= 1
    assert result["free_space_ratio"] >= 0.0
    assert result["free_space_ratio"] <= 1.0


def test_lidar_export_free_space_map_returns_no_scan_on_failure():
    async def fake_get_scan(timeout_s=3.0):
        raise RuntimeError("no scan")

    with patch.object(lidar_v3, "_get_one_scan", side_effect=fake_get_scan):
        result = asyncio.run(lidar_v3.tbot_lidar_export_free_space_map())

    assert result["status"] == "no_scan"
    assert result["grid"] == []
