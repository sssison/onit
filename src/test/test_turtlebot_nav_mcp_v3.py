"""Tests for src/mcp/turtlebot_v3/nav_mcp_server.py."""

import asyncio
import os
import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import nav_mcp_server as nav_v3


class _FakeNavClient:
    def __init__(self, url: str, state: dict[str, Any]):
        self._url = url
        self._state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call_tool(self, name: str, payload: dict[str, Any]):
        if self._url == nav_v3.MOTION_MCP_URL_V3:
            self._state["motion_calls"].append((name, payload))
            if name in ("tbot_motion_move_timed", "tbot_motion_stop"):
                return {"status": "ok"}
            raise AssertionError(f"Unexpected motion tool call: {name}")

        if self._url == nav_v3.LIDAR_MCP_URL_V3:
            self._state["lidar_calls"].append((name, payload))
            if name == "tbot_lidar_check_collision":
                if self._state["collision_results"]:
                    return self._state["collision_results"].pop(0)
                return {
                    "risk_level": "clear",
                    "min_forward_distance_m": 2.0,
                    "distances": {"front": 2.0, "left": 2.0, "right": 2.0, "rear": 2.0},
                }
            raise AssertionError(f"Unexpected lidar tool call: {name}")

        raise AssertionError(f"Unexpected MCP URL in test: {self._url}")


def _make_state() -> dict[str, Any]:
    return {
        "motion_calls": [],
        "lidar_calls": [],
        "collision_results": [],
    }


def _odom(x_m: float, y_m: float, yaw_rad: float = 0.0) -> dict[str, Any]:
    return {
        "frame_id": "odom",
        "child_frame_id": "base_link",
        "x_m": x_m,
        "y_m": y_m,
        "yaw_rad": yaw_rad,
        "linear_mps": 0.0,
        "angular_rps": 0.0,
        "covariance_pose": [0.0] * 36,
        "covariance_twist": [0.0] * 36,
        "stamp_sec": 1.0,
        "age_mono_s": 0.01,
    }


def test_nav_get_pose_returns_no_odom_when_rclpy_unavailable():
    with patch.object(nav_v3, "RCLPY_AVAILABLE", False), patch.object(nav_v3, "RCLPY_IMPORT_ERROR", RuntimeError("missing")):
        result = asyncio.run(nav_v3.tbot_nav_get_pose())
    assert result["status"] == "no_odom"


def test_nav_get_pose_returns_pose():
    with patch.object(nav_v3, "RCLPY_AVAILABLE", True), patch.object(
        nav_v3,
        "_get_one_odom",
        new=AsyncMock(return_value=_odom(1.2, -0.4, 0.3)),
    ):
        result = asyncio.run(nav_v3.tbot_nav_get_pose())
    assert result["status"] == "ok"
    assert result["x_m"] == pytest.approx(1.2)
    assert result["y_m"] == pytest.approx(-0.4)
    assert result["yaw_rad"] == pytest.approx(0.3)


def test_nav_go_to_pose_reaches_target():
    state = _make_state()

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    odom_sequence = [
        _odom(0.0, 0.0, 0.0),
        _odom(0.95, 0.0, 0.0),
        _odom(1.0, 0.0, 0.0),
    ]

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "_get_one_odom",
        new=AsyncMock(side_effect=odom_sequence),
    ):
        result = asyncio.run(
            nav_v3.tbot_nav_go_to_pose(
                target_x_m=1.0,
                target_y_m=0.0,
                pos_tolerance_m=0.1,
                timeout_s=5.0,
            )
        )

    assert result["status"] == "reached"
    assert result["distance_remaining_m"] <= 0.1
    assert any(name == "tbot_motion_move_timed" for name, _ in state["motion_calls"])


def test_nav_go_to_pose_collision_blocked():
    state = _make_state()
    state["collision_results"] = [
        {"risk_level": "stop", "min_forward_distance_m": 0.08, "distances": {"front": 0.08}},
    ]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "_get_one_odom",
        new=AsyncMock(return_value=_odom(0.0, 0.0, 0.0)),
    ):
        result = asyncio.run(
            nav_v3.tbot_nav_go_to_pose(
                target_x_m=1.0,
                target_y_m=0.0,
                timeout_s=3.0,
            )
        )

    assert result["status"] == "collision_blocked"
    assert any(name == "tbot_motion_stop" for name, _ in state["motion_calls"])


def test_nav_patrol_returns_partial_completed_on_failure():
    with patch.object(
        nav_v3,
        "tbot_nav_go_to_pose",
        new=AsyncMock(
            side_effect=[
                {"status": "reached"},
                {"status": "collision_blocked"},
            ]
        ),
    ):
        result = asyncio.run(
            nav_v3.tbot_nav_patrol(
                waypoints=[{"x_m": 0.5, "y_m": 0.0}, {"x_m": 1.0, "y_m": 0.0}],
                cycles=1,
                timeout_s_total=30.0,
            )
        )

    assert result["status"] == "partial_completed"
    assert result["completed_waypoints"] == 1
