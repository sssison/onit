"""Tests for src/mcp/turtlebot_v3/nav_mcp_server.py."""

import asyncio
import math
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
        self._state["call_order"].append((self._url, name))
        if self._url == nav_v3.MOTION_MCP_URL_V3:
            self._state["motion_calls"].append((name, payload))
            if name == "tbot_motion_bypass_obstacle":
                if self._state["bypass_results"]:
                    return self._state["bypass_results"].pop(0)
                return {"status": "completed"}
            if name in ("tbot_motion_turn", "tbot_motion_stop", "tbot_motion_move_forward_distance"):
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
            if name == "tbot_lidar_get_distance_at_angle":
                if self._state["angle_distance_results"]:
                    return self._state["angle_distance_results"].pop(0)
                return {"status": "ok", "distance_m": 1.0, "valid_count": 3}
            raise AssertionError(f"Unexpected lidar tool call: {name}")

        if self._url == nav_v3.VISION_MCP_URL_V3:
            self._state["vision_calls"].append((name, payload))
            if name == "tbot_vision_find_object":
                if self._state["vision_find_results"]:
                    return self._state["vision_find_results"].pop(0)
                return {
                    "visible": True,
                    "confidence": 0.9,
                    "position": "center",
                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                }
            if name == "tbot_vision_describe_scene":
                return {"description": "Target object confirmed in scene."}
            raise AssertionError(f"Unexpected vision tool call: {name}")

        raise AssertionError(f"Unexpected MCP URL in test: {self._url}")


def _make_state() -> dict[str, Any]:
    return {
        "call_order": [],
        "motion_calls": [],
        "lidar_calls": [],
        "vision_calls": [],
        "collision_results": [],
        "vision_find_results": [],
        "angle_distance_results": [],
        "bypass_results": [],
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


def test_navigate_to_object_requires_target_visible_at_start():
    state = _make_state()
    state["vision_find_results"] = [
        {"visible": False, "confidence": 0.1, "bbox": None},
    ]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(nav_v3.tbot_navigate_to_object(target="cabinet", stop_distance=0.6))

    assert result["success"] is False
    assert result["stopped_reason"] == "target_not_visible"
    assert all(name != "tbot_motion_move_forward_distance" for name, _ in state["motion_calls"])


def test_navigate_to_object_returns_target_lost_after_move():
    state = _make_state()
    state["vision_find_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": False, "confidence": 0.1, "bbox": None},
    ]
    state["collision_results"] = [
        {"risk_level": "clear", "min_forward_distance_m": 1.0, "distances": {"front": 1.0}},
    ]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.2, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(nav_v3.tbot_navigate_to_object(target="cabinet", stop_distance=0.6))

    assert result["success"] is False
    assert result["stopped_reason"] == "target_lost"
    assert any(name == "tbot_motion_move_forward_distance" for name, _ in state["motion_calls"])


def test_nav_go_to_pose_reaches_target_without_timed_motion():
    state = _make_state()

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    odom_sequence = [
        _odom(0.0, 0.0, 0.0),
        _odom(0.5, 0.0, 0.0),
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
    assert any(name == "tbot_motion_move_forward_distance" for name, _ in state["motion_calls"])
    assert all(name != "tbot_motion_move_timed" for name, _ in state["motion_calls"])


def test_nav_go_to_pose_turn_phase_does_not_query_lidar_until_forward_motion():
    state = _make_state()

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    odom_sequence = [
        _odom(0.0, 0.0, 0.0),
        _odom(0.0, 0.0, math.pi / 2.0),
        _odom(0.0, 0.95, math.pi / 2.0),
        _odom(0.0, 1.0, math.pi / 2.0),
    ]

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "_get_one_odom",
        new=AsyncMock(side_effect=odom_sequence),
    ):
        result = asyncio.run(
            nav_v3.tbot_nav_go_to_pose(
                target_x_m=0.0,
                target_y_m=1.0,
                pos_tolerance_m=0.1,
                timeout_s=5.0,
            )
        )

    assert result["status"] == "reached"
    first_turn_idx = next(i for i, (_, name) in enumerate(state["call_order"]) if name == "tbot_motion_turn")
    first_lidar_idx = next(i for i, (_, name) in enumerate(state["call_order"]) if name == "tbot_lidar_check_collision")
    assert first_turn_idx < first_lidar_idx


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


def test_estimate_object_pose_straight_ahead():
    state = _make_state()
    state["vision_find_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
    ]
    state["angle_distance_results"] = [{"status": "ok", "distance_m": 2.0, "valid_count": 4}]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 1.0, "y_m": 2.0, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(nav_v3.tbot_estimate_object_pose(target="door"))

    assert result["success"] is True
    assert result["heading_deg"] == pytest.approx(0.0)
    assert result["x"] == pytest.approx(3.0)
    assert result["y"] == pytest.approx(2.0)
    assert result["confidence"] == "high"


def test_estimate_object_pose_offset_heading_geometry():
    state = _make_state()
    state["vision_find_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 1.0, "cy": 0.5, "w": 0.2, "h": 0.2}},
    ]
    state["angle_distance_results"] = [{"status": "ok", "distance_m": math.sqrt(2.0), "valid_count": 3}]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "TBOT_CAMERA_FOV_DEG",
        90.0,
    ), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(nav_v3.tbot_estimate_object_pose(target="cone"))

    assert result["success"] is True
    assert result["heading_deg"] == pytest.approx(45.0)
    assert result["x"] == pytest.approx(1.0, abs=1e-3)
    assert result["y"] == pytest.approx(1.0, abs=1e-3)


def test_estimate_object_pose_lidar_occluded_returns_low_confidence():
    state = _make_state()
    state["vision_find_results"] = [
        {"visible": True, "confidence": 0.7, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.4}},
    ]
    state["angle_distance_results"] = [{"status": "no_data", "distance_m": None}]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(nav_v3.tbot_estimate_object_pose(target="chair"))

    assert result["success"] is True
    assert result["confidence"] == "low"
    assert isinstance(result["distance_m"], float)


def test_navigate_to_object_reaches_target_and_confirms_scene():
    state = _make_state()
    state["vision_find_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
    ]
    state["collision_results"] = [
        {"risk_level": "clear", "min_forward_distance_m": 1.0, "distances": {"front": 1.0}},
        {"risk_level": "clear", "min_forward_distance_m": 0.55, "distances": {"front": 0.55}},
    ]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.4, "y_m": 0.1, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(
            nav_v3.tbot_navigate_to_object(
                target="door",
                stop_distance=0.6,
                confirm_in_frame=True,
            )
        )

    assert result["success"] is True
    assert result["stopped_reason"] == "reached_target"
    assert result["object_in_frame"] is True
    assert "confirmed" in result["scene_description"].lower()
    assert any(name == "tbot_motion_move_forward_distance" for name, _ in state["motion_calls"])


def test_navigate_to_object_uses_frame_only_checks():
    state = _make_state()
    state["vision_find_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
    ]
    state["collision_results"] = [
        {"risk_level": "clear", "min_forward_distance_m": 1.0, "distances": {"front": 1.0}},
        {"risk_level": "clear", "min_forward_distance_m": 0.55, "distances": {"front": 0.55}},
    ]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.4, "y_m": 0.1, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(nav_v3.tbot_navigate_to_object(target="cabinet", stop_distance=0.6))

    assert result["success"] is True
    vision_payloads = [payload for name, payload in state["vision_calls"] if name == "tbot_vision_find_object"]
    assert len(vision_payloads) >= 2
    assert all(payload.get("search_mode") == "frame_only" for payload in vision_payloads)


def test_navigate_to_object_collision_stop_when_bypass_fails():
    state = _make_state()
    state["vision_find_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
    ]
    state["collision_results"] = [{"risk_level": "stop", "min_forward_distance_m": 0.1, "distances": {"front": 0.1}}]
    state["bypass_results"] = [{"status": "timeout"}]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(nav_v3.tbot_navigate_to_object(target="cabinet", stop_distance=0.6))

    assert result["success"] is False
    assert result["stopped_reason"] == "collision_stop"


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
