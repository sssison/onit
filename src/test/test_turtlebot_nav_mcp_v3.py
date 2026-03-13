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
            if name == "tbot_vision_get_object_bbox":
                if self._state["vision_bbox_results"]:
                    return self._state["vision_bbox_results"].pop(0)
                return {
                    "visible": True,
                    "confidence": 0.9,
                    "position": "center",
                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                    "success": True,
                }
            if name == "tbot_vision_describe_scene":
                if self._state["describe_results"]:
                    return self._state["describe_results"].pop(0)
                return {"description": "{\"visible\": true, \"confidence\": 0.9}"}
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
        "vision_bbox_results": [],
        "describe_results": [],
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


def _reset_spatial_map_state() -> None:
    with nav_v3._spatial_map_lock:
        nav_v3._spatial_map_state["anchor_pose"] = None
        nav_v3._spatial_map_state["objects"] = []
        nav_v3._spatial_map_state["last_scan_pose"] = None
        nav_v3._spatial_map_state["last_scan_time_s"] = None
        nav_v3._spatial_map_state["scan_count"] = 0


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
    state["describe_results"] = [
        {"description": "{\"visible\": false, \"confidence\": 0.2}"},
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
    state["describe_results"] = [
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": false, \"confidence\": 0.1}"},
        {"description": "{\"visible\": false, \"confidence\": 0.1}"},
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


def test_navigate_to_object_recenters_when_visible_off_center_after_move():
    state = _make_state()
    state["vision_bbox_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.82, "cy": 0.5, "w": 0.2, "h": 0.2}},
    ]
    state["describe_results"] = [
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "Target object confirmed in scene."},
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
    assert result["stopped_reason"] == "reached_target"
    assert any(name == "tbot_motion_turn" for name, _ in state["motion_calls"])
    assert result["stopped_reason"] != "target_lost"


def test_navigate_to_object_does_not_declare_loss_on_single_miss():
    state = _make_state()
    state["describe_results"] = [
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": false, \"confidence\": 0.1}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "Target object confirmed in scene."},
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
    assert result["stopped_reason"] == "reached_target"


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


def test_nav_go_to_pose_collision_check_uses_forward_cone_half_width():
    state = _make_state()
    state["collision_results"] = [
        {"risk_level": "clear", "min_forward_distance_m": 1.5, "distances": {"front": 1.5}},
    ]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    odom_sequence = [
        _odom(0.0, 0.0, 0.0),
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
    lidar_check_calls = [payload for name, payload in state["lidar_calls"] if name == "tbot_lidar_check_collision"]
    assert lidar_check_calls
    assert lidar_check_calls[0]["sector_half_width_deg"] == pytest.approx(nav_v3.NAV_FORWARD_CONE_HALF_WIDTH_DEG)


def test_nav_go_to_pose_reduces_step_distance_when_forward_clearance_is_tight():
    state = _make_state()
    state["collision_results"] = [
        {"risk_level": "clear", "min_forward_distance_m": 0.22, "distances": {"front": 0.22}},
    ]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    odom_sequence = [
        _odom(0.0, 0.0, 0.0),
        _odom(1.0, 0.0, 0.0),
    ]

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "_get_one_odom",
        new=AsyncMock(side_effect=odom_sequence),
    ), patch.object(
        nav_v3,
        "NAV_FORWARD_STEP_M",
        0.20,
    ), patch.object(
        nav_v3,
        "NAV_FRONT_THRESHOLD_M",
        0.10,
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
    forward_calls = [payload for name, payload in state["motion_calls"] if name == "tbot_motion_move_forward_distance"]
    assert forward_calls
    assert forward_calls[0]["distance_m"] == pytest.approx(0.12)


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
    state["describe_results"] = [
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "Target object confirmed in scene."},
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


def test_navigate_to_object_uses_describe_scene_and_bbox_checks():
    state = _make_state()
    state["describe_results"] = [
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "Target object confirmed in scene."},
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
    find_calls = [payload for name, payload in state["vision_calls"] if name == "tbot_vision_find_object"]
    describe_calls = [payload for name, payload in state["vision_calls"] if name == "tbot_vision_describe_scene"]
    bbox_calls = [payload for name, payload in state["vision_calls"] if name == "tbot_vision_get_object_bbox"]
    assert len(find_calls) == 0
    assert len(describe_calls) >= 2
    assert len(bbox_calls) >= 1


def test_navigate_to_object_collision_stop_when_bypass_fails():
    state = _make_state()
    state["describe_results"] = [
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": false, \"confidence\": 0.1}"},
    ]
    state["vision_bbox_results"] = [
        {"visible": False, "confidence": 0.0, "bbox": None, "success": False},
        {"visible": False, "confidence": 0.0, "bbox": None, "success": False},
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
    assert any(name == "tbot_motion_bypass_obstacle" for name, _ in state["motion_calls"])


def test_navigate_to_object_stop_risk_with_visible_target_stops_without_bypass():
    state = _make_state()
    state["describe_results"] = [
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "Target object confirmed in scene."},
    ]
    state["collision_results"] = [{"risk_level": "stop", "min_forward_distance_m": 0.1, "distances": {"front": 0.1}}]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(nav_v3.tbot_navigate_to_object(target="trashcan", stop_distance=0.2))

    assert result["success"] is True
    assert result["stopped_reason"] == "reached_target"
    assert all(name != "tbot_motion_bypass_obstacle" for name, _ in state["motion_calls"])
    assert all(name != "tbot_motion_move_forward_distance" for name, _ in state["motion_calls"])


def test_navigate_to_object_stop_risk_with_visible_target_not_close_returns_blocked():
    state = _make_state()
    state["describe_results"] = [
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
        {"description": "{\"visible\": true, \"confidence\": 0.9}"},
    ]
    state["collision_results"] = [{"risk_level": "stop", "min_forward_distance_m": 0.1, "distances": {"front": 0.1}}]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(nav_v3.tbot_navigate_to_object(target="trashcan", stop_distance=0.05))

    assert result["success"] is False
    assert result["stopped_reason"] == "target_blocked_visible"
    assert all(name != "tbot_motion_bypass_obstacle" for name, _ in state["motion_calls"])


def test_go_to_midpoint_between_objects_clear_path_reaches_midpoint():
    state = _make_state()
    state["angle_distance_results"] = [
        {"status": "ok", "distance_m": 1.0, "valid_count": 3},
        {"status": "ok", "distance_m": 1.1, "valid_count": 3},
        {"status": "ok", "distance_m": 1.2, "valid_count": 3},
    ]
    pose_a = {"success": True, "x": 0.4, "y": 0.0, "heading_deg": 0.0, "distance_m": 0.4, "confidence": "high"}
    pose_b = {"success": True, "x": 0.8, "y": 0.0, "heading_deg": 0.0, "distance_m": 0.8, "confidence": "high"}

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    nav_go_mock = AsyncMock(return_value={"status": "reached"})
    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_estimate_object_pose",
        new=AsyncMock(side_effect=[pose_a, pose_b]),
    ), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ), patch.object(
        nav_v3,
        "tbot_nav_go_to_pose",
        new=nav_go_mock,
    ):
        result = asyncio.run(
            nav_v3.tbot_nav_go_to_midpoint_between_objects(
                object_1="chair",
                object_2="desk",
            )
        )

    assert result["success"] is True
    assert result["used_bypass"] is False
    assert result["midpoint_safety"]["classification"] == "clear"
    assert result["midpoint"]["x_m"] == pytest.approx(0.6)
    assert result["midpoint"]["y_m"] == pytest.approx(0.0)
    assert nav_go_mock.await_count == 1
    assert all(name != "tbot_motion_bypass_obstacle" for name, _ in state["motion_calls"])


def test_go_to_midpoint_between_objects_unsafe_path_bypass_then_reach():
    state = _make_state()
    state["angle_distance_results"] = [
        {"status": "ok", "distance_m": 0.2, "valid_count": 3},
        {"status": "ok", "distance_m": 0.25, "valid_count": 3},
        {"status": "ok", "distance_m": 0.3, "valid_count": 3},
    ]
    pose_a = {"success": True, "x": 0.4, "y": 0.0, "heading_deg": 0.0, "distance_m": 0.4, "confidence": "high"}
    pose_b = {"success": True, "x": 0.8, "y": 0.0, "heading_deg": 0.0, "distance_m": 0.8, "confidence": "high"}

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    nav_go_mock = AsyncMock(return_value={"status": "reached"})
    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_estimate_object_pose",
        new=AsyncMock(side_effect=[pose_a, pose_b]),
    ), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ), patch.object(
        nav_v3,
        "tbot_nav_go_to_pose",
        new=nav_go_mock,
    ):
        result = asyncio.run(
            nav_v3.tbot_nav_go_to_midpoint_between_objects(
                object_1="chair",
                object_2="desk",
            )
        )

    assert result["success"] is True
    assert result["used_bypass"] is True
    assert result["midpoint_safety"]["classification"] == "blocked"
    assert nav_go_mock.await_count == 1
    assert any(name == "tbot_motion_bypass_obstacle" for name, _ in state["motion_calls"])


def test_go_to_midpoint_between_objects_unsafe_path_bypass_fails():
    state = _make_state()
    state["angle_distance_results"] = [
        {"status": "ok", "distance_m": 0.2, "valid_count": 3},
        {"status": "ok", "distance_m": 0.25, "valid_count": 3},
        {"status": "ok", "distance_m": 0.3, "valid_count": 3},
    ]
    state["bypass_results"] = [{"status": "timeout"}]
    pose_a = {"success": True, "x": 0.4, "y": 0.0, "heading_deg": 0.0, "distance_m": 0.4, "confidence": "high"}
    pose_b = {"success": True, "x": 0.8, "y": 0.0, "heading_deg": 0.0, "distance_m": 0.8, "confidence": "high"}

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    nav_go_mock = AsyncMock(return_value={"status": "reached"})
    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_estimate_object_pose",
        new=AsyncMock(side_effect=[pose_a, pose_b]),
    ), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ), patch.object(
        nav_v3,
        "tbot_nav_go_to_pose",
        new=nav_go_mock,
    ):
        result = asyncio.run(
            nav_v3.tbot_nav_go_to_midpoint_between_objects(
                object_1="chair",
                object_2="desk",
            )
        )

    assert result["success"] is False
    assert result["error"] == "midpoint_unsafe_bypass_failed"
    assert result["midpoint_safety"]["classification"] == "blocked"
    assert nav_go_mock.await_count == 0
    assert any(name == "tbot_motion_bypass_obstacle" for name, _ in state["motion_calls"])


def test_go_to_midpoint_between_objects_collision_blocked_then_single_bypass_retry():
    state = _make_state()
    state["angle_distance_results"] = [
        {"status": "ok", "distance_m": 0.2, "valid_count": 3},
        {"status": "ok", "distance_m": 1.0, "valid_count": 3},
        {"status": "ok", "distance_m": 1.2, "valid_count": 3},
    ]
    pose_a = {"success": True, "x": 0.4, "y": 0.0, "heading_deg": 0.0, "distance_m": 0.4, "confidence": "high"}
    pose_b = {"success": True, "x": 0.8, "y": 0.0, "heading_deg": 0.0, "distance_m": 0.8, "confidence": "high"}

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    nav_go_mock = AsyncMock(side_effect=[{"status": "collision_blocked"}, {"status": "reached"}])
    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_estimate_object_pose",
        new=AsyncMock(side_effect=[pose_a, pose_b]),
    ), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ), patch.object(
        nav_v3,
        "tbot_nav_go_to_pose",
        new=nav_go_mock,
    ):
        result = asyncio.run(
            nav_v3.tbot_nav_go_to_midpoint_between_objects(
                object_1="chair",
                object_2="desk",
            )
        )

    assert result["success"] is True
    assert result["used_bypass"] is True
    assert result["midpoint_safety"]["classification"] == "inconclusive"
    assert nav_go_mock.await_count == 2
    assert any(name == "tbot_motion_bypass_obstacle" for name, _ in state["motion_calls"])


def test_go_to_midpoint_between_objects_fails_when_object_estimation_fails():
    pose_fail = {
        "success": False,
        "x": None,
        "y": None,
        "heading_deg": None,
        "distance_m": None,
        "confidence": "low",
        "error": "target_not_found",
    }
    with patch.object(
        nav_v3,
        "tbot_estimate_object_pose",
        new=AsyncMock(side_effect=[pose_fail]),
    ):
        result = asyncio.run(
            nav_v3.tbot_nav_go_to_midpoint_between_objects(
                object_1="chair",
                object_2="desk",
            )
        )

    assert result["success"] is False
    assert result["error"] == "object_1_not_found_or_unresolved"


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


def test_scan_spatial_map_collects_visible_objects():
    _reset_spatial_map_state()
    state = _make_state()
    state["vision_bbox_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": False, "confidence": 0.0, "bbox": None},
    ]
    state["angle_distance_results"] = [
        {"status": "ok", "distance_m": 1.0, "valid_count": 4},
        {"status": "ok", "distance_m": 1.5, "valid_count": 4},
    ]

    def fake_client(url: str):
        return _FakeNavClient(url, state)

    with patch.object(nav_v3, "Client", side_effect=fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        result = asyncio.run(
            nav_v3.tbot_scan_spatial_map(
                labels=["chair"],
                step_deg=180.0,
                turn_speed=0.5,
                turn_duration_s=0.56,
                lidar_half_width_deg=5.0,
            )
        )

    assert result["success"] is True
    assert result["steps_requested"] == 2
    assert result["steps_completed"] == 2
    assert len(result["objects"]) == 1
    assert result["objects"][0]["label"] == "chair"
    assert any(name == "tbot_motion_turn" for name, _ in state["motion_calls"])
    assert any(name == "tbot_lidar_get_distance_at_angle" for name, _ in state["lidar_calls"])
    assert any(name == "tbot_vision_get_object_bbox" for name, _ in state["vision_calls"])


def test_get_spatial_map_is_empty_before_scan():
    _reset_spatial_map_state()
    result = asyncio.run(nav_v3.tbot_get_spatial_map())
    assert result["status"] == "empty"
    assert result["objects"] == []


def test_merge_spatial_detection_merges_nearby_and_splits_far_entries():
    objects: list[dict[str, Any]] = []
    nav_v3._merge_spatial_detection(objects, label="chair", x_m=0.0, y_m=1.0, confidence=0.8)
    nav_v3._merge_spatial_detection(objects, label="chair", x_m=0.1, y_m=1.0, confidence=0.7)
    assert len(objects) == 1
    assert objects[0]["sighting_count"] == 2

    nav_v3._merge_spatial_detection(objects, label="chair", x_m=1.0, y_m=1.0, confidence=0.9)
    assert len(objects) == 2


def test_scan_spatial_map_resets_when_robot_moves_past_threshold():
    _reset_spatial_map_state()
    first_state = _make_state()
    first_state["vision_bbox_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": False, "confidence": 0.0, "bbox": None},
    ]
    first_state["angle_distance_results"] = [
        {"status": "ok", "distance_m": 1.0, "valid_count": 4},
        {"status": "ok", "distance_m": 1.0, "valid_count": 4},
    ]

    def first_fake_client(url: str):
        return _FakeNavClient(url, first_state)

    with patch.object(nav_v3, "Client", side_effect=first_fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        first_result = asyncio.run(nav_v3.tbot_scan_spatial_map(labels=["chair"], step_deg=180.0))
    assert first_result["success"] is True
    assert any(obj["label"] == "chair" for obj in first_result["objects"])

    second_state = _make_state()
    second_state["vision_bbox_results"] = [
        {"visible": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"visible": False, "confidence": 0.0, "bbox": None},
    ]
    second_state["angle_distance_results"] = [
        {"status": "ok", "distance_m": 1.0, "valid_count": 4},
        {"status": "ok", "distance_m": 1.0, "valid_count": 4},
    ]

    def second_fake_client(url: str):
        return _FakeNavClient(url, second_state)

    with patch.object(nav_v3, "Client", side_effect=second_fake_client), patch.object(
        nav_v3,
        "tbot_nav_get_pose",
        new=AsyncMock(return_value={"status": "ok", "x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0}),
    ):
        second_result = asyncio.run(
            nav_v3.tbot_scan_spatial_map(
                labels=["desk"],
                step_deg=180.0,
                reset_if_moved_m=0.5,
            )
        )

    assert second_result["success"] is True
    assert second_result["reset_performed"] is True
    assert second_result["anchor_pose"]["x"] == pytest.approx(1.0)
    labels = [obj["label"] for obj in second_result["objects"]]
    assert "desk" in labels
    assert "chair" not in labels


def test_bearing_distance_from_xy_cardinal_conventions():
    bearing_0, distance_0 = nav_v3._bearing_distance_from_xy(0.0, 1.0)
    bearing_90, _ = nav_v3._bearing_distance_from_xy(1.0, 0.0)
    bearing_180, _ = nav_v3._bearing_distance_from_xy(0.0, -1.0)
    bearing_270, _ = nav_v3._bearing_distance_from_xy(-1.0, 0.0)

    assert distance_0 == pytest.approx(1.0)
    assert bearing_0 == pytest.approx(0.0)
    assert bearing_90 == pytest.approx(90.0)
    assert bearing_180 == pytest.approx(180.0)
    assert bearing_270 == pytest.approx(270.0)
