"""Targeted tests for TurtleBot v3 obstacle bypass motion logic."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import motion_mcp_server as motion_v3


class _FakeLidarClient:
    def __init__(self, _url: str, responses: list[dict]):
        self._responses = responses
        self._idx = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call_tool(self, name: str, _args: dict):
        if name != "tbot_lidar_get_obstacle_distances":
            raise AssertionError(f"Unexpected lidar tool call: {name}")
        if not self._responses:
            return {"distances": {"front": None, "left": None, "right": None, "rear": None}}
        if self._idx >= len(self._responses):
            return self._responses[-1]
        response = self._responses[self._idx]
        self._idx += 1
        return response


def test_pick_bypass_side_auto_uses_wider_side():
    assert motion_v3._pick_bypass_side("auto", 0.7, 0.4) == "left"
    assert motion_v3._pick_bypass_side("auto", 0.2, 0.9) == "right"


def test_compute_bypass_turn_angle_respects_min_and_max():
    angle_deg, raw_deg, shift_m = motion_v3._compute_bypass_turn_angle_deg(
        front_distance_m=0.4,
        side_distance_m=0.42,
        lateral_clearance_m=0.45,
        min_turn_angle_deg=15.0,
        max_turn_angle_deg=60.0,
    )
    assert raw_deg < 15.0
    assert angle_deg == pytest.approx(15.0)
    assert shift_m == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_bypass_returns_no_obstacle_when_front_is_already_clear():
    lidar_responses = [
        {"distances": {"front": 1.2, "left": 0.5, "right": 0.6, "rear": 0.7}},
    ]

    def _client_factory(url: str):
        return _FakeLidarClient(url, lidar_responses)

    turn_mock = AsyncMock(return_value={"status": "completed"})
    set_continuous_mock = AsyncMock(return_value=False)

    async def fake_post(path: str, payload=None):
        return {"status": "stopped", "path": path, "payload": payload}

    with patch.object(motion_v3, "Client", side_effect=_client_factory), patch.object(
        motion_v3, "tbot_motion_turn", new=turn_mock
    ), patch.object(motion_v3, "_set_continuous_motion", new=set_continuous_mock), patch.object(
        motion_v3, "_post_json", side_effect=fake_post
    ):
        result = await motion_v3.tbot_motion_bypass_obstacle(final_front_clearance_m=0.9)

    assert result["status"] == "no_obstacle"
    assert result["initial_distances"]["front"] == pytest.approx(1.2)
    assert turn_mock.await_count == 0


@pytest.mark.asyncio
async def test_bypass_completes_two_leg_maneuver():
    # 1 initial scan + first leg samples + second leg samples
    lidar_responses = [
        {"distances": {"front": 0.3, "left": 0.25, "right": 0.7, "rear": 0.8}},
        {"distances": {"front": 0.7, "left": 0.25, "right": 0.6, "rear": 0.8}},
        {"distances": {"front": 0.75, "left": 0.24, "right": 0.58, "rear": 0.8}},
        {"distances": {"front": 0.95, "left": 0.22, "right": 0.55, "rear": 0.8}},
        {"distances": {"front": 1.0, "left": 0.2, "right": 0.5, "rear": 0.8}},
        {"distances": {"front": 1.05, "left": 0.2, "right": 0.48, "rear": 0.8}},
    ]

    def _client_factory(url: str):
        return _FakeLidarClient(url, lidar_responses)

    turn_mock = AsyncMock(return_value={"status": "completed"})
    set_continuous_mock = AsyncMock(return_value=False)

    async def fake_post(path: str, payload=None):
        return {"status": "ok", "path": path, "payload": payload}

    # Keep real sleeps in this test so elapsed monotonic time advances naturally.
    with patch.object(motion_v3, "Client", side_effect=_client_factory), patch.object(
        motion_v3, "tbot_motion_turn", new=turn_mock
    ), patch.object(motion_v3, "_set_continuous_motion", new=set_continuous_mock), patch.object(
        motion_v3, "_post_json", side_effect=fake_post
    ):
        result = await motion_v3.tbot_motion_bypass_obstacle(
            preferred_side="auto",
            speed=0.1,
            turn_speed=0.4,
            parallel_front_clearance_m=0.6,
            final_front_clearance_m=0.9,
            max_first_leg_s=2.0,
            max_second_leg_s=2.5,
        )

    assert result["status"] == "completed"
    assert result["chosen_side"] == "right"
    assert result["first_leg_status"] == "parallel_reached"
    assert result["second_leg_status"] == "clear_path_reached"
    assert result["turn_angle_deg"] >= 15.0
    assert turn_mock.await_count == 2
