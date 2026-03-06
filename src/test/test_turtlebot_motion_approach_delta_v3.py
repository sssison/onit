"""Tests for precomputed-distance forward approach in TurtleBot v3 motion MCP."""

import asyncio
import os
import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import motion_mcp_server as motion_v3


class _FakeLidarClient:
    def __init__(self, _url: str, responses: list[dict[str, Any]]):
        self._responses = responses
        self._idx = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call_tool(self, name: str, _args: dict[str, Any]):
        if name != "tbot_lidar_check_collision":
            raise AssertionError(f"Unexpected lidar tool call: {name}")
        if not self._responses:
            return {"risk_level": "clear", "min_forward_distance_m": 1.0, "distances": {"front": 1.0}}
        if self._idx >= len(self._responses):
            return self._responses[-1]
        response = self._responses[self._idx]
        self._idx += 1
        return response


@pytest.mark.asyncio
async def test_move_forward_distance_computes_duration():
    async def fake_post(path: str, payload=None):
        return {"status": "ok", "path": path, "payload": payload}

    with patch.object(motion_v3, "_set_continuous_motion", new=AsyncMock(return_value=False)), patch.object(
        motion_v3, "_post_move_for_duration", new=AsyncMock(return_value=({"status": "updated"}, 7))
    ) as move_mock, patch.object(motion_v3, "_post_json", side_effect=fake_post):
        result = await motion_v3.tbot_motion_move_forward_distance(distance_m=0.6, speed=0.2)

    assert result["status"] == "completed"
    assert result["distance_m"] == pytest.approx(0.6)
    assert result["duration_s"] == pytest.approx(3.0)
    move_mock.assert_awaited_once_with(0.2, 0.0, pytest.approx(3.0))


@pytest.mark.asyncio
async def test_move_forward_distance_clamps_speed():
    async def fake_post(path: str, payload=None):
        return {"status": "ok", "path": path}

    with patch.object(motion_v3, "_set_continuous_motion", new=AsyncMock(return_value=False)), patch.object(
        motion_v3, "_post_move_for_duration", new=AsyncMock(return_value=({"status": "updated"}, 3))
    ) as move_mock, patch.object(motion_v3, "_post_json", side_effect=fake_post):
        result = await motion_v3.tbot_motion_move_forward_distance(distance_m=0.5, speed=999.0)

    assert result["speed"] == pytest.approx(motion_v3.MAX_LINEAR)
    expected_duration = 0.5 / motion_v3.MAX_LINEAR
    move_mock.assert_awaited_once_with(motion_v3.MAX_LINEAR, 0.0, pytest.approx(expected_duration))


@pytest.mark.asyncio
async def test_move_forward_distance_rejects_non_positive_distance():
    with pytest.raises(ValueError, match="distance_m must be > 0"):
        await motion_v3.tbot_motion_move_forward_distance(distance_m=0.0, speed=0.1)


@pytest.mark.asyncio
async def test_approach_plans_delta_then_moves_once():
    # front=1.0, target=0.4 => required move = 0.6
    lidar_responses = [
        {"risk_level": "clear", "min_forward_distance_m": 1.0, "distances": {"front": 1.0}},
        {"risk_level": "clear", "min_forward_distance_m": 0.39, "distances": {"front": 0.39}},
    ]

    def _client_factory(url: str):
        return _FakeLidarClient(url, lidar_responses)

    move_mock = AsyncMock(return_value={"status": "completed", "move_posts": 8})

    async def fake_post(path: str, payload=None):
        return {"status": "ok", "path": path}

    with patch.object(motion_v3, "Client", side_effect=_client_factory), patch.object(
        motion_v3, "_execute_forward_distance", new=move_mock
    ), patch.object(motion_v3, "_set_continuous_motion", new=AsyncMock(return_value=False)), patch.object(
        motion_v3, "_post_json", side_effect=fake_post
    ):
        result = await motion_v3.tbot_motion_approach_until_close(
            target_distance_m=0.4,
            stop_distance_m=0.1,
            speed=0.2,
            timeout_s=10.0,
        )

    assert result["status"] == "reached"
    assert result["initial_front_distance_m"] == pytest.approx(1.0)
    assert result["required_move_distance_m"] == pytest.approx(0.6)
    assert result["requested_move_duration_s"] == pytest.approx(3.0)
    assert result["executed_move_duration_s"] == pytest.approx(3.0)
    assert result["front_distance"] == pytest.approx(0.39)

    move_mock.assert_awaited_once()
    kwargs = move_mock.await_args.kwargs
    assert kwargs["distance_m"] == pytest.approx(0.6)
    assert kwargs["speed"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_approach_keeps_10cm_standoff_by_default_delta():
    # front=0.5, target=0.1 => required move = 0.4
    lidar_responses = [
        {"risk_level": "clear", "min_forward_distance_m": 0.5, "distances": {"front": 0.5}},
        {"risk_level": "clear", "min_forward_distance_m": 0.12, "distances": {"front": 0.12}},
    ]

    def _client_factory(url: str):
        return _FakeLidarClient(url, lidar_responses)

    move_mock = AsyncMock(return_value={"status": "completed", "move_posts": 5})

    async def fake_post(path: str, payload=None):
        return {"status": "ok", "path": path}

    with patch.object(motion_v3, "Client", side_effect=_client_factory), patch.object(
        motion_v3, "_execute_forward_distance", new=move_mock
    ), patch.object(motion_v3, "_set_continuous_motion", new=AsyncMock(return_value=False)), patch.object(
        motion_v3, "_post_json", side_effect=fake_post
    ):
        result = await motion_v3.tbot_motion_approach_until_close(
            target_distance_m=0.1,
            stop_distance_m=0.1,
            speed=0.2,
            timeout_s=10.0,
        )

    assert result["status"] == "completed"
    assert result["required_move_distance_m"] == pytest.approx(0.4)
    kwargs = move_mock.await_args.kwargs
    assert kwargs["distance_m"] == pytest.approx(0.4)
    assert kwargs["speed"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_approach_skips_move_when_already_reached():
    lidar_responses = [
        {"risk_level": "clear", "min_forward_distance_m": 0.35, "distances": {"front": 0.35}},
    ]

    def _client_factory(url: str):
        return _FakeLidarClient(url, lidar_responses)

    move_mock = AsyncMock()

    async def fake_post(path: str, payload=None):
        return {"status": "ok", "path": path}

    with patch.object(motion_v3, "Client", side_effect=_client_factory), patch.object(
        motion_v3, "_execute_forward_distance", new=move_mock
    ), patch.object(motion_v3, "_set_continuous_motion", new=AsyncMock(return_value=False)), patch.object(
        motion_v3, "_post_json", side_effect=fake_post
    ):
        result = await motion_v3.tbot_motion_approach_until_close(
            target_distance_m=0.5,
            stop_distance_m=0.1,
            speed=0.2,
            timeout_s=10.0,
        )

    assert result["status"] == "reached"
    assert result["required_move_distance_m"] == pytest.approx(0.0)
    assert result["requested_move_duration_s"] == pytest.approx(0.0)
    assert result["executed_move_duration_s"] == pytest.approx(0.0)
    move_mock.assert_not_called()


@pytest.mark.asyncio
async def test_approach_times_out_when_planned_move_exceeds_timeout():
    lidar_responses = [
        {"risk_level": "clear", "min_forward_distance_m": 1.0, "distances": {"front": 1.0}},
        {"risk_level": "clear", "min_forward_distance_m": 0.8, "distances": {"front": 0.8}},
    ]

    def _client_factory(url: str):
        return _FakeLidarClient(url, lidar_responses)

    move_mock = AsyncMock(return_value={"status": "completed", "move_posts": 4})

    async def fake_post(path: str, payload=None):
        return {"status": "ok", "path": path}

    with patch.object(motion_v3, "Client", side_effect=_client_factory), patch.object(
        motion_v3, "_execute_forward_distance", new=move_mock
    ), patch.object(motion_v3, "_set_continuous_motion", new=AsyncMock(return_value=False)), patch.object(
        motion_v3, "_post_json", side_effect=fake_post
    ):
        result = await motion_v3.tbot_motion_approach_until_close(
            target_distance_m=0.2,
            stop_distance_m=0.1,
            speed=0.1,
            timeout_s=2.0,
        )

    assert result["status"] == "timeout"
    assert result["required_move_distance_m"] == pytest.approx(0.8)
    assert result["requested_move_duration_s"] == pytest.approx(8.0)
    assert result["executed_move_duration_s"] <= 2.1
    move_mock.assert_awaited_once()
