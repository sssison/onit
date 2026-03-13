"""Tests for src/mcp/turtlebot_v3/motion_mcp_server.py."""

import os
import sys
import types
from unittest.mock import AsyncMock, patch

import pytest


def _install_motion_test_stubs() -> None:
    """Install lightweight stubs when FastMCP is unavailable."""
    if "fastmcp" not in sys.modules:
        fastmcp_mod = types.ModuleType("fastmcp")

        class FastMCP:
            def __init__(self, _name: str):
                self._name = _name

            def tool(self):
                def _decorator(fn):
                    return fn

                return _decorator

            def run(self, **_kwargs):
                return None

        fastmcp_mod.FastMCP = FastMCP
        sys.modules["fastmcp"] = fastmcp_mod


_install_motion_test_stubs()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import motion_mcp_server as motion_v3


async def _run_turn(
    direction: str,
    speed: float = 0.5,
    duration_seconds: float = 0.2,
    health_offset: float = 0.0,
) -> tuple[dict, dict]:
    """Run tbot_motion_turn with mocked network calls and capture motion payloads."""
    captured: dict[str, float] = {}

    async def fake_post_move_for_duration(
        linear: float,
        angular: float,
        duration_s: float,
    ) -> tuple[dict, int]:
        captured["linear"] = linear
        captured["angular"] = angular
        captured["duration_s"] = duration_s
        return {"status": "updated", "linear": linear, "angular": angular}, 3

    async def fake_post_json(path: str, payload=None) -> dict:
        return {"status": "ok", "path": path, "payload": payload}

    async def fake_health() -> tuple[dict, None]:
        angular = float(captured.get("angular", 0.0)) + health_offset
        return {"angular": angular}, None

    with patch.object(motion_v3, "_set_continuous_motion", new=AsyncMock(return_value=False)), patch.object(
        motion_v3, "_post_move_for_duration", new=AsyncMock(side_effect=fake_post_move_for_duration)
    ), patch.object(
        motion_v3, "_post_json", new=AsyncMock(side_effect=fake_post_json)
    ), patch.object(
        motion_v3, "_try_get_health", new=AsyncMock(side_effect=fake_health)
    ):
        result = await motion_v3.tbot_motion_turn(
            direction=direction,
            speed=speed,
            duration_seconds=duration_seconds,
        )
    return result, captured


@pytest.mark.asyncio
@pytest.mark.parametrize(("direction", "input_multiplier"), [("left", -1.0), ("right", 1.0)])
async def test_turn_direction_maps_to_expected_angular_cmd(direction: str, input_multiplier: float):
    speed = 0.4
    result, captured = await _run_turn(
        direction=direction,
        speed=speed,
        duration_seconds=0.3,
    )

    expected_angular = input_multiplier * speed * motion_v3.ANGULAR_SIGN
    assert result["status"] == "completed"
    assert result["angular_cmd"] == pytest.approx(expected_angular)
    assert captured["angular"] == pytest.approx(expected_angular)
    assert result["command_received_by_server"] is True


@pytest.mark.asyncio
async def test_turn_respects_angular_sign_override():
    with patch.object(motion_v3, "ANGULAR_SIGN", 1.0):
        result_right, _ = await _run_turn(direction="right", speed=0.5, duration_seconds=0.2)
        result_left, _ = await _run_turn(direction="left", speed=0.5, duration_seconds=0.2)

    assert result_right["angular_cmd"] == pytest.approx(0.5)
    assert result_left["angular_cmd"] == pytest.approx(-0.5)


@pytest.mark.asyncio
async def test_turn_clamps_speed_and_uses_abs_value():
    result, _ = await _run_turn(
        direction="right",
        speed=-999.0,
        duration_seconds=0.2,
    )

    assert result["speed"] == pytest.approx(motion_v3.MAX_ANGULAR)
    assert result["was_clamped"] is True


@pytest.mark.asyncio
async def test_turn_marks_command_not_received_when_health_differs():
    result, _ = await _run_turn(
        direction="left",
        speed=0.3,
        duration_seconds=0.2,
        health_offset=0.01,
    )

    assert result["command_received_by_server"] is False


@pytest.mark.asyncio
async def test_turn_rejects_invalid_direction():
    with pytest.raises(ValueError, match="direction must be 'left' or 'right'"):
        await motion_v3.tbot_motion_turn(direction="up", speed=0.3, duration_seconds=0.2)


@pytest.mark.asyncio
async def test_move_forward_continuous_streams_then_stops():
    set_stream_mock = AsyncMock(side_effect=[False, True])
    post_json_mock = AsyncMock(return_value={"status": "ok", "path": motion_v3.STOP_PATH})
    sleep_mock = AsyncMock(return_value=None)
    with patch.object(motion_v3, "_set_continuous_motion", new=set_stream_mock), patch.object(
        motion_v3, "_post_json", new=post_json_mock
    ), patch.object(
        motion_v3.asyncio, "sleep", new=sleep_mock
    ), patch.object(
        motion_v3.time, "monotonic", side_effect=[10.0, 10.2]
    ):
        result = await motion_v3.tbot_motion_move_forward_continuous(duration_seconds=0.2, speed=0.1)

    assert result["status"] == "completed"
    assert result["speed"] == pytest.approx(0.1)
    assert result["duration_seconds"] == pytest.approx(0.2)
    assert result["executed_duration_seconds"] == pytest.approx(0.2)
    assert set_stream_mock.await_args_list[0].args == (0.1, 0.0)
    assert set_stream_mock.await_args_list[1].args == (None,)
    post_json_mock.assert_awaited_once_with(motion_v3.STOP_PATH)
    sleep_mock.assert_awaited_once_with(0.2)


@pytest.mark.asyncio
async def test_move_forward_continuous_clamps_speed():
    set_stream_mock = AsyncMock(side_effect=[False, False])
    with patch.object(motion_v3, "_set_continuous_motion", new=set_stream_mock), patch.object(
        motion_v3, "_post_json", new=AsyncMock(return_value={"status": "ok"})
    ), patch.object(
        motion_v3.asyncio, "sleep", new=AsyncMock(return_value=None)
    ), patch.object(
        motion_v3.time, "monotonic", side_effect=[1.0, 1.0]
    ):
        result = await motion_v3.tbot_motion_move_forward_continuous(duration_seconds=0.1, speed=999.0)

    assert result["speed"] == pytest.approx(motion_v3.MAX_LINEAR)
    assert result["was_clamped"] is True
    assert set_stream_mock.await_args_list[0].args == (motion_v3.MAX_LINEAR, 0.0)


@pytest.mark.asyncio
async def test_move_forward_continuous_requires_positive_duration():
    with pytest.raises(ValueError, match="duration_seconds must be > 0"):
        await motion_v3.tbot_motion_move_forward_continuous(duration_seconds=0.0, speed=0.1)
