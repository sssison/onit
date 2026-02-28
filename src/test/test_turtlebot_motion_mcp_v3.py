"""Tests for src/mcp/turtlebot_v3/motion_mcp_server.py."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import motion_mcp_server as motion_v3


class _DummyResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.request = httpx.Request("POST", "http://unit.test/move")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=self.request, response=self)

    def json(self):
        return self._json_data


class _DummyClient:
    def __init__(self, post_response=None, post_error=None, get_response=None):
        self._post_response = post_response
        self._post_error = post_error
        self._get_response = get_response or _DummyResponse(200, {"status": "online"})

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None):
        if self._post_error is not None:
            raise self._post_error
        return self._post_response

    async def get(self, url):
        return self._get_response


def test_post_json_success():
    response = _DummyResponse(status_code=200, json_data={"status": "updated"})
    with patch.object(motion_v3.httpx, "AsyncClient", return_value=_DummyClient(post_response=response)):
        result = asyncio.run(motion_v3._post_json("/move", {"linear": 0.1, "angular": 0.0}))
    assert result["status"] == "updated"


@pytest.mark.parametrize("status_code", [404, 500])
def test_post_json_http_error(status_code):
    response = _DummyResponse(status_code=status_code, json_data={"error": "bad"}, text="bad request")
    with patch.object(motion_v3.httpx, "AsyncClient", return_value=_DummyClient(post_response=response)):
        with pytest.raises(RuntimeError, match=f"{status_code}"):
            asyncio.run(motion_v3._post_json("/move", {"linear": 0.1, "angular": 0.0}))


def test_post_json_request_error():
    request = httpx.Request("POST", "http://unit.test/move")
    timeout_error = httpx.RequestError("timeout", request=request)
    with patch.object(motion_v3.httpx, "AsyncClient", return_value=_DummyClient(post_error=timeout_error)):
        with pytest.raises(RuntimeError, match="Failed to reach motion server"):
            asyncio.run(motion_v3._post_json("/move", {"linear": 0.1, "angular": 0.0}))


def test_motion_health_online():
    async def fake_health():
        return {"linear": 0.0, "angular": 0.0}, None

    with patch.object(motion_v3, "_try_get_health", side_effect=fake_health):
        result = asyncio.run(motion_v3.tbot_motion_health())

    assert result["status"] == "online"
    assert result["reachable"] is True
    assert result["base_url"] == motion_v3.BASE_URL


def test_motion_health_offline():
    async def fake_health():
        return None, "connection refused"

    with patch.object(motion_v3, "_try_get_health", side_effect=fake_health):
        result = asyncio.run(motion_v3.tbot_motion_health())

    assert result["status"] == "offline"
    assert result["reachable"] is False


def test_motion_stop_calls_stop_endpoint():
    calls = []

    async def fake_post(path, payload=None):
        calls.append(path)
        return {"status": "stopped"}

    with patch.object(motion_v3, "_post_json", side_effect=fake_post):
        result = asyncio.run(motion_v3.tbot_motion_stop())

    assert motion_v3.STOP_PATH in calls
    assert result["base_url"] == motion_v3.BASE_URL


def test_motion_move_forward_clamps_speed():
    calls = []

    async def fake_post(path, payload=None):
        calls.append((path, payload))
        return {"status": "ok"}

    with patch.object(motion_v3, "_post_json", side_effect=fake_post), patch(
        "src.mcp.turtlebot_v3.motion_mcp_server.asyncio.sleep", new=AsyncMock(return_value=None)
    ):
        result = asyncio.run(motion_v3.tbot_motion_move_forward(speed=999.0, duration_seconds=0.5))

    assert result["status"] == "completed"
    assert result["was_clamped"] is True
    move_call = next(c for c in calls if c[0] == motion_v3.MOVE_PATH)
    assert move_call[1]["linear"] == motion_v3.MAX_LINEAR
    assert move_call[1]["angular"] == 0.0
    # stop should be called
    assert any(c[0] == motion_v3.STOP_PATH for c in calls)


def test_motion_move_forward_rejects_non_positive_duration():
    with pytest.raises(ValueError, match="duration_seconds must be > 0"):
        asyncio.run(motion_v3.tbot_motion_move_forward(speed=0.1, duration_seconds=0.0))


def test_motion_turn_left_angular_direction():
    calls = []

    async def fake_post(path, payload=None):
        calls.append((path, payload))
        return {"status": "ok"}

    with patch.object(motion_v3, "_post_json", side_effect=fake_post), patch(
        "src.mcp.turtlebot_v3.motion_mcp_server.asyncio.sleep", new=AsyncMock(return_value=None)
    ):
        result = asyncio.run(motion_v3.tbot_motion_turn(direction="left", speed=0.5, duration_seconds=0.5))

    assert result["status"] == "completed"
    assert result["direction"] == "left"
    move_call = next(c for c in calls if c[0] == motion_v3.MOVE_PATH)
    assert move_call[1]["linear"] == 0.0
    # left: input_frame_sign=-1 (V2 convention: positive=right), cmd = -speed * ANGULAR_SIGN
    expected_angular = -0.5 * motion_v3.ANGULAR_SIGN
    assert abs(move_call[1]["angular"] - expected_angular) < 1e-9


def test_motion_turn_right_angular_direction():
    calls = []

    async def fake_post(path, payload=None):
        calls.append((path, payload))
        return {"status": "ok"}

    with patch.object(motion_v3, "_post_json", side_effect=fake_post), patch(
        "src.mcp.turtlebot_v3.motion_mcp_server.asyncio.sleep", new=AsyncMock(return_value=None)
    ):
        asyncio.run(motion_v3.tbot_motion_turn(direction="right", speed=0.5, duration_seconds=0.5))

    move_call = next(c for c in calls if c[0] == motion_v3.MOVE_PATH)
    expected_angular = 0.5 * motion_v3.ANGULAR_SIGN  # right: input_frame_sign=+1
    assert abs(move_call[1]["angular"] - expected_angular) < 1e-9


def test_motion_turn_rejects_invalid_direction():
    with pytest.raises(ValueError, match="direction must be 'left' or 'right'"):
        asyncio.run(motion_v3.tbot_motion_turn(direction="forward", speed=0.5, duration_seconds=0.5))


def test_motion_get_robot_status_moving():
    async def fake_health():
        return {"linear": 0.1, "angular": 0.0}, None

    with patch.object(motion_v3, "_try_get_health", side_effect=fake_health):
        result = asyncio.run(motion_v3.tbot_motion_get_robot_status())

    assert result["moving"] is True
    assert result["linear"] == pytest.approx(0.1)


def test_motion_get_robot_status_stopped():
    async def fake_health():
        return {"linear": 0.0, "angular": 0.0}, None

    with patch.object(motion_v3, "_try_get_health", side_effect=fake_health):
        result = asyncio.run(motion_v3.tbot_motion_get_robot_status())

    assert result["moving"] is False


def test_motion_get_robot_status_offline():
    async def fake_health():
        return None, "timeout"

    with patch.object(motion_v3, "_try_get_health", side_effect=fake_health):
        result = asyncio.run(motion_v3.tbot_motion_get_robot_status())

    assert result["moving"] is None
    assert result["error"] == "timeout"


def test_approach_until_close_reached():
    move_calls = []
    tick = 0

    async def fake_post(path, payload=None):
        move_calls.append((path, payload))
        return {"status": "ok"}

    async def fake_sleep(t):
        pass

    class _FakeLidarClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def call_tool(self, name, args=None):
            nonlocal tick
            if name == "tbot_lidar_check_collision":
                return {"risk_level": "clear", "distances": {"front": 0.8, "left": None, "right": None, "rear": None}}
            if name == "tbot_lidar_get_obstacle_distances":
                tick += 1
                dist = 0.8 if tick < 3 else 0.2  # reach target on tick 3
                return {"status": "ok", "distance_m": dist, "distances": {"front": dist}}
            return {}

    with patch.object(motion_v3, "_post_json", side_effect=fake_post), patch(
        "src.mcp.turtlebot_v3.motion_mcp_server.asyncio.sleep", side_effect=fake_sleep
    ), patch.object(motion_v3, "Client", return_value=_FakeLidarClient()):
        result = asyncio.run(
            motion_v3.tbot_motion_approach_until_close(
                target_distance_m=0.3,
                stop_distance_m=0.15,
                speed=0.1,
                timeout_s=10.0,
            )
        )

    assert result["status"] == "reached"
    assert result["front_distance"] == pytest.approx(0.2)


def test_approach_until_close_collision_risk():
    async def fake_post(path, payload=None):
        return {"status": "ok"}

    async def fake_sleep(t):
        pass

    class _FakeLidarClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def call_tool(self, name, args=None):
            if name == "tbot_lidar_check_collision":
                return {"risk_level": "stop", "distances": {"front": 0.1, "left": None, "right": None, "rear": None}}
            return {}

    with patch.object(motion_v3, "_post_json", side_effect=fake_post), patch(
        "src.mcp.turtlebot_v3.motion_mcp_server.asyncio.sleep", side_effect=fake_sleep
    ), patch.object(motion_v3, "Client", return_value=_FakeLidarClient()):
        result = asyncio.run(
            motion_v3.tbot_motion_approach_until_close(
                target_distance_m=0.3,
                stop_distance_m=0.15,
                speed=0.1,
                timeout_s=10.0,
            )
        )

    assert result["status"] == "collision_risk"
