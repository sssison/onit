"""Tests for src/mcp/turtlebot_v2/motion_mcp_server.py."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v2 import motion_mcp_server as motion_v2


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
    def __init__(self, post_response=None, post_error=None):
        self._post_response = post_response
        self._post_error = post_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None):
        if self._post_error is not None:
            raise self._post_error
        return self._post_response

    async def get(self, url):
        return _DummyResponse(status_code=200, json_data={"status": "online"})


def test_post_json_success():
    response = _DummyResponse(status_code=200, json_data={"status": "updated"})
    with patch.object(motion_v2.httpx, "AsyncClient", return_value=_DummyClient(post_response=response)):
        result = asyncio.run(motion_v2._post_json("/move", {"linear": 0.1, "angular": 0.0}))
    assert result["status"] == "updated"


@pytest.mark.parametrize("status_code", [404, 500])
def test_post_json_http_error(status_code):
    response = _DummyResponse(status_code=status_code, json_data={"error": "bad"}, text="bad request")
    with patch.object(motion_v2.httpx, "AsyncClient", return_value=_DummyClient(post_response=response)):
        with pytest.raises(RuntimeError, match=f"{status_code}"):
            asyncio.run(motion_v2._post_json("/move", {"linear": 0.1, "angular": 0.0}))


def test_post_json_request_error():
    request = httpx.Request("POST", "http://unit.test/move")
    timeout_error = httpx.RequestError("timeout", request=request)
    with patch.object(motion_v2.httpx, "AsyncClient", return_value=_DummyClient(post_error=timeout_error)):
        with pytest.raises(RuntimeError, match="Failed to reach motion server"):
            asyncio.run(motion_v2._post_json("/move", {"linear": 0.1, "angular": 0.0}))


def test_motion_move_clamps_and_verifies_health():
    expected_angular = (-motion_v2.MAX_ANGULAR) * motion_v2.ANGULAR_SIGN

    async def fake_post(path, payload=None):
        assert path == motion_v2.MOVE_PATH
        return {"status": "updated", "linear": payload["linear"], "angular": payload["angular"]}

    async def fake_health():
        return {"linear": motion_v2.MAX_LINEAR, "angular": expected_angular}, None

    with patch.object(motion_v2, "_post_json", side_effect=fake_post), patch.object(
        motion_v2, "_try_get_health", side_effect=fake_health
    ):
        result = asyncio.run(motion_v2.tbot_motion_move(linear=999.0, angular=-999.0))

    assert result["was_clamped"] is True
    assert result["commanded"]["linear"] == motion_v2.MAX_LINEAR
    assert result["commanded"]["angular"] == expected_angular
    assert result["commanded_input_frame"]["angular"] == -motion_v2.MAX_ANGULAR
    assert result["verification"]["command_matches_health"] is True


def test_motion_move_auto_stop_skips_stale_command():
    calls = []

    async def fake_post(path, payload=None):
        calls.append((path, payload))
        if path == motion_v2.MOVE_PATH:
            return {"status": "updated"}
        return {"status": "stopped"}

    with patch.object(motion_v2, "_post_json", side_effect=fake_post), patch.object(
        motion_v2, "_try_get_health", new=AsyncMock(return_value=(None, "no health"))
    ), patch.object(motion_v2, "_is_latest_motion_command", return_value=False), patch(
        "src.mcp.turtlebot_v2.motion_mcp_server.asyncio.sleep", new=AsyncMock(return_value=None)
    ):
        result = asyncio.run(motion_v2.tbot_motion_move(0.05, 0.0, duration_s=0.1))

    assert result["auto_stop_skipped_stale"] is True
    assert calls == [(motion_v2.MOVE_PATH, {"linear": 0.05, "angular": 0.0})]
