"""Regression tests for TurtleBot v3 scan-for-object behavior."""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import vision_mcp_server as vision_v3


class _DummyAsyncClient:
    def __init__(self, posts: list[tuple[str, dict | None]]):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url: str, json=None):
        self._posts.append((url, json))
        return {"status": "ok"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("direction", "expected_multiplier"), [("left", 1.0), ("right", -1.0)])
async def test_turn_step_direction_mapping_matches_motion_convention(direction: str, expected_multiplier: float):
    posts: list[tuple[str, dict | None]] = []

    def _client_factory(*_args, **_kwargs):
        return _DummyAsyncClient(posts)

    with patch.object(vision_v3, "MOTION_ANGULAR_SIGN", -1.0), patch.object(
        vision_v3.httpx, "AsyncClient", side_effect=_client_factory
    ):
        await vision_v3._turn_step_async(direction=direction, speed_rad_s=0.3, duration_s=0.001)

    move_payloads = [payload for url, payload in posts if url.endswith("/move") and isinstance(payload, dict)]
    assert move_payloads, "Expected at least one /move post from _turn_step_async"
    assert move_payloads[0]["angular"] == pytest.approx(expected_multiplier * 0.3)


@pytest.mark.asyncio
async def test_scan_for_object_promotes_180_budget_to_full_360():
    detect_mock = AsyncMock(
        return_value={
            "visible": False,
            "confidence": 0.0,
            "bbox": None,
            "position": None,
        }
    )
    turn_mock = AsyncMock(return_value=None)

    with patch.object(vision_v3, "_find_object_in_frame", new=detect_mock), patch.object(
        vision_v3, "_turn_step_async", new=turn_mock
    ):
        result = await vision_v3.tbot_vision_scan_for_object(
            object_name="bottle",
            step_deg=90.0,
            max_rotation_deg=180.0,
            direction="left",
        )

    assert result["status"] == "not_found"
    assert result["total_rotation_deg"] == pytest.approx(360.0)
    assert result["steps_taken"] == 4
    assert turn_mock.await_count == 4
