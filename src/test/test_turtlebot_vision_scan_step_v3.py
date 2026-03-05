"""Targeted tests for TurtleBot v3 vision scan step behavior."""

import asyncio
import math
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import vision_mcp_server as vision_v3


class _FakeMotionClient:
    def __init__(self, _url: str, calls: list[tuple[str, dict]]):
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call_tool(self, name: str, payload: dict):
        self._calls.append((name, payload))
        return {"status": "ok"}


def test_search_step_constant_is_15_degrees():
    assert vision_v3._SEARCH_STEP_DEG == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_vision_search_object_rotates_in_15_degree_steps():
    motion_calls: list[tuple[str, dict]] = []

    def _client_factory(url: str):
        return _FakeMotionClient(url, motion_calls)

    detect_mock = AsyncMock(
        side_effect=[
            {"visible": False, "position": None, "confidence": 0.1, "bbox": None, "model_info": {"model": "x"}},
            {"visible": True, "position": "left", "confidence": 0.9, "bbox": {"cx": 0.2, "cy": 0.5, "w": 0.2, "h": 0.2}, "model_info": {"model": "x"}},
        ]
    )

    with patch.object(vision_v3, "Client", side_effect=_client_factory), patch.object(
        vision_v3, "tbot_vision_find_object", new=detect_mock
    ), patch("src.mcp.turtlebot_v3.vision_mcp_server.asyncio.sleep", new=AsyncMock(return_value=None)):
        result = await vision_v3._vision_search_object(
            object_name="chair",
            min_confidence=0.5,
            max_steps=3,
        )

    assert result["found"] is True
    assert result["steps_taken"] == 1
    assert result["degrees_rotated"] == pytest.approx(15.0)

    assert len(motion_calls) == 2
    assert motion_calls[0][0] == "tbot_motion_turn"
    expected_duration = math.radians(15.0) / vision_v3._SEARCH_ANGULAR_SPEED
    assert motion_calls[0][1]["duration_seconds"] == pytest.approx(expected_duration)
    assert motion_calls[1][0] == "tbot_motion_stop"
