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


@pytest.mark.asyncio
async def test_scan_for_object_stops_when_visible_even_low_confidence():
    detect_mock = AsyncMock(
        return_value={
            "visible": True,
            "confidence": 0.1,
            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
            "position": "center",
        }
    )
    turn_mock = AsyncMock(return_value=None)

    with patch.object(vision_v3, "_find_object_in_frame", new=detect_mock), patch.object(
        vision_v3, "_turn_step_async", new=turn_mock
    ):
        result = await vision_v3.tbot_vision_scan_for_object(
            object_name="bottle",
            step_deg=15.0,
            confidence_threshold=0.95,
        )

    assert result["status"] == "found"
    assert result["steps_taken"] == 0
    assert result["total_rotation_deg"] == pytest.approx(0.0)
    assert turn_mock.await_count == 0


@pytest.mark.asyncio
async def test_scan_for_object_checks_presence_before_each_turn_step():
    detect_mock = AsyncMock(
        side_effect=[
            {"visible": False, "confidence": 0.0, "bbox": None, "position": None},
            {"visible": False, "confidence": 0.0, "bbox": None, "position": None},
            {"visible": True, "confidence": 0.2, "bbox": {"cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.2}, "position": "left"},
        ]
    )
    turn_mock = AsyncMock(return_value=None)

    with patch.object(vision_v3, "_find_object_in_frame", new=detect_mock), patch.object(
        vision_v3, "_turn_step_async", new=turn_mock
    ):
        result = await vision_v3.tbot_vision_scan_for_object(
            object_name="chair",
            step_deg=15.0,
            max_rotation_deg=360.0,
            direction="left",
        )

    assert result["status"] == "found"
    assert result["steps_taken"] == 2
    assert result["total_rotation_deg"] == pytest.approx(30.0)
    assert turn_mock.await_count == 2


@pytest.mark.asyncio
async def test_scan_for_object_invalid_step_defaults_to_15_degrees():
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
            object_name="cone",
            step_deg=0.0,
            max_rotation_deg=360.0,
        )

    assert result["status"] == "not_found"
    assert result["steps_taken"] == 24
    assert result["total_rotation_deg"] == pytest.approx(360.0)
    assert turn_mock.await_count == 24
