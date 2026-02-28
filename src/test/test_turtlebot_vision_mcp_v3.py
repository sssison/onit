"""Tests for src/mcp/turtlebot_v3/vision_mcp_server.py."""

import asyncio
import base64
import json
import os
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import vision_mcp_server as vision_v3


def _completion_with_text(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _make_mock_llm_client(response_text: str):
    mock_create = AsyncMock(return_value=_completion_with_text(response_text))
    mock_client = MagicMock()
    mock_client.chat = MagicMock(completions=MagicMock(create=mock_create))
    return mock_client


def _fake_frame_bytes() -> bytes:
    return base64.b64decode(base64.b64encode(b"fake-jpeg-data"))


def test_load_frame_as_base64_returns_ascii():
    raw = b"\xff\xd8\xff" + b"\x00" * 10
    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False), read=MagicMock(return_value=raw)))):
        result = vision_v3._load_frame_as_base64("/dev/shm/latest_frame.jpg")
    assert result == base64.b64encode(raw).decode("ascii")


def test_load_frame_raises_on_missing_file():
    with patch("builtins.open", side_effect=FileNotFoundError("not found")):
        with pytest.raises(RuntimeError, match="Frame file not found"):
            vision_v3._load_frame_as_base64("/dev/shm/missing.jpg")


def test_load_frame_raises_on_os_error():
    with patch("builtins.open", side_effect=OSError("permission denied")):
        with pytest.raises(RuntimeError, match="Could not read frame file"):
            vision_v3._load_frame_as_base64("/dev/shm/latest_frame.jpg")


def test_vision_health_frame_exists():
    with patch("os.stat") as mock_stat:
        mock_stat.return_value = SimpleNamespace(st_size=12345)
        result = asyncio.run(vision_v3.tbot_vision_health())

    assert result["status"] == "online"
    assert result["frame_exists"] is True
    assert result["frame_size_bytes"] == 12345


def test_vision_health_no_frame():
    with patch("os.stat", side_effect=OSError("no such file")):
        result = asyncio.run(vision_v3.tbot_vision_health())

    assert result["status"] == "no_frame"
    assert result["frame_exists"] is False
    assert result["frame_size_bytes"] is None


def test_vision_describe_scene_returns_description():
    raw = b"\xff\xd8\xff"
    description_text = "A clear corridor with a door on the left."
    mock_client = _make_mock_llm_client(description_text)

    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False), read=MagicMock(return_value=raw)))), patch.object(
        vision_v3, "AsyncOpenAI", return_value=mock_client
    ):
        result = asyncio.run(vision_v3.tbot_vision_describe_scene(prompt="What do you see?"))

    assert result["description"] == description_text
    assert "model_info" in result


def test_vision_describe_scene_propagates_frame_error():
    with patch("builtins.open", side_effect=FileNotFoundError("missing")):
        with pytest.raises(RuntimeError, match="Frame file not found"):
            asyncio.run(vision_v3.tbot_vision_describe_scene())


def test_vision_find_object_visible_with_position():
    raw = b"\xff\xd8\xff"
    response_json = json.dumps({"matched": True, "confidence": 0.85, "bbox": {"cx": 0.7, "cy": 0.5, "w": 0.2, "h": 0.3}})
    mock_client = _make_mock_llm_client(response_json)

    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False), read=MagicMock(return_value=raw)))), patch.object(
        vision_v3, "AsyncOpenAI", return_value=mock_client
    ):
        result = asyncio.run(vision_v3.tbot_vision_find_object("chair"))

    assert result["visible"] is True
    assert result["position"] == "right"  # cx=0.7 > 0.66
    assert result["confidence"] == pytest.approx(0.85)
    assert result["bbox"] == {"cx": 0.7, "cy": 0.5, "w": 0.2, "h": 0.3}


def test_vision_find_object_position_left():
    raw = b"\xff\xd8\xff"
    response_json = json.dumps({"matched": True, "confidence": 0.9, "bbox": {"cx": 0.2, "cy": 0.5, "w": 0.1, "h": 0.2}})
    mock_client = _make_mock_llm_client(response_json)

    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False), read=MagicMock(return_value=raw)))), patch.object(
        vision_v3, "AsyncOpenAI", return_value=mock_client
    ):
        result = asyncio.run(vision_v3.tbot_vision_find_object("box"))

    assert result["position"] == "left"  # cx=0.2 < 0.33


def test_vision_find_object_position_center():
    raw = b"\xff\xd8\xff"
    response_json = json.dumps({"matched": True, "confidence": 0.75, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.2}})
    mock_client = _make_mock_llm_client(response_json)

    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False), read=MagicMock(return_value=raw)))), patch.object(
        vision_v3, "AsyncOpenAI", return_value=mock_client
    ):
        result = asyncio.run(vision_v3.tbot_vision_find_object("door"))

    assert result["position"] == "center"  # cx=0.5 in [0.33, 0.66]


def test_vision_find_object_not_visible():
    raw = b"\xff\xd8\xff"
    response_json = json.dumps({"matched": False, "confidence": 0.05, "bbox": None})
    mock_client = _make_mock_llm_client(response_json)

    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False), read=MagicMock(return_value=raw)))), patch.object(
        vision_v3, "AsyncOpenAI", return_value=mock_client
    ):
        result = asyncio.run(vision_v3.tbot_vision_find_object("elephant"))

    assert result["visible"] is False
    assert result["position"] is None
    assert result["bbox"] is None


def test_vision_find_object_no_bbox_when_matched_but_null_bbox():
    raw = b"\xff\xd8\xff"
    response_json = json.dumps({"matched": True, "confidence": 0.8, "bbox": None})
    mock_client = _make_mock_llm_client(response_json)

    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False), read=MagicMock(return_value=raw)))), patch.object(
        vision_v3, "AsyncOpenAI", return_value=mock_client
    ):
        result = asyncio.run(vision_v3.tbot_vision_find_object("table"))

    assert result["visible"] is True
    assert result["position"] is None  # no valid bbox to derive position from
    assert result["bbox"] is None


def test_vision_find_object_rejects_empty_name():
    with pytest.raises(ValueError, match="object_name must be a non-empty string"):
        asyncio.run(vision_v3.tbot_vision_find_object(""))


def test_normalize_confidence_clamps():
    assert vision_v3._normalize_confidence(1.5) == 1.0
    assert vision_v3._normalize_confidence(-0.1) == 0.0
    assert vision_v3._normalize_confidence(0.7) == pytest.approx(0.7)
    assert vision_v3._normalize_confidence(None) is None
    assert vision_v3._normalize_confidence("not-a-number") is None


def test_normalize_bbox_valid():
    result = vision_v3._normalize_bbox({"cx": 0.5, "cy": 0.4, "w": 0.2, "h": 0.3})
    assert result == {"cx": 0.5, "cy": 0.4, "w": 0.2, "h": 0.3}


def test_normalize_bbox_invalid_missing_key():
    result = vision_v3._normalize_bbox({"cx": 0.5, "cy": 0.4, "w": 0.2})  # missing "h"
    assert result is None


def test_normalize_bbox_rejects_non_dict():
    assert vision_v3._normalize_bbox("0.5,0.4,0.2,0.3") is None
    assert vision_v3._normalize_bbox(None) is None


class _FakeApproachClient:
    def __init__(self, url: str, state: dict[str, Any]):
        self._url = url
        self._state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call_tool(self, name: str, payload: dict[str, Any]):
        if self._url == vision_v3.MOTION_MCP_URL_V3:
            self._state["motion_calls"].append((name, payload))
            if name == "tbot_motion_move_forward":
                if not self._state["forward_results"]:
                    raise AssertionError("No mocked forward results left")
                return self._state["forward_results"].pop(0)
            if name in ("tbot_motion_turn", "tbot_motion_stop"):
                return {"status": "ok"}
            raise AssertionError(f"Unexpected motion tool call: {name}")

        if self._url == vision_v3.LIDAR_MCP_URL_V3:
            self._state["lidar_calls"].append((name, payload))
            if name == "tbot_lidar_check_collision":
                if self._state["collision_results"]:
                    return self._state["collision_results"].pop(0)
                return {"risk_level": "clear", "distances": {"front": 1.0, "left": 1.0, "right": 1.0, "rear": 1.0}}
            if name == "tbot_lidar_get_obstacle_distances":
                if not self._state["lidar_results"]:
                    raise AssertionError("No mocked lidar results left")
                return self._state["lidar_results"].pop(0)
            raise AssertionError(f"Unexpected lidar tool call: {name}")

        raise AssertionError(f"Unexpected MCP URL in test: {self._url}")


def _make_approach_state() -> dict[str, Any]:
    return {
        "motion_calls": [],
        "lidar_calls": [],
        "forward_results": [],
        "collision_results": [],
        "lidar_results": [],
    }


def test_search_and_approach_reaches_target_distance():
    state = _make_approach_state()
    state["forward_results"] = [
        {"status": "completed"},
        {"status": "completed"},
    ]
    state["lidar_results"] = [
        {"status": "ok", "distance_m": 0.9, "distances": {"front": 0.9}},
        {"status": "ok", "distance_m": 0.45, "distances": {"front": 0.45}},
    ]

    def fake_client(url: str):
        return _FakeApproachClient(url, state)

    with patch.object(
        vision_v3,
        "tbot_vision_search_object",
        new=AsyncMock(return_value={"found": True, "steps_taken": 2, "confidence": 0.9}),
    ), patch.object(
        vision_v3,
        "tbot_vision_find_object",
        new=AsyncMock(
            side_effect=[
                {"visible": True, "confidence": 0.9, "position": "center", "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                {"visible": True, "confidence": 0.9, "position": "center", "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
            ]
        ),
    ), patch.object(vision_v3, "Client", side_effect=fake_client):
        result = asyncio.run(
            vision_v3.tbot_vision_search_and_approach_object(
                object_name="bottle",
                target_distance_m=0.5,
                timeout_s=10.0,
            )
        )

    assert result["status"] == "reached"
    assert result["phases"]["search_steps"] == 2
    assert result["phases"]["forward_segments"] == 2
    assert result["final_front_distance_m"] == pytest.approx(0.45)
    assert any(name == "tbot_motion_stop" for name, _ in state["motion_calls"])


def test_search_and_approach_lost_then_reacquires_locally():
    state = _make_approach_state()
    state["forward_results"] = [{"status": "completed"}]
    state["lidar_results"] = [{"status": "ok", "distance_m": 0.4, "distances": {"front": 0.4}}]

    def fake_client(url: str):
        return _FakeApproachClient(url, state)

    with patch.object(
        vision_v3,
        "tbot_vision_search_object",
        new=AsyncMock(return_value={"found": True, "steps_taken": 1, "confidence": 0.9}),
    ), patch.object(
        vision_v3,
        "tbot_vision_find_object",
        new=AsyncMock(
            side_effect=[
                {"visible": False, "confidence": 0.0, "position": None, "bbox": None},
                {"visible": True, "confidence": 0.86, "position": "center", "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                {"visible": True, "confidence": 0.87, "position": "center", "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
            ]
        ),
    ), patch.object(vision_v3, "Client", side_effect=fake_client):
        result = asyncio.run(
            vision_v3.tbot_vision_search_and_approach_object(
                object_name="bottle",
                target_distance_m=0.5,
                reacquire_local_steps=1,
                timeout_s=10.0,
            )
        )

    assert result["status"] == "reached"
    assert result["phases"]["reacquire_attempts"] == 1
    assert any(name == "tbot_motion_turn" for name, _ in state["motion_calls"])


def test_search_and_approach_returns_collision_blocked():
    state = _make_approach_state()
    state["forward_results"] = [{"status": "collision_risk", "front_distance": 0.22}]

    def fake_client(url: str):
        return _FakeApproachClient(url, state)

    with patch.object(
        vision_v3,
        "tbot_vision_search_object",
        new=AsyncMock(return_value={"found": True, "steps_taken": 0, "confidence": 0.9}),
    ), patch.object(
        vision_v3,
        "tbot_vision_find_object",
        new=AsyncMock(return_value={"visible": True, "confidence": 0.95, "position": "center", "bbox": None}),
    ), patch.object(vision_v3, "Client", side_effect=fake_client):
        result = asyncio.run(
            vision_v3.tbot_vision_search_and_approach_object(
                object_name="box",
                target_distance_m=0.5,
                timeout_s=10.0,
            )
        )

    assert result["status"] == "collision_blocked"
    assert result["final_front_distance_m"] == pytest.approx(0.22)
    assert any(name == "tbot_motion_stop" for name, _ in state["motion_calls"])


def test_search_and_approach_returns_not_found_on_initial_scan_failure():
    with patch.object(
        vision_v3,
        "tbot_vision_search_object",
        new=AsyncMock(return_value={"found": False, "steps_taken": 11}),
    ):
        result = asyncio.run(
            vision_v3.tbot_vision_search_and_approach_object(
                object_name="cup",
                target_distance_m=0.5,
                timeout_s=10.0,
            )
        )

    assert result["status"] == "not_found"
    assert result["phases"]["search_steps"] == 11
