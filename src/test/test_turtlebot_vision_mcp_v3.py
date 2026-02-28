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


def test_reposition_for_object_returns_already_visible_without_scan():
    search_mock = AsyncMock()
    with patch.object(
        vision_v3,
        "tbot_vision_find_object",
        new=AsyncMock(
            return_value={
                "visible": True,
                "confidence": 0.9,
                "position": "center",
                "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                "model_info": {"name": "test-model"},
            }
        ),
    ), patch.object(vision_v3, "tbot_vision_search_object", new=search_mock):
        result = asyncio.run(vision_v3.tbot_vision_reposition_for_object(object_name="bottle"))

    assert result["status"] == "already_visible"
    assert result["found"] is True
    assert result["steps_taken"] == 0
    search_mock.assert_not_awaited()


def test_reposition_for_object_reacquires_with_search():
    search_mock = AsyncMock(
        return_value={
            "found": True,
            "position": "left",
            "confidence": 0.87,
            "bbox": {"cx": 0.3, "cy": 0.5, "w": 0.2, "h": 0.2},
            "steps_taken": 3,
            "degrees_rotated": 30.0,
            "model_info": {"name": "test-model"},
        }
    )
    with patch.object(
        vision_v3,
        "tbot_vision_find_object",
        new=AsyncMock(return_value={"visible": False, "confidence": 0.1, "position": None, "bbox": None}),
    ), patch.object(vision_v3, "tbot_vision_search_object", new=search_mock):
        result = asyncio.run(
            vision_v3.tbot_vision_reposition_for_object(
                object_name="bottle",
                min_confidence=0.5,
                max_steps=6,
            )
        )

    assert result["status"] == "reacquired"
    assert result["found"] is True
    assert result["steps_taken"] == 3
    assert result["degrees_rotated"] == pytest.approx(30.0)
    search_mock.assert_awaited_once()


def test_reposition_for_object_not_found_when_max_steps_zero():
    search_mock = AsyncMock()
    with patch.object(
        vision_v3,
        "tbot_vision_find_object",
        new=AsyncMock(return_value={"visible": False, "confidence": 0.2, "position": None, "bbox": None}),
    ), patch.object(vision_v3, "tbot_vision_search_object", new=search_mock):
        result = asyncio.run(
            vision_v3.tbot_vision_reposition_for_object(
                object_name="cup",
                min_confidence=0.5,
                max_steps=0,
            )
        )

    assert result["status"] == "not_found"
    assert result["found"] is False
    assert result["steps_taken"] == 0
    search_mock.assert_not_awaited()


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
            if name == "tbot_motion_approach_until_close":
                if not self._state["approach_results"]:
                    raise AssertionError("No mocked approach results left")
                return self._state["approach_results"].pop(0)
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
        "approach_results": [],
        "collision_results": [],
        "lidar_results": [],
    }


def test_search_and_approach_reaches_target_distance():
    state = _make_approach_state()
    state["approach_results"] = [{"status": "reached", "front_distance": 0.45, "move_posts": 4}]
    state["lidar_results"] = [
        {"status": "ok", "distance_m": 0.9, "distances": {"front": 0.9}},
        {"status": "ok", "distance_m": 0.45, "distances": {"front": 0.45}},
    ]

    def fake_client(url: str):
        return _FakeApproachClient(url, state)

    reposition_mock = AsyncMock(
        side_effect=[
            {"status": "already_visible", "found": True, "confidence": 0.9, "position": "center", "bbox": None, "steps_taken": 0},
            {"status": "already_visible", "found": True, "confidence": 0.9, "position": "center", "bbox": None, "steps_taken": 0},
        ]
    )
    with patch.object(
        vision_v3,
        "tbot_vision_search_object",
        new=AsyncMock(return_value={"found": True, "steps_taken": 2, "confidence": 0.9, "position": "center", "bbox": None}),
    ), patch.object(
        vision_v3,
        "tbot_vision_reposition_for_object",
        new=reposition_mock,
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
    assert result["phases"]["verification_scans"] == 2
    assert result["phases"]["reposition_turns"] == 0
    assert result["phases"]["forward_segments"] == 1
    assert result["final_front_distance_m"] == pytest.approx(0.45)
    assert len([name for name, _ in state["motion_calls"] if name == "tbot_motion_approach_until_close"]) == 1
    assert any(name == "tbot_motion_stop" for name, _ in state["motion_calls"])
    assert reposition_mock.await_count == 2


def test_search_and_approach_reached_without_forward_when_already_close():
    state = _make_approach_state()
    state["lidar_results"] = [{"status": "ok", "distance_m": 0.4, "distances": {"front": 0.4}}]

    def fake_client(url: str):
        return _FakeApproachClient(url, state)

    reposition_mock = AsyncMock(
        return_value={
            "status": "already_visible",
            "found": True,
            "confidence": 0.86,
            "position": "center",
            "bbox": None,
            "steps_taken": 0,
        }
    )
    with patch.object(
        vision_v3,
        "tbot_vision_search_object",
        new=AsyncMock(return_value={"found": True, "steps_taken": 1, "confidence": 0.9, "position": "center", "bbox": None}),
    ), patch.object(
        vision_v3,
        "tbot_vision_reposition_for_object",
        new=reposition_mock,
    ), patch.object(vision_v3, "Client", side_effect=fake_client):
        result = asyncio.run(
            vision_v3.tbot_vision_search_and_approach_object(
                object_name="bottle",
                target_distance_m=0.5,
                timeout_s=10.0,
            )
        )

    assert result["status"] == "reached"
    assert result["phases"]["forward_segments"] == 0
    assert result["phases"]["verification_scans"] == 1
    assert len([name for name, _ in state["motion_calls"] if name == "tbot_motion_approach_until_close"]) == 0
    assert reposition_mock.await_count == 1


def test_search_and_approach_verification_failed_after_one_forward():
    state = _make_approach_state()
    state["approach_results"] = [{"status": "reached", "front_distance": 0.5, "move_posts": 3}]
    state["lidar_results"] = [{"status": "ok", "distance_m": 0.9, "distances": {"front": 0.9}}]

    def fake_client(url: str):
        return _FakeApproachClient(url, state)

    reposition_mock = AsyncMock(
        side_effect=[
            {"status": "already_visible", "found": True, "confidence": 0.9, "position": "center", "bbox": None, "steps_taken": 0},
            {"status": "not_found", "found": False, "confidence": 0.2, "position": None, "bbox": None, "steps_taken": 4},
        ]
    )
    with patch.object(
        vision_v3,
        "tbot_vision_search_object",
        new=AsyncMock(return_value={"found": True, "steps_taken": 1, "confidence": 0.9, "position": "center", "bbox": None}),
    ), patch.object(
        vision_v3,
        "tbot_vision_reposition_for_object",
        new=reposition_mock,
    ), patch.object(vision_v3, "Client", side_effect=fake_client):
        result = asyncio.run(
            vision_v3.tbot_vision_search_and_approach_object(
                object_name="bottle",
                target_distance_m=0.5,
                timeout_s=10.0,
            )
        )

    assert result["status"] == "verification_failed"
    assert result["phases"]["forward_segments"] == 1
    assert result["phases"]["verification_scans"] == 2
    assert result["phases"]["reposition_turns"] == 4
    assert len([name for name, _ in state["motion_calls"] if name == "tbot_motion_approach_until_close"]) == 1
    assert reposition_mock.await_count == 2


def test_search_and_approach_returns_collision_blocked():
    state = _make_approach_state()
    state["approach_results"] = [{"status": "collision_risk", "front_distance": 0.22, "move_posts": 1}]
    state["lidar_results"] = [{"status": "ok", "distance_m": 0.9, "distances": {"front": 0.9}}]

    def fake_client(url: str):
        return _FakeApproachClient(url, state)

    with patch.object(
        vision_v3,
        "tbot_vision_search_object",
        new=AsyncMock(return_value={"found": True, "steps_taken": 0, "confidence": 0.9, "position": "center", "bbox": None}),
    ), patch.object(
        vision_v3,
        "tbot_vision_reposition_for_object",
        new=AsyncMock(return_value={"status": "already_visible", "found": True, "confidence": 0.95, "position": "center", "bbox": None, "steps_taken": 0}),
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
