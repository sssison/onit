"""Tests for src/mcp/turtlebot_v3/vision_mcp_server.py."""

import asyncio
import base64
import json
import os
import sys
from types import SimpleNamespace
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


class _FakeMotionClient:
    def __init__(self, calls):
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call_tool(self, name: str, payload: dict):
        self._calls.append((name, payload))
        return {"status": "ok"}


def test_load_frame_as_base64_returns_ascii():
    raw = b"\xff\xd8\xff" + b"\x00" * 10
    with patch(
        "builtins.open",
        MagicMock(
            return_value=MagicMock(
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
                read=MagicMock(return_value=raw),
            )
        ),
    ):
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


def test_vision_describe_scene_returns_description():
    raw = b"\xff\xd8\xff"
    description_text = "A clear corridor with a door on the left."
    mock_client = _make_mock_llm_client(description_text)

    with patch(
        "builtins.open",
        MagicMock(
            return_value=MagicMock(
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
                read=MagicMock(return_value=raw),
            )
        ),
    ), patch.object(vision_v3, "AsyncOpenAI", return_value=mock_client):
        result = asyncio.run(vision_v3.tbot_vision_describe_scene(prompt="What do you see?"))

    assert result["description"] == description_text
    assert "model_info" in result
    create_kwargs = mock_client.chat.completions.create.await_args.kwargs
    assert create_kwargs["extra_body"] == vision_v3._vision_extra_body()
    user_text = create_kwargs["messages"][1]["content"][0]["text"]
    assert user_text.startswith("/no_think\n")


def test_vision_describe_scene_propagates_frame_error():
    with patch("builtins.open", side_effect=FileNotFoundError("missing")):
        with pytest.raises(RuntimeError, match="Frame file not found"):
            asyncio.run(vision_v3.tbot_vision_describe_scene())


def test_vision_find_object_visible_with_position():
    raw = b"\xff\xd8\xff"
    response_json = json.dumps(
        {
            "matched": True,
            "confidence": 0.85,
            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3},
        }
    )
    mock_client = _make_mock_llm_client(response_json)
    motion_calls: list[tuple[str, dict]] = []

    def fake_motion_client(url: str):
        assert url == vision_v3.MOTION_MCP_URL_V3
        return _FakeMotionClient(motion_calls)

    with patch(
        "builtins.open",
        MagicMock(
            return_value=MagicMock(
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
                read=MagicMock(return_value=raw),
            )
        ),
    ), patch.object(vision_v3, "AsyncOpenAI", return_value=mock_client), patch.object(
        vision_v3, "Client", side_effect=fake_motion_client
    ):
        result = asyncio.run(vision_v3.tbot_vision_find_object("chair"))

    assert result["visible"] is True
    assert result["position"] == "center"
    assert result["confidence"] == pytest.approx(0.85)
    assert result["scan_steps"] == 0
    assert len(motion_calls) == 0
    create_kwargs = mock_client.chat.completions.create.await_args.kwargs
    assert create_kwargs["extra_body"] == vision_v3._vision_extra_body()
    user_text = create_kwargs["messages"][1]["content"][0]["text"]
    assert user_text.startswith("/no_think\n")


def test_vision_find_object_not_visible():
    raw = b"\xff\xd8\xff"
    response_json = json.dumps({"matched": False, "confidence": 0.05, "bbox": None})
    mock_client = _make_mock_llm_client(response_json)
    motion_calls: list[tuple[str, dict]] = []

    def fake_motion_client(url: str):
        assert url == vision_v3.MOTION_MCP_URL_V3
        return _FakeMotionClient(motion_calls)

    with patch(
        "builtins.open",
        MagicMock(
            return_value=MagicMock(
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
                read=MagicMock(return_value=raw),
            )
        ),
    ), patch.object(vision_v3, "AsyncOpenAI", return_value=mock_client), patch.object(
        vision_v3, "Client", side_effect=fake_motion_client
    ), patch.object(
        vision_v3, "VISION_SCAN_MAX_STEPS", 1
    ):
        result = asyncio.run(vision_v3.tbot_vision_find_object("elephant"))

    assert result["visible"] is False
    assert result["position"] is None
    assert result["bbox"] is None
    assert result["stopped_reason"] == "max_retries"


def test_vision_get_object_bbox_is_frame_only():
    raw = b"\xff\xd8\xff"
    response_json = json.dumps(
        {
            "matched": True,
            "confidence": 0.85,
            "bbox": {"cx": 0.7, "cy": 0.5, "w": 0.2, "h": 0.3},
        }
    )
    mock_client = _make_mock_llm_client(response_json)

    with patch(
        "builtins.open",
        MagicMock(
            return_value=MagicMock(
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
                read=MagicMock(return_value=raw),
            )
        ),
    ), patch.object(vision_v3, "AsyncOpenAI", return_value=mock_client):
        result = asyncio.run(vision_v3.tbot_vision_get_object_bbox("chair"))

    assert result["visible"] is True
    assert result["position"] == "right"
    assert result["success"] is True


def test_vision_find_object_rejects_empty_name():
    with pytest.raises(ValueError, match="object_name must be a non-empty string"):
        asyncio.run(vision_v3.tbot_vision_find_object(""))


def test_vision_find_object_scan360_rotates_and_recenters():
    raw = b"\xff\xd8\xff"
    responses = [
        json.dumps({"matched": False, "confidence": 0.1, "bbox": None}),
        json.dumps({"matched": True, "confidence": 0.8, "bbox": {"cx": 0.8, "cy": 0.5, "w": 0.2, "h": 0.2}}),
        json.dumps({"matched": True, "confidence": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}),
    ]
    mock_create = AsyncMock(side_effect=[_completion_with_text(text) for text in responses])
    mock_client = MagicMock()
    mock_client.chat = MagicMock(completions=MagicMock(create=mock_create))
    motion_calls: list[tuple[str, dict]] = []

    def fake_motion_client(url: str):
        assert url == vision_v3.MOTION_MCP_URL_V3
        return _FakeMotionClient(motion_calls)

    with patch(
        "builtins.open",
        MagicMock(
            return_value=MagicMock(
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
                read=MagicMock(return_value=raw),
            )
        ),
    ), patch.object(vision_v3, "AsyncOpenAI", return_value=mock_client), patch.object(
        vision_v3, "Client", side_effect=fake_motion_client
    ):
        result = asyncio.run(vision_v3.tbot_vision_find_object("chair"))

    assert result["success"] is True
    assert result["visible"] is True
    assert result["scan_steps"] == 1
    assert result["recenter_applied"] is True
    assert len(motion_calls) == 2
    assert motion_calls[0][0] == "tbot_motion_turn"
    assert motion_calls[1][0] == "tbot_motion_turn"


def test_vision_find_object_scan360_fails_when_bbox_unavailable():
    raw = b"\xff\xd8\xff"
    responses = [
        json.dumps({"matched": True, "confidence": 0.8, "bbox": None}),
        json.dumps({"matched": True, "confidence": 0.8, "bbox": None}),
        json.dumps({"matched": True, "confidence": 0.8, "bbox": None}),
    ]
    mock_create = AsyncMock(side_effect=[_completion_with_text(text) for text in responses])
    mock_client = MagicMock()
    mock_client.chat = MagicMock(completions=MagicMock(create=mock_create))
    motion_calls: list[tuple[str, dict]] = []

    def fake_motion_client(url: str):
        assert url == vision_v3.MOTION_MCP_URL_V3
        return _FakeMotionClient(motion_calls)

    with patch(
        "builtins.open",
        MagicMock(
            return_value=MagicMock(
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
                read=MagicMock(return_value=raw),
            )
        ),
    ), patch.object(vision_v3, "AsyncOpenAI", return_value=mock_client), patch.object(
        vision_v3, "Client", side_effect=fake_motion_client
    ):
        result = asyncio.run(vision_v3.tbot_vision_find_object("chair"))

    assert result["success"] is False
    assert result["stopped_reason"] == "bbox_unavailable"
    assert len(motion_calls) == 0


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
    result = vision_v3._normalize_bbox({"cx": 0.5, "cy": 0.4, "w": 0.2})
    assert result is None


def test_inspect_floor_detects_requested_target_tokens():
    with patch.object(
        vision_v3,
        "tbot_vision_describe_scene",
        new=AsyncMock(return_value={"description": "Detected cables and floor tape in lower half.", "model_info": {}}),
    ):
        result = asyncio.run(vision_v3.tbot_vision_inspect_floor(targets=["cables", "floor_tape", "spill"]))

    assert result["status"] == "ok"
    assert "cables" in result["detected"]
    assert "floor_tape" in result["detected"]
    assert "spill" not in result["detected"]
