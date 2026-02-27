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
