"""Tests for src/mcp/turtlebot_v2/vision_mcp_server.py."""

import asyncio
import base64
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v2 import vision_mcp_server as vision_v2


def _completion_with_text(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_vision_analyze_scene_returns_structured_json():
    image_b64 = base64.b64encode(b"first-image").decode("ascii")
    content = json.dumps(
        {
            "summary": "Clear path ahead",
            "objects": ["wall", "chair"],
            "visible_text": ["EXIT"],
            "navigation": {"forward_clear": True},
            "hazards": [],
            "confidence": 0.9,
        }
    )

    mock_create = AsyncMock(return_value=_completion_with_text(content))
    mock_client = MagicMock()
    mock_client.chat = MagicMock(completions=MagicMock(create=mock_create))

    with patch.object(vision_v2, "AsyncOpenAI", return_value=mock_client):
        result = asyncio.run(vision_v2.tbot_vision_analyze_scene(images=[image_b64], task="Analyze this scene."))

    assert result["summary"] == "Clear path ahead"
    assert result["objects"] == ["wall", "chair"]
    assert result["visible_text"] == ["EXIT"]
    assert result["navigation"]["forward_clear"] is True
    assert result["confidence"] == 0.9
    assert "model_info" in result


def test_vision_analyze_scene_parses_json_fallback():
    image_b64 = base64.b64encode(b"first-image").decode("ascii")
    malformed = (
        "analysis result:\n"
        "```json\n"
        '{"summary":"Looks safe","objects":["box"],"visible_text":[],"navigation":{"turn":"left"},"hazards":[],"confidence":0.8}\n'
        "```"
    )

    mock_create = AsyncMock(return_value=_completion_with_text(malformed))
    mock_client = MagicMock()
    mock_client.chat = MagicMock(completions=MagicMock(create=mock_create))

    with patch.object(vision_v2, "AsyncOpenAI", return_value=mock_client):
        result = asyncio.run(vision_v2.tbot_vision_analyze_scene(images=[image_b64], task="Analyze."))

    assert result["summary"] == "Looks safe"
    assert result["objects"] == ["box"]
    assert result["hazards"] == []


def test_vision_analyze_scene_requires_images():
    with pytest.raises(ValueError, match="images is required"):
        asyncio.run(vision_v2.tbot_vision_analyze_scene(images=[], task="Analyze."))


def test_vision_analyze_scene_uses_first_image_only():
    first = base64.b64encode(b"first-image").decode("ascii")
    second = base64.b64encode(b"second-image").decode("ascii")
    content = json.dumps(
        {
            "summary": "ok",
            "objects": [],
            "visible_text": [],
            "navigation": {},
            "hazards": [],
            "confidence": 0.7,
        }
    )

    mock_create = AsyncMock(return_value=_completion_with_text(content))
    mock_client = MagicMock()
    mock_client.chat = MagicMock(completions=MagicMock(create=mock_create))

    with patch.object(vision_v2, "AsyncOpenAI", return_value=mock_client):
        asyncio.run(vision_v2.tbot_vision_analyze_scene(images=[first, second], task="Analyze."))

    call_kwargs = mock_create.call_args.kwargs
    image_url = call_kwargs["messages"][1]["content"][1]["image_url"]["url"]
    assert first in image_url
    assert second not in image_url


def test_normalize_detection_with_position_uses_bbox_center():
    parsed = {
        "matched": True,
        "confidence": 0.9,
        "evidence": "object found",
        "bbox": {"cx": 0.8, "cy": 0.4, "w": 0.2, "h": 0.3},
    }

    result = vision_v2._normalize_detection_with_position(parsed, raw_text=json.dumps(parsed))

    assert result["bbox"] == {"cx": 0.8, "cy": 0.4, "w": 0.2, "h": 0.3}
    assert result["x_center"] == pytest.approx(0.8)
    assert result["in_frame_offset_deg"] == pytest.approx((0.8 - 0.5) * vision_v2.CAMERA_HFOV_DEG)


def test_normalize_detection_with_position_selects_highest_confidence_bbox():
    parsed = {
        "matched": True,
        "confidence": 0.7,
        "evidence": "multiple",
        "detections": [
            {"confidence": 0.55, "bbox": {"cx": 0.2, "cy": 0.5, "w": 0.2, "h": 0.2}},
            {"confidence": 0.95, "bbox": {"cx": 0.7, "cy": 0.4, "w": 0.3, "h": 0.3}},
        ],
    }

    result = vision_v2._normalize_detection_with_position(parsed, raw_text=json.dumps(parsed))

    assert result["bbox"] == {"cx": 0.7, "cy": 0.4, "w": 0.3, "h": 0.3}
    assert result["x_center"] == pytest.approx(0.7)
    assert result["in_frame_offset_deg"] == pytest.approx((0.7 - 0.5) * vision_v2.CAMERA_HFOV_DEG)


def test_normalize_detection_with_position_falls_back_to_x_center():
    parsed = {
        "matched": True,
        "confidence": 0.8,
        "evidence": "legacy response",
        "x_center": 0.1,
    }

    result = vision_v2._normalize_detection_with_position(parsed, raw_text=json.dumps(parsed))

    assert result["bbox"] == {"cx": 0.1, "cy": 0.5, "w": 0.0, "h": 0.0}
    assert result["x_center"] == pytest.approx(0.1)
