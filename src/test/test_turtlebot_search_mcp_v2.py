"""Tests for src/mcp/turtlebot_v2/search_mcp_server.py."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v2 import search_mcp_server as search_v2


class _DummyClient:
    def __init__(self, _url):
        self.url = _url

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _base_patches(call_tool_side_effect, detect_side_effect=None):
    patches = [
        patch.object(search_v2, "Client", side_effect=lambda url: _DummyClient(url)),
        patch.object(search_v2, "_call_mcp_tool", side_effect=call_tool_side_effect),
        patch.object(search_v2, "AsyncOpenAI", return_value=MagicMock()),
        patch("src.mcp.turtlebot_v2.search_mcp_server.asyncio.sleep", new=AsyncMock(return_value=None)),
    ]
    if detect_side_effect is not None:
        patches.append(
            patch.object(search_v2, "_detect_target_with_vision", side_effect=detect_side_effect)
        )
    return patches


def _run_with_patches(coro, patches):
    with patches[0], patches[1], patches[2], patches[3]:
        if len(patches) > 4:
            with patches[4]:
                return asyncio.run(coro)
        return asyncio.run(coro)


def test_search_found_path_stops_early():
    calls = {"move": 0, "stop": 0, "camera": 0}
    detections = [
        {"matched": False, "confidence": 0.2, "evidence": "", "raw_response": ""},
        {"matched": False, "confidence": 0.4, "evidence": "", "raw_response": ""},
        {"matched": True, "confidence": 0.9, "evidence": "target seen", "raw_response": ""},
    ]

    async def fake_call_tool(_client, tool_name, _args):
        if tool_name == "tbot_motion_move":
            calls["move"] += 1
            return {"angular_sign": 1}
        if tool_name == "tbot_motion_stop":
            calls["stop"] += 1
            return {"status": "stopped"}
        if tool_name == "tbot_camera_get_decoded_frame":
            calls["camera"] += 1
            return {"image_base64": "abcd"}
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    async def fake_detect(**_kwargs):
        return detections.pop(0)

    result = _run_with_patches(
        search_v2.tbot_search_while_rotating(
            target="camera",
            timeout_s=5.0,
            sample_period_s=0.01,
            max_frames=20,
            stop_on_collision=False,
            detection_confidence_threshold=0.65,
        ),
        _base_patches(fake_call_tool, fake_detect),
    )

    assert result["status"] == "found"
    assert result["found"] is True
    assert result["frames_processed"] == 3
    assert result["best_frame_index"] == 3
    assert result["best_confidence"] == 0.9
    assert calls["move"] == 1
    assert calls["stop"] == 1
    assert calls["camera"] == 3


def test_search_timeout_path_calls_stop():
    calls = {"stop": 0}

    async def fake_call_tool(_client, tool_name, _args):
        if tool_name == "tbot_motion_move":
            return {"angular_sign": 1}
        if tool_name == "tbot_motion_stop":
            calls["stop"] += 1
            return {"status": "stopped"}
        if tool_name == "tbot_camera_get_decoded_frame":
            return {"image_base64": "abcd"}
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    async def fake_detect(**_kwargs):
        return {"matched": False, "confidence": 0.3, "evidence": "", "raw_response": ""}

    result = _run_with_patches(
        search_v2.tbot_search_while_rotating(
            target="camera",
            timeout_s=10.0,
            sample_period_s=0.01,
            max_frames=4,
            stop_on_collision=False,
        ),
        _base_patches(fake_call_tool, fake_detect),
    )

    assert result["status"] == "timeout"
    assert result["found"] is False
    assert result["stop_reason"] == "max_frames_reached"
    assert result["frames_processed"] == 4
    assert calls["stop"] == 1


def test_search_collision_stop_path():
    calls = {"stop": 0}

    async def fake_call_tool(_client, tool_name, _args):
        if tool_name == "tbot_motion_move":
            return {"angular_sign": -1}
        if tool_name == "tbot_motion_stop":
            calls["stop"] += 1
            return {"status": "stopped"}
        if tool_name == "tbot_lidar_check_collision":
            return {"risk_level": "stop", "scan_age_s": 0.05}
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    result = _run_with_patches(
        search_v2.tbot_search_while_rotating(
            target="camera",
            timeout_s=5.0,
            sample_period_s=0.01,
            max_frames=10,
            stop_on_collision=True,
        ),
        _base_patches(fake_call_tool, detect_side_effect=None),
    )

    assert result["status"] == "stopped_collision"
    assert result["found"] is False
    assert result["frames_processed"] == 0
    assert len(result["collision_events"]) == 1
    assert calls["stop"] == 1


def test_search_recovers_from_transient_errors():
    camera_calls = {"count": 0}

    async def fake_call_tool(_client, tool_name, _args):
        if tool_name == "tbot_motion_move":
            return {"angular_sign": 1}
        if tool_name == "tbot_motion_stop":
            return {"status": "stopped"}
        if tool_name == "tbot_camera_get_decoded_frame":
            camera_calls["count"] += 1
            if camera_calls["count"] == 1:
                raise RuntimeError("camera transient")
            return {"image_base64": "abcd"}
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    detections = [
        {"matched": False, "confidence": 0.2, "evidence": "", "raw_response": ""},
        {"matched": True, "confidence": 0.95, "evidence": "seen", "raw_response": ""},
    ]

    async def fake_detect(**_kwargs):
        return detections.pop(0)

    result = _run_with_patches(
        search_v2.tbot_search_while_rotating(
            target="camera",
            timeout_s=10.0,
            sample_period_s=0.01,
            max_frames=10,
            stop_on_collision=False,
        ),
        _base_patches(fake_call_tool, fake_detect),
    )

    assert result["status"] == "found"
    assert result["found"] is True
    assert result["frames_processed"] == 2
    assert any(item["matched"] is False and item["confidence"] == 0.0 for item in result["detection_history"])


def test_search_hard_error_after_consecutive_failures():
    calls = {"stop": 0}

    async def fake_call_tool(_client, tool_name, _args):
        if tool_name == "tbot_motion_move":
            return {"angular_sign": 1}
        if tool_name == "tbot_motion_stop":
            calls["stop"] += 1
            return {"status": "stopped"}
        if tool_name == "tbot_camera_get_decoded_frame":
            raise RuntimeError("camera failed")
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    result = _run_with_patches(
        search_v2.tbot_search_while_rotating(
            target="camera",
            timeout_s=10.0,
            sample_period_s=0.01,
            max_frames=100,
            stop_on_collision=False,
        ),
        _base_patches(fake_call_tool, detect_side_effect=None),
    )

    assert result["status"] == "error"
    assert result["found"] is False
    assert result["frames_processed"] == 0
    assert result["stop_reason"].startswith("consecutive_failures:")
    assert calls["stop"] == 1


def test_search_response_schema_and_history_bound():
    async def fake_call_tool(_client, tool_name, _args):
        if tool_name == "tbot_motion_move":
            return {"angular_sign": 1}
        if tool_name == "tbot_motion_stop":
            return {"status": "stopped"}
        if tool_name == "tbot_camera_get_decoded_frame":
            return {"image_base64": "abcd"}
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    async def fake_detect(**_kwargs):
        return {"matched": False, "confidence": 0.1, "evidence": "", "raw_response": ""}

    result = _run_with_patches(
        search_v2.tbot_search_while_rotating(
            target="camera",
            timeout_s=20.0,
            sample_period_s=0.0 + 0.01,
            max_frames=30,
            stop_on_collision=False,
        ),
        _base_patches(fake_call_tool, fake_detect),
    )

    required_keys = {
        "status",
        "target",
        "found",
        "best_confidence",
        "best_frame_index",
        "frames_processed",
        "elapsed_s",
        "stop_reason",
        "detection_history",
        "collision_events",
        "motion",
        "final_action",
    }
    assert required_keys.issubset(result.keys())
    assert len(result["detection_history"]) <= search_v2.MAX_DETECTION_HISTORY

