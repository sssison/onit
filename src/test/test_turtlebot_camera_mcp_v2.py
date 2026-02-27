"""Tests for src/mcp/turtlebot_v2/camera_mcp_server.py."""

import asyncio
import os
import sys
from unittest.mock import patch

import pytest
from fastmcp.tools import ToolResult
from mcp.types import ImageContent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v2 import camera_mcp_server as camera_v2

PIL = pytest.importorskip("PIL")


def _make_image_bytes(fmt="PNG", size=(64, 48), color=(255, 0, 0)):
    from io import BytesIO
    from PIL import Image

    image = Image.new("RGB", size, color)
    buf = BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


class _FakeCameraNode:
    def __init__(self, frame_bytes, frame_present=True, frame_count=3, wait_result=True, topic="/camera/image_raw/compressed"):
        self._frame_bytes = frame_bytes
        self._frame_present = frame_present
        self._frame_count = frame_count
        self._wait_result = wait_result
        self._topic = topic

    def snapshot(self):
        return {
            "topic": self._topic,
            "frame_present": self._frame_present,
            "frame_count": self._frame_count,
            "latest_frame_bytes": len(self._frame_bytes) if self._frame_present else 0,
            "latest_format": "jpeg",
            "latest_stamp_s": 123.0,
            "latest_received_unix_s": 1700000000.0,
            "latest_age_s": 0.05,
        }

    def latest_frame(self):
        if not self._frame_present:
            raise RuntimeError("No frame received yet on topic")
        return {
            "bytes": self._frame_bytes,
            "format": "jpeg",
            "stamp_s": 123.0,
            "received_unix_s": 1700000000.0,
            "age_s": 0.05,
            "frame_count": self._frame_count,
            "topic": self._topic,
        }

    def wait_for_frame(self, timeout_s, min_frame_count=0):
        return self._wait_result


def test_get_decoded_frame_success():
    source_bytes = _make_image_bytes(fmt="PNG", size=(80, 60))
    fake_node = _FakeCameraNode(frame_bytes=source_bytes, frame_present=True, frame_count=9)

    with patch.object(camera_v2, "_get_camera_node", return_value=fake_node):
        result = asyncio.run(
            camera_v2.tbot_camera_get_decoded_frame(
                wait_for_new_frame=False,
                wait_timeout_s=0.0,
                max_bytes=1_500_000,
            )
        )

    assert isinstance(result, ToolResult)
    assert isinstance(result.content[0], ImageContent)
    assert result.content[0].mimeType == "image/jpeg"

    payload = result.structured_content
    assert payload["width"] == 60
    assert payload["height"] == 80
    assert payload["mime_type"] == "image/jpeg"
    assert payload["frame_bytes"] > 0
    assert payload["frame_count"] == 9


def test_get_decoded_frame_rejects_oversized_output():
    source_bytes = _make_image_bytes(fmt="PNG", size=(120, 120))
    fake_node = _FakeCameraNode(frame_bytes=source_bytes, frame_present=True, frame_count=5)

    with patch.object(camera_v2, "_get_camera_node", return_value=fake_node):
        with pytest.raises(RuntimeError, match="exceeds max_bytes"):
            asyncio.run(
                camera_v2.tbot_camera_get_decoded_frame(
                    wait_for_new_frame=False,
                    wait_timeout_s=0.0,
                    max_bytes=100,
                )
            )


def test_get_decoded_frame_no_frame_after_wait_raises():
    fake_node = _FakeCameraNode(frame_bytes=b"", frame_present=False, frame_count=0, wait_result=False)

    with patch.object(camera_v2, "_get_camera_node", return_value=fake_node):
        with pytest.raises(RuntimeError, match="No frame received yet on topic"):
            asyncio.run(
                camera_v2.tbot_camera_get_decoded_frame(
                    wait_for_new_frame=True,
                    wait_timeout_s=0.1,
                    max_bytes=1_500_000,
                )
            )


class _FakeMCPClient:
    def __init__(self, url, vision_results, motion_calls):
        self._url = url
        self._vision_results = vision_results
        self._motion_calls = motion_calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def call_tool(self, name, payload):
        if self._url == camera_v2.VISION_MCP_URL:
            assert name == "tbot_vision_analyze_frames_for_object"
            if not self._vision_results:
                raise AssertionError("No mocked vision responses left")
            return self._vision_results.pop(0)
        if self._url == camera_v2.MOTION_MCP_URL:
            self._motion_calls.append((name, payload))
            return {"status": "ok"}
        raise AssertionError(f"Unexpected MCP URL in test: {self._url}")


def test_reorient_to_object_uses_bbox_center_and_reports_ready():
    source_bytes = _make_image_bytes(fmt="PNG", size=(80, 60))
    fake_node = _FakeCameraNode(frame_bytes=source_bytes, frame_present=True, frame_count=9)
    vision_results = [
        {
            "status": "found",
            "bbox": {"cx": 0.8, "cy": 0.5, "w": 0.2, "h": 0.2},
            "in_frame_offset_deg": 999.0,
        },
        {
            "status": "found",
            "bbox": {"cx": 0.51, "cy": 0.5, "w": 0.2, "h": 0.2},
            "in_frame_offset_deg": 999.0,
        },
    ]
    motion_calls = []

    def fake_client(url):
        return _FakeMCPClient(url, vision_results, motion_calls)

    with patch.object(camera_v2, "_get_camera_node", return_value=fake_node), patch.object(
        camera_v2, "Client", side_effect=fake_client
    ), patch.object(camera_v2, "REORIENT_MAX_STEP_DEG", 30.0), patch.object(
        camera_v2, "REORIENT_CORRECTION_SIGN", -1.0
    ):
        result = asyncio.run(camera_v2.tbot_camera_reorient_to_object("bottle", threshold_deg=8.0, max_iterations=4))

    assert result["status"] == "centered"
    assert result["reason"] == "centered"
    assert result["ready_to_approach"] is True
    assert result["final_bbox"]["cx"] == pytest.approx(0.51)
    assert len(motion_calls) == 1
    assert motion_calls[0][0] == "tbot_motion_scan_rotate"
    expected_cmd = (0.8 - 0.5) * camera_v2.CAMERA_HFOV_DEG * -1.0
    assert motion_calls[0][1]["degrees"] == pytest.approx(expected_cmd)


def test_reorient_to_object_clamps_large_correction():
    source_bytes = _make_image_bytes(fmt="PNG", size=(80, 60))
    fake_node = _FakeCameraNode(frame_bytes=source_bytes, frame_present=True, frame_count=9)
    vision_results = [
        {"status": "found", "bbox": {"cx": 1.0, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"status": "found", "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
    ]
    motion_calls = []

    def fake_client(url):
        return _FakeMCPClient(url, vision_results, motion_calls)

    with patch.object(camera_v2, "_get_camera_node", return_value=fake_node), patch.object(
        camera_v2, "Client", side_effect=fake_client
    ), patch.object(camera_v2, "REORIENT_MAX_STEP_DEG", 10.0), patch.object(
        camera_v2, "REORIENT_CORRECTION_SIGN", -1.0
    ):
        result = asyncio.run(camera_v2.tbot_camera_reorient_to_object("bottle", threshold_deg=8.0, max_iterations=4))

    assert result["status"] == "centered"
    assert len(motion_calls) == 1
    assert motion_calls[0][1]["degrees"] == pytest.approx(-10.0)


def test_reorient_to_object_stops_on_no_progress():
    source_bytes = _make_image_bytes(fmt="PNG", size=(80, 60))
    fake_node = _FakeCameraNode(frame_bytes=source_bytes, frame_present=True, frame_count=9)
    vision_results = [
        {"status": "found", "bbox": {"cx": 0.80, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"status": "found", "bbox": {"cx": 0.82, "cy": 0.5, "w": 0.2, "h": 0.2}},
        {"status": "found", "bbox": {"cx": 0.84, "cy": 0.5, "w": 0.2, "h": 0.2}},
    ]
    motion_calls = []

    def fake_client(url):
        return _FakeMCPClient(url, vision_results, motion_calls)

    with patch.object(camera_v2, "_get_camera_node", return_value=fake_node), patch.object(
        camera_v2, "Client", side_effect=fake_client
    ), patch.object(camera_v2, "REORIENT_MAX_STEP_DEG", 30.0), patch.object(
        camera_v2, "REORIENT_CORRECTION_SIGN", -1.0
    ), patch.object(camera_v2, "REORIENT_IMPROVEMENT_EPS_DEG", 0.5), patch.object(
        camera_v2, "REORIENT_MAX_NO_PROGRESS", 2
    ):
        result = asyncio.run(camera_v2.tbot_camera_reorient_to_object("bottle", threshold_deg=3.0, max_iterations=6))

    assert result["status"] == "no_progress"
    assert result["reason"] == "no_progress"
    assert result["ready_to_approach"] is False
    assert len(motion_calls) == 2
