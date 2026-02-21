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
    assert payload["width"] == 80
    assert payload["height"] == 60
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


def test_wait_for_frame_timeout_no_frame():
    fake_node = _FakeCameraNode(frame_bytes=b"", frame_present=False, frame_count=0, wait_result=False)

    with patch.object(camera_v2, "_get_camera_node", return_value=fake_node):
        result = asyncio.run(camera_v2.tbot_camera_wait_for_frame(timeout_s=0.2))

    assert result["frame_arrived"] is False
    assert result["status"] == "waiting_for_frames"
    assert result["frame_present"] is False


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
