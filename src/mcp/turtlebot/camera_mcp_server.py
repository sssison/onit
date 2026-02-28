"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import asyncio
import atexit
import base64
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import rclpy
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


CAMERA_TOPIC = os.getenv("CAMERA_TOPIC", "camera/image_raw/compressed")
NODE_NAME = os.getenv("CAMERA_NODE_NAME", "camera_mcp_server_node")
FRAME_LOG_EVERY = max(1, int(_env_float("CAMERA_FRAME_LOG_EVERY", 30)))

mcp_camera = FastMCP("TurtleBot Camera Vision MCP")


DEFAULT_FRAME_MAX_BYTES = max(1, int(_env_float("CAMERA_FRAME_MAX_BYTES", 400_000)))
INITIAL_FRAME_WAIT_S = 2.0
DEFAULT_CURRENT_VIEW_WAIT_S = _env_float("CAMERA_CURRENT_VIEW_WAIT_S", 1.0)


def _ensure_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _ensure_non_negative(name: str, value: float) -> float:
    parsed = _ensure_finite(name, value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _normalize_format(fmt: str | None) -> str:
    if not fmt:
        return ""
    return fmt.strip().lower()


def _mime_from_format(fmt: str | None) -> str:
    normalized = _normalize_format(fmt)
    if "png" in normalized:
        return "image/png"
    if "jpg" in normalized or "jpeg" in normalized:
        return "image/jpeg"
    if "webp" in normalized:
        return "image/webp"
    return "application/octet-stream"


def _extension_from_format(fmt: str | None) -> str:
    mime = _mime_from_format(fmt)
    if mime == "image/png":
        return "png"
    if mime == "image/webp":
        return "webp"
    return "jpg"


class CompressedCameraSubscriber(Node):
    def __init__(self, topic_name: str) -> None:
        super().__init__(NODE_NAME)
        self._topic_name = topic_name
        self._lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._frame_count = 0
        self._latest_bytes: bytes | None = None
        self._latest_format = ""
        self._latest_stamp_s: float | None = None
        self._latest_rx_wall_s: float | None = None
        self._latest_rx_mono_s: float | None = None

        self.create_subscription(
            CompressedImage,
            self._topic_name,
            self._on_message,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Subscribed to topic: {self._topic_name}")
        logger.info("Camera subscriber ready topic=%s node_name=%s", self._topic_name, NODE_NAME)

    def _on_message(self, msg: CompressedImage) -> None:
        now_wall = time.time()
        now_mono = time.monotonic()
        stamp_s = None
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            stamp_s = float(msg.header.stamp.sec) + (float(msg.header.stamp.nanosec) / 1_000_000_000.0)

        with self._frame_condition:
            self._latest_bytes = bytes(msg.data)
            self._latest_format = msg.format
            self._latest_stamp_s = stamp_s
            self._latest_rx_wall_s = now_wall
            self._latest_rx_mono_s = now_mono
            self._frame_count += 1
            self._frame_condition.notify_all()
            if self._frame_count == 1 or self._frame_count % FRAME_LOG_EVERY == 0:
                logger.debug(
                    "Received frame count=%d bytes=%d format=%s stamp_s=%s",
                    self._frame_count,
                    len(self._latest_bytes),
                    self._latest_format,
                    self._latest_stamp_s,
                )

    def snapshot(self) -> dict[str, Any]:
        with self._frame_condition:
            frame_present = self._latest_bytes is not None
            frame_size = len(self._latest_bytes) if self._latest_bytes is not None else 0
            frame_age = None
            if self._latest_rx_mono_s is not None:
                frame_age = max(0.0, time.monotonic() - self._latest_rx_mono_s)
            return {
                "topic": self._topic_name,
                "frame_present": frame_present,
                "frame_count": self._frame_count,
                "latest_frame_bytes": frame_size,
                "latest_format": self._latest_format,
                "latest_stamp_s": self._latest_stamp_s,
                "latest_received_unix_s": self._latest_rx_wall_s,
                "latest_age_s": frame_age,
            }

    def latest_frame(self) -> dict[str, Any]:
        with self._frame_condition:
            if self._latest_bytes is None:
                raise RuntimeError(
                    f"No frame received yet on topic '{self._topic_name}'. "
                    "Use camera_wait_for_frame() or confirm the publisher is active."
                )
            frame_age = None
            if self._latest_rx_mono_s is not None:
                frame_age = max(0.0, time.monotonic() - self._latest_rx_mono_s)
            return {
                "bytes": self._latest_bytes,
                "format": self._latest_format,
                "stamp_s": self._latest_stamp_s,
                "received_unix_s": self._latest_rx_wall_s,
                "age_s": frame_age,
                "frame_count": self._frame_count,
                "topic": self._topic_name,
            }

    def wait_for_frame(self, timeout_s: float, min_frame_count: int = 0) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._frame_condition:
            while self._frame_count <= min_frame_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._frame_condition.wait(timeout=remaining)
            return True


_node_ready = threading.Event()
_ros_thread: Optional[threading.Thread] = None
_camera_node: Optional[CompressedCameraSubscriber] = None
_spin_error: Optional[BaseException] = None


def _ros_spin_worker() -> None:
    global _camera_node, _spin_error
    logger.info("Starting ROS spin worker topic=%s", CAMERA_TOPIC)
    try:
        if not rclpy.ok():
            rclpy.init()
        _camera_node = CompressedCameraSubscriber(CAMERA_TOPIC)
        _node_ready.set()
        rclpy.spin(_camera_node)
    except BaseException as exc:
        _spin_error = exc
        logger.exception("ROS spin worker failed: %s", exc)
        _node_ready.set()
    finally:
        if _camera_node is not None:
            _camera_node.destroy_node()
            _camera_node = None
        if rclpy.ok():
            rclpy.shutdown()
        logger.info("ROS spin worker stopped")


def _start_ros_thread() -> None:
    global _ros_thread, _spin_error
    if _ros_thread is not None and _ros_thread.is_alive():
        logger.debug("ROS thread already running")
        return
    _spin_error = None
    _node_ready.clear()
    _ros_thread = threading.Thread(target=_ros_spin_worker, name="camera-mcp-ros-spin", daemon=True)
    _ros_thread.start()
    logger.debug("ROS thread launched name=%s", _ros_thread.name)


def _get_camera_node() -> CompressedCameraSubscriber:
    _start_ros_thread()
    if not _node_ready.wait(timeout=3.0):
        raise RuntimeError("Camera ROS node did not start within 3 seconds.")
    if _spin_error is not None:
        raise RuntimeError(f"Camera ROS node failed to start: {_spin_error}") from _spin_error
    if _camera_node is None:
        raise RuntimeError("Camera ROS node unavailable.")
    return _camera_node


def _shutdown_ros() -> None:
    if rclpy.ok():
        logger.info("Shutting down ROS context")
        rclpy.shutdown()


atexit.register(_shutdown_ros)


async def _camera_status() -> dict[str, Any]:
    node = _get_camera_node()
    snapshot = node.snapshot()
    status = "online" if snapshot["frame_present"] else "waiting_for_frames"
    return {"status": status, **snapshot}


@mcp_camera.tool(
    description=(
        "Wait for a newer TurtleBot camera frame before visual tasks such as "
        "describing what is visible, locating objects, or reading scene text."
    ),
    annotations={
        "readOnlyHint": True,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def camera_wait_for_frame(timeout_s: float = 5.0) -> dict[str, Any]:
    """Wait up to timeout_s for a frame, then return camera health."""
    timeout_value = _ensure_finite("timeout_s", timeout_s)
    if timeout_value <= 0:
        raise ValueError("timeout_s must be > 0")
    logger.info("Tool camera_wait_for_frame called timeout_s=%.3f", timeout_value)
    node = _get_camera_node()
    initial_frame_count = node.snapshot()["frame_count"]
    frame_arrived = await asyncio.to_thread(node.wait_for_frame, timeout_value, initial_frame_count)
    logger.debug(
        "camera_wait_for_frame frame_arrived=%s initial_frame_count=%d",
        frame_arrived,
        initial_frame_count,
    )
    return await _camera_status()


@mcp_camera.tool(
    title="Get Current Camera View",
    description=(
        "Primary camera tool for scene understanding. Returns the current TurtleBot "
        "camera image as MCP ImageContent for visual tasks like 'describe what you "
        "see', finding objects, and reading signs/text."
    ),
    annotations={
        "readOnlyHint": True,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def camera_get_latest_frame(
    max_bytes: int = DEFAULT_FRAME_MAX_BYTES,
    as_data_url: bool = False,
    include_base64: bool = False,
    wait_for_new_frame: bool = True,
    wait_timeout_s: float = DEFAULT_CURRENT_VIEW_WAIT_S,
) -> dict[str, Any]:
    """
    Return a compressed camera frame with image content for vision models.

    By default this tries to wait briefly for a newer frame so the result reflects the current view.
    If frame size exceeds max_bytes, this returns an error so callers can raise the limit explicitly.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    wait_timeout_value = _ensure_non_negative("wait_timeout_s", wait_timeout_s)
    logger.info(
        "Tool camera_get_latest_frame called max_bytes=%d as_data_url=%s include_base64=%s wait_for_new_frame=%s wait_timeout_s=%.3f",
        max_bytes,
        as_data_url,
        include_base64,
        wait_for_new_frame,
        wait_timeout_value,
    )

    node = _get_camera_node()
    snapshot = node.snapshot()
    initial_frame_count = snapshot["frame_count"]
    waited_for_new_frame = False
    new_frame_arrived = False

    if not snapshot["frame_present"]:
        waited_for_new_frame = True
        first_wait_s = max(INITIAL_FRAME_WAIT_S, wait_timeout_value)
        logger.info("camera_get_latest_frame waiting up to %.3fs for first frame", first_wait_s)
        new_frame_arrived = await asyncio.to_thread(node.wait_for_frame, first_wait_s, initial_frame_count)
        logger.debug("camera_get_latest_frame first-frame wait new_frame_arrived=%s", new_frame_arrived)
    elif wait_for_new_frame and wait_timeout_value > 0:
        waited_for_new_frame = True
        new_frame_arrived = await asyncio.to_thread(node.wait_for_frame, wait_timeout_value, initial_frame_count)
        logger.debug(
            "camera_get_latest_frame current-view wait new_frame_arrived=%s initial_frame_count=%d",
            new_frame_arrived,
            initial_frame_count,
        )

    frame = node.latest_frame()
    frame_bytes: bytes = frame["bytes"]
    frame_size = len(frame_bytes)
    if frame_size > max_bytes:
        logger.warning("camera_get_latest_frame rejected frame_size=%d max_bytes=%d", frame_size, max_bytes)
        raise RuntimeError(
            f"Latest frame is {frame_size} bytes, which exceeds max_bytes={max_bytes}. "
            "Call again with a larger max_bytes or use camera_save_latest_frame()."
        )

    mime_type = _mime_from_format(frame["format"])
    encoded = base64.b64encode(frame_bytes).decode("ascii")

    response: dict[str, Any] = {
        "topic": frame["topic"],
        "frame_count": frame["frame_count"],
        "frame_bytes": frame_size,
        "format": frame["format"],
        "mime_type": mime_type,
        "stamp_s": frame["stamp_s"],
        "received_unix_s": frame["received_unix_s"],
        "age_s": frame["age_s"],
        "waited_for_new_frame": waited_for_new_frame,
        "new_frame_arrived": new_frame_arrived,
        "wait_timeout_s": wait_timeout_value,
        "wait_for_new_frame": wait_for_new_frame,
        "initial_frame_count": initial_frame_count,
        "frame_is_newer_than_request_start": frame["frame_count"] > initial_frame_count,
    }
    if include_base64:
        response["image_base64"] = encoded
    if as_data_url:
        response["image_data_url"] = f"data:{mime_type};base64,{encoded}"
    logger.debug("camera_get_latest_frame returned frame_bytes=%d mime_type=%s", frame_size, mime_type)

    return ToolResult(
        content=[
            ImageContent(type="image", data=encoded, mimeType=mime_type),
            TextContent(
                type="text",
                text=(
                    f"Latest camera frame from topic '{frame['topic']}' "
                    f"(frame_count={frame['frame_count']}, bytes={frame_size}, mime_type={mime_type}, "
                    f"newer_frame={response['frame_is_newer_than_request_start']})."
                ),
            ),
        ],
        structured_content=response,
    )


@mcp_camera.tool()
async def camera_save_latest_frame(path: Optional[str] = None, overwrite: bool = False) -> dict[str, Any]:
    """Save the latest compressed frame to disk and return the saved path."""
    logger.info("Tool camera_save_latest_frame called path=%s overwrite=%s", path, overwrite)
    node = _get_camera_node()
    frame = node.latest_frame()
    frame_bytes: bytes = frame["bytes"]
    ext = _extension_from_format(frame["format"])

    if path is None or path.strip() == "":
        target_path = Path("/tmp") / f"camera_frame_{int(time.time() * 1000)}.{ext}"
    else:
        target_path = Path(path).expanduser()
        if not target_path.is_absolute():
            target_path = Path.cwd() / target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        logger.warning("camera_save_latest_frame target exists and overwrite=false path=%s", target_path)
        raise RuntimeError(f"File already exists: {target_path}. Set overwrite=true to replace it.")

    target_path.write_bytes(frame_bytes)
    logger.debug("camera_save_latest_frame wrote path=%s bytes=%d", target_path, len(frame_bytes))

    return {
        "saved_path": str(target_path.resolve()),
        "topic": frame["topic"],
        "frame_count": frame["frame_count"],
        "frame_bytes": len(frame_bytes),
        "format": frame["format"],
        "mime_type": _mime_from_format(frame["format"]),
        "stamp_s": frame["stamp_s"],
        "received_unix_s": frame["received_unix_s"],
        "age_s": frame["age_s"],
    }


def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18202,
    path: str = "/camera",
    options: dict = {},
) -> None:
    """Run the Camera MCP server."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting Camera MCP server topic=%s node_name=%s frame_max_bytes=%d frame_log_every=%d at %s:%s%s",
        CAMERA_TOPIC,
        NODE_NAME,
        DEFAULT_FRAME_MAX_BYTES,
        FRAME_LOG_EVERY,
        host,
        port,
        path,
    )
    _start_ros_thread()
    mcp_camera.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    _start_ros_thread()
    mcp_camera.run()
