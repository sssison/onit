"""
TurtleBot Camera MCP Server V2.

Subscribes to /camera/image_raw/compressed, decodes frames, and returns a
normalized JPEG image payload for downstream tools.
"""

import asyncio
import atexit
import base64
import io
import json
import logging
import math
import os
import threading
import time
from typing import Any, Optional

from fastmcp import Client, FastMCP
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent

try:
    from PIL import Image, UnidentifiedImageError

    PIL_AVAILABLE = True
    PIL_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:
    PIL_AVAILABLE = False
    PIL_IMPORT_ERROR = e
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = Exception  # type: ignore[assignment]

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    RCLPY_AVAILABLE = True
    RCLPY_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment]
    qos_profile_sensor_data = None  # type: ignore[assignment]
    CompressedImage = object  # type: ignore[assignment]
    RCLPY_AVAILABLE = False
    RCLPY_IMPORT_ERROR = e

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected an integer") from e


CAMERA_TOPIC = os.getenv("CAMERA_TOPIC", "/camera/image_raw/compressed")
NODE_NAME = os.getenv("CAMERA_NODE_NAME", "camera_mcp_server_node_v2")
MOTION_MCP_URL = os.getenv("TBOT_MOTION_MCP_URL", "http://127.0.0.1:18205/turtlebot-motion-v2")
VISION_MCP_URL = os.getenv("TBOT_VISION_MCP_URL", "http://127.0.0.1:18207/turtlebot-vision-v2")
REORIENT_THRESHOLD_DEG = _env_float("REORIENT_THRESHOLD_DEG", 8.0)
REORIENT_MAX_ITERATIONS = 6
CAMERA_HFOV_DEG = _env_float("CAMERA_HFOV_DEG", 62.0)
REORIENT_CORRECTION_SIGN = _env_float("REORIENT_CORRECTION_SIGN", -1.0)
REORIENT_MAX_STEP_DEG = abs(_env_float("REORIENT_MAX_STEP_DEG", 15.0))
REORIENT_IMPROVEMENT_EPS_DEG = max(0.0, _env_float("REORIENT_IMPROVEMENT_EPS_DEG", 1.0))
REORIENT_MAX_NO_PROGRESS = max(1, _env_int("REORIENT_MAX_NO_PROGRESS", 2))
FRAME_LOG_EVERY = max(1, int(_env_float("CAMERA_FRAME_LOG_EVERY", 30)))

DEFAULT_FRAME_MAX_BYTES = max(1, int(_env_float("CAMERA_FRAME_MAX_BYTES", 1_500_000)))
INITIAL_FRAME_WAIT_S = 2.0
DEFAULT_CURRENT_VIEW_WAIT_S = _env_float("CAMERA_CURRENT_VIEW_WAIT_S", 1.0)

mcp_camera_v2 = FastMCP("TurtleBot Camera MCP Server V2")


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


def _decode_and_reencode_jpeg(compressed_bytes: bytes) -> tuple[bytes, int, int, str]:
    if not PIL_AVAILABLE:
        raise RuntimeError(f"Pillow is required for decoding compressed frames: {PIL_IMPORT_ERROR}")

    try:
        with Image.open(io.BytesIO(compressed_bytes)) as image:
            image.load()
            width, height = image.size
            mode = image.mode

            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            image = image.rotate(-90, expand=True)
            width, height = image.size

            encoded_buffer = io.BytesIO()
            image.save(encoded_buffer, format="JPEG", quality=90, optimize=False)
            encoded_bytes = encoded_buffer.getvalue()
            return encoded_bytes, int(width), int(height), str(mode)
    except UnidentifiedImageError as e:
        raise RuntimeError("Compressed frame could not be decoded as an image.") from e


if RCLPY_AVAILABLE:

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
                        "Confirm the camera publisher is active or check tbot_camera_health()."
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

else:

    class CompressedCameraSubscriber:  # pragma: no cover - fallback stub
        def __init__(self, topic_name: str) -> None:
            raise RuntimeError(
                f"rclpy is required for camera MCP server but is unavailable: {RCLPY_IMPORT_ERROR}"
            )


_node_ready = threading.Event()
_ros_thread: Optional[threading.Thread] = None
_camera_node: Optional[CompressedCameraSubscriber] = None
_spin_error: Optional[BaseException] = None


def _ros_spin_worker() -> None:
    global _camera_node, _spin_error
    if not RCLPY_AVAILABLE:
        _spin_error = RuntimeError(f"rclpy unavailable: {RCLPY_IMPORT_ERROR}")
        _node_ready.set()
        return

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


def _start_ros_thread() -> None:
    global _ros_thread, _spin_error
    if _ros_thread is not None and _ros_thread.is_alive():
        return
    _spin_error = None
    _node_ready.clear()
    _ros_thread = threading.Thread(target=_ros_spin_worker, name="camera-mcp-v2-ros-spin", daemon=True)
    _ros_thread.start()


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
    if RCLPY_AVAILABLE and rclpy.ok():
        rclpy.shutdown()


atexit.register(_shutdown_ros)


@mcp_camera_v2.tool()
async def tbot_camera_health() -> dict[str, Any]:
    """Return current subscriber status for /camera/image_raw/compressed."""
    try:
        node = _get_camera_node()
    except Exception as e:
        return {
            "status": "offline",
            "topic": CAMERA_TOPIC,
            "ros_available": RCLPY_AVAILABLE,
            "error": str(e),
        }

    snapshot = node.snapshot()
    status = "online" if snapshot["frame_present"] else "waiting_for_frames"
    return {"status": status, "ros_available": RCLPY_AVAILABLE, **snapshot}


@mcp_camera_v2.tool(
    title="Get Decoded TurtleBot Camera Frame",
    description=(
        "Fetch latest compressed frame, decode it, and return normalized JPEG "
        "ImageContent plus structured metadata."
    ),
)
async def tbot_camera_get_decoded_frame(
    wait_for_new_frame: bool = True,
    wait_timeout_s: float = DEFAULT_CURRENT_VIEW_WAIT_S,
    max_bytes: int = DEFAULT_FRAME_MAX_BYTES,
    include_base64: bool = False,
) -> ToolResult:
    """Return the latest camera frame as decoded JPEG image content."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    wait_timeout_value = _ensure_non_negative("wait_timeout_s", wait_timeout_s)

    node = _get_camera_node()
    snapshot = node.snapshot()
    initial_frame_count = snapshot["frame_count"]
    waited_for_new_frame = False
    new_frame_arrived = False

    if not snapshot["frame_present"]:
        waited_for_new_frame = True
        first_wait_s = max(INITIAL_FRAME_WAIT_S, wait_timeout_value)
        new_frame_arrived = await asyncio.to_thread(node.wait_for_frame, first_wait_s, initial_frame_count)
    elif wait_for_new_frame and wait_timeout_value > 0:
        waited_for_new_frame = True
        new_frame_arrived = await asyncio.to_thread(node.wait_for_frame, wait_timeout_value, initial_frame_count)

    frame = node.latest_frame()
    compressed_bytes: bytes = frame["bytes"]
    encoded_bytes, width, height, mode = _decode_and_reencode_jpeg(compressed_bytes)

    frame_size = len(encoded_bytes)
    if frame_size > max_bytes:
        raise RuntimeError(
            f"Decoded frame is {frame_size} bytes, which exceeds max_bytes={max_bytes}. "
            "Increase max_bytes and try again."
        )

    encoded = base64.b64encode(encoded_bytes).decode("ascii")
    mime_type = "image/jpeg"

    response: dict[str, Any] = {
        "topic": frame["topic"],
        "frame_count": frame["frame_count"],
        "width": width,
        "height": height,
        "mode": mode,
        "mime_type": mime_type,
        "frame_bytes": frame_size,
        "compressed_frame_bytes": len(compressed_bytes),
        "compressed_format": frame["format"],
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

    return ToolResult(
        content=[
            ImageContent(type="image", data=encoded, mimeType=mime_type),
            TextContent(
                type="text",
                text=(
                    f"Decoded camera frame from topic '{frame['topic']}' "
                    f"(frame_count={frame['frame_count']}, size={width}x{height}, bytes={frame_size})."
                ),
            ),
        ],
        structured_content=response,
    )


@mcp_camera_v2.tool()
async def tbot_camera_capture_frames_during_rotation(
    degrees: float = 360.0,
    num_frames: int = 8,
) -> dict[str, Any]:
    """Rotate the robot while capturing frames at evenly spaced angular intervals. Returns all frames with their heading offsets for use with tbot_vision_analyze_frames_for_object."""
    degrees_f = _ensure_finite("degrees", degrees)
    if not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")

    step_deg = degrees_f / num_frames
    node = _get_camera_node()

    captured_frames: list[dict[str, Any]] = []
    cumulative_deg = 0.0

    async with Client(MOTION_MCP_URL) as motion:
        for _ in range(num_frames):
            await motion.call_tool("tbot_motion_scan_rotate", {"degrees": step_deg})
            cumulative_deg += step_deg

            snapshot = node.snapshot()
            initial_count = snapshot["frame_count"]
            await asyncio.to_thread(node.wait_for_frame, DEFAULT_CURRENT_VIEW_WAIT_S, initial_count)

            frame = node.latest_frame()
            encoded_bytes, width, height, _ = _decode_and_reencode_jpeg(frame["bytes"])
            image_b64 = base64.b64encode(encoded_bytes).decode("ascii")

            captured_frames.append({
                "heading_offset_deg": round(cumulative_deg, 4),
                "image_base64": image_b64,
                "width": width,
                "height": height,
                "frame_count": frame["frame_count"],
            })

    return {
        "status": "ok",
        "frames": captured_frames,
        "total_degrees_rotated": round(cumulative_deg, 4),
        "num_frames": len(captured_frames),
        "step_deg": round(step_deg, 4),
    }


def _extract_tool_dict(tool_result: Any) -> dict[str, Any]:
    """Extract a dict from a FastMCP Client.call_tool() ToolResult."""
    if isinstance(tool_result, dict):
        return tool_result
    structured = getattr(tool_result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    structured_camel = getattr(tool_result, "structuredContent", None)
    if isinstance(structured_camel, dict):
        return structured_camel
    content = getattr(tool_result, "content", None)
    if isinstance(content, list):
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
    return {}


def _normalize_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None

    normalized: dict[str, float] = {}
    for key in ("cx", "cy", "w", "h"):
        raw = value.get(key)
        if not isinstance(raw, (int, float)):
            return None
        parsed = float(raw)
        if not math.isfinite(parsed):
            return None
        normalized[key] = max(0.0, min(1.0, parsed))
    return normalized


def _clamp(value: float, limit: float) -> float:
    if limit <= 0.0:
        return 0.0
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


@mcp_camera_v2.tool()
async def tbot_camera_reorient_to_object(
    object_name: str,
    threshold_deg: float = REORIENT_THRESHOLD_DEG,
    max_iterations: int = REORIENT_MAX_ITERATIONS,
) -> dict[str, Any]:
    """Run a closed-loop fine-tuning pass to center the robot on a target object using its bounding box center."""
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")
    threshold = _ensure_non_negative("threshold_deg", threshold_deg)
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    node = _get_camera_node()
    iterations = 0
    final_in_frame_offset_deg: float | None = None
    final_bbox: dict[str, float] | None = None
    error_history_deg: list[float] = []
    last_command_deg: float | None = None
    previous_abs_error_deg: float | None = None
    no_progress_count = 0

    for _ in range(max_iterations):
        iterations += 1

        snapshot = node.snapshot()
        initial_count = snapshot["frame_count"]
        await asyncio.to_thread(node.wait_for_frame, DEFAULT_CURRENT_VIEW_WAIT_S, initial_count)

        frame = node.latest_frame()
        encoded_bytes, _, _, _ = _decode_and_reencode_jpeg(frame["bytes"])
        b64 = base64.b64encode(encoded_bytes).decode("ascii")

        async with Client(VISION_MCP_URL) as vision:
            result = await vision.call_tool(
                "tbot_vision_analyze_frames_for_object",
                {
                    "frames": [{"heading_offset_deg": 0.0, "image_base64": b64}],
                    "object_name": object_name_clean,
                },
            )
        data = _extract_tool_dict(result)

        status = data.get("status")
        final_bbox = _normalize_bbox(data.get("bbox"))
        if final_bbox is not None:
            final_in_frame_offset_deg = (final_bbox["cx"] - 0.5) * CAMERA_HFOV_DEG
        else:
            in_frame_offset_deg = data.get("in_frame_offset_deg")
            if isinstance(in_frame_offset_deg, (int, float)):
                final_in_frame_offset_deg = float(in_frame_offset_deg)
            else:
                final_in_frame_offset_deg = None

        if status == "not_found" or final_in_frame_offset_deg is None:
            return {
                "status": "lost",
                "reason": "lost",
                "ready_to_approach": False,
                "iterations": iterations,
                "final_in_frame_offset_deg": final_in_frame_offset_deg,
                "final_bbox": final_bbox,
                "error_history_deg": error_history_deg,
                "last_command_deg": last_command_deg,
                "object_name": object_name_clean,
            }

        error_history_deg.append(final_in_frame_offset_deg)
        abs_error_deg = abs(final_in_frame_offset_deg)

        if abs_error_deg <= threshold:
            return {
                "status": "centered",
                "reason": "centered",
                "ready_to_approach": True,
                "iterations": iterations,
                "final_in_frame_offset_deg": final_in_frame_offset_deg,
                "final_bbox": final_bbox,
                "error_history_deg": error_history_deg,
                "last_command_deg": last_command_deg,
                "object_name": object_name_clean,
            }

        if previous_abs_error_deg is not None:
            improved = abs_error_deg <= (previous_abs_error_deg - REORIENT_IMPROVEMENT_EPS_DEG)
            no_progress_count = 0 if improved else (no_progress_count + 1)
            if no_progress_count >= REORIENT_MAX_NO_PROGRESS:
                return {
                    "status": "no_progress",
                    "reason": "no_progress",
                    "ready_to_approach": False,
                    "iterations": iterations,
                    "final_in_frame_offset_deg": final_in_frame_offset_deg,
                    "final_bbox": final_bbox,
                    "error_history_deg": error_history_deg,
                    "last_command_deg": last_command_deg,
                    "object_name": object_name_clean,
                }
        previous_abs_error_deg = abs_error_deg

        correction_deg = final_in_frame_offset_deg * REORIENT_CORRECTION_SIGN
        command_deg = _clamp(correction_deg, REORIENT_MAX_STEP_DEG)
        last_command_deg = command_deg

        async with Client(MOTION_MCP_URL) as motion:
            await motion.call_tool("tbot_motion_scan_rotate", {"degrees": command_deg})

    return {
        "status": "max_iterations_reached",
        "reason": "max_iterations",
        "ready_to_approach": False,
        "iterations": iterations,
        "final_in_frame_offset_deg": final_in_frame_offset_deg,
        "final_bbox": final_bbox,
        "error_history_deg": error_history_deg,
        "last_command_deg": last_command_deg,
        "object_name": object_name_clean,
    }


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18206,
    path: str = "/turtlebot-camera-v2",
    options: dict = {},
) -> None:
    """Run the TurtleBot Camera MCP Server V2."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting TurtleBot Camera MCP V2 topic=%s frame_max_bytes=%d at %s:%s%s",
        CAMERA_TOPIC,
        DEFAULT_FRAME_MAX_BYTES,
        host,
        port,
        path,
    )
    _start_ros_thread()
    mcp_camera_v2.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
