"""
TurtleBot Vision MCP Server V3.

Reads frames directly from /dev/shm/latest_frame.jpg — no ROS dependency.
"""

import asyncio
import base64
import json
import logging
import math
import os
import re
import time
from typing import Any

from fastmcp import Client, FastMCP
from openai import APITimeoutError, AsyncOpenAI, OpenAIError

logger = logging.getLogger(__name__)

mcp_vision_v3 = FastMCP("TurtleBot Vision MCP Server V3")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


FRAME_PATH = os.getenv("TBOT_FRAME_PATH", "/dev/shm/latest_frame.jpg")
DEFAULT_VISION_HOST = os.getenv("TBOT_VISION_HOST") or os.getenv("ONIT_HOST", "http://202.92.159.240:8001/v1")
DEFAULT_VISION_MODEL = os.getenv("TBOT_VISION_MODEL", "Qwen/Qwen3.5-35B-A3B")
DEFAULT_VISION_API_KEY = os.getenv("TBOT_VISION_API_KEY", "EMPTY")
VISION_TIMEOUT_S = _env_float("TBOT_VISION_TIMEOUT_S", 60.0)
CAMERA_HFOV_DEG = _env_float("CAMERA_HFOV_DEG", 62.0)
MOTION_MCP_URL_V3 = os.getenv("TBOT_MOTION_MCP_URL_V3", "http://127.0.0.1:18210/turtlebot-motion-v3")
LIDAR_MCP_URL_V3 = os.getenv("TBOT_LIDAR_MCP_URL_V3", "http://127.0.0.1:18212/turtlebot-lidar-v3")
_LINE_FOLLOW_STOP_DISTANCE_M = _env_float("LINE_FOLLOW_STOP_DISTANCE_M", 0.35)

_SEARCH_STEP_DEG = 10.0
_SEARCH_ANGULAR_SPEED = 0.3   # rad/s — speed used for each 10° rotation step
_SEARCH_FRAME_SETTLE_S = 0.3  # seconds to wait after rotation for a fresh frame

_LINE_FOLLOW_SPEED = _env_float("LINE_FOLLOW_SPEED", 0.1)
_LINE_FOLLOW_ANGULAR = _env_float("LINE_FOLLOW_ANGULAR", 0.3)


def _extract_tool_result_dict(tool_result: Any) -> dict[str, Any]:
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


def _resolve_api_key(host: str) -> str:
    explicit_key = os.getenv("TBOT_VISION_API_KEY", DEFAULT_VISION_API_KEY)
    if "openrouter.ai" in host:
        if explicit_key and explicit_key != "EMPTY":
            return explicit_key
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError(
                "Vision host is OpenRouter but no API key was provided. "
                "Set TBOT_VISION_API_KEY or OPENROUTER_API_KEY."
            )
        return key
    return explicit_key


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return None

    try:
        parsed = json.loads(text[start:end])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, min(1.0, parsed))


def _normalize_unit_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, min(1.0, parsed))


def _normalize_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    keys = ("cx", "cy", "w", "h")
    parsed: dict[str, float] = {}
    for key in keys:
        normalized = _normalize_unit_float(value.get(key))
        if normalized is None:
            return None
        parsed[key] = normalized
    return parsed


def _load_frame_as_base64(path: str) -> str:
    """Read frame bytes from path and return a base64-encoded ASCII string."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("ascii")
    except FileNotFoundError:
        raise RuntimeError(f"Frame file not found: {path!r}")
    except OSError as e:
        raise RuntimeError(f"Could not read frame file {path!r}: {e}")


def _is_detection_good(detection: dict[str, Any], min_confidence: float) -> bool:
    visible = bool(detection.get("visible", False))
    confidence = float(detection.get("confidence", 0.0) or 0.0)
    return visible and confidence >= min_confidence


def _front_distance_from_lidar(data: dict[str, Any]) -> float | None:
    raw = data.get("distance_m")
    if isinstance(raw, (int, float)):
        return float(raw)

    distances = data.get("distances")
    if isinstance(distances, dict):
        front_raw = distances.get("front")
        if isinstance(front_raw, (int, float)):
            return float(front_raw)
    return None


async def _safe_motion_stop(motion: Client) -> None:
    try:
        await motion.call_tool("tbot_motion_stop", {})
    except Exception:
        pass


@mcp_vision_v3.tool()
async def tbot_vision_health() -> dict[str, Any]:
    """Report whether the current frame file exists and is readable."""
    frame_path = os.getenv("TBOT_FRAME_PATH", FRAME_PATH)
    try:
        stat = os.stat(frame_path)
        return {
            "status": "online",
            "frame_path": frame_path,
            "frame_exists": True,
            "frame_size_bytes": int(stat.st_size),
        }
    except OSError:
        return {
            "status": "no_frame",
            "frame_path": frame_path,
            "frame_exists": False,
            "frame_size_bytes": None,
        }


@mcp_vision_v3.tool()
async def tbot_vision_describe_scene(
    prompt: str = "Describe what you see.",
) -> dict[str, Any]:
    """
    Load the latest camera frame and describe the scene using the vision LLM.

    Returns {"description": str, "model_info": dict}.
    """
    frame_path = os.getenv("TBOT_FRAME_PATH", FRAME_PATH)
    host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    api_key = _resolve_api_key(host)

    model_info = {"host": host, "model": model, "timeout_s": VISION_TIMEOUT_S}

    image_b64 = _load_frame_as_base64(frame_path)

    system_prompt = "You are a robot perception assistant. Describe the scene in natural language."
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        },
    ]

    raw_response = ""
    try:
        client = AsyncOpenAI(base_url=host, api_key=api_key)
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=VISION_TIMEOUT_S,
        )
        raw_response = completion.choices[0].message.content or ""
        return {"description": raw_response, "model_info": model_info}
    except APITimeoutError:
        raw_response = f"Vision request timed out after {VISION_TIMEOUT_S}s."
    except OpenAIError as e:
        raw_response = f"Vision model request failed: {e}"
    except Exception as e:
        raw_response = f"Unexpected vision processing error: {e}"

    return {"description": raw_response, "model_info": model_info}


@mcp_vision_v3.tool()
async def tbot_vision_find_object(object_name: str) -> dict[str, Any]:
    """
    Check if a named object is visible in the latest camera frame.

    Returns {"visible": bool, "position": "left"|"center"|"right"|null, "confidence": float}.
    Position is derived from the detected bounding box center (cx < 0.33 → left, 0.33-0.66 → center, > 0.66 → right).
    """
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")

    frame_path = os.getenv("TBOT_FRAME_PATH", FRAME_PATH)
    host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    api_key = _resolve_api_key(host)

    model_info = {"host": host, "model": model, "timeout_s": VISION_TIMEOUT_S}

    image_b64 = _load_frame_as_base64(frame_path)

    system_prompt = (
        "Return only JSON with keys: matched (boolean), confidence (0..1), "
        "bbox (object with normalized cx, cy, w, h or null)."
    )
    user_prompt = f"Is the target object '{object_name_clean}' visible in this image?"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        },
    ]

    raw_response = ""
    parsed: dict[str, Any] | None = None
    try:
        client = AsyncOpenAI(base_url=host, api_key=api_key)
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=VISION_TIMEOUT_S,
        )
        raw_response = completion.choices[0].message.content or ""
        parsed = _extract_first_json_object(raw_response)
    except APITimeoutError:
        raw_response = f"Vision request timed out after {VISION_TIMEOUT_S}s."
    except OpenAIError as e:
        raw_response = f"Vision model request failed: {e}"
    except Exception as e:
        raw_response = f"Unexpected vision processing error: {e}"

    parsed = parsed or {}
    matched = bool(parsed.get("matched", False))
    confidence = _normalize_confidence(parsed.get("confidence")) or 0.0

    position: str | None = None
    bbox: dict[str, float] | None = None
    if matched:
        bbox = _normalize_bbox(parsed.get("bbox"))
        if bbox is not None:
            cx = bbox["cx"]
            if cx < 0.33:
                position = "left"
            elif cx <= 0.66:
                position = "center"
            else:
                position = "right"

    return {
        "visible": matched,
        "position": position,
        "confidence": confidence,
        "bbox": bbox,
        "model_info": model_info,
    }


@mcp_vision_v3.tool()
async def tbot_vision_search_object(
    object_name: str,
    min_confidence: float = 0.5,
    max_steps: int = 36,
) -> dict[str, Any]:
    """
    Rotate the robot in 10-degree steps and search for a named object.

    Checks the current frame first, then rotates left (CCW) 10° at a time until the
    object is found or max_steps rotations have been completed (default 36 = full 360°).

    min_confidence: minimum confidence threshold (0–1) to accept a detection as found.
    max_steps: maximum number of 10° rotation steps before giving up (default 36 = 360°).

    Returns {"found": bool, "position": "left"|"center"|"right"|null, "confidence": float,
             "bbox": ..., "steps_taken": int, "degrees_rotated": float}.
    """
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")
    if not (0.0 < min_confidence <= 1.0):
        raise ValueError("min_confidence must be between 0 (exclusive) and 1 (inclusive)")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    step_rad = _SEARCH_STEP_DEG * math.pi / 180.0
    step_duration_s = step_rad / _SEARCH_ANGULAR_SPEED
    motion_url = os.getenv("TBOT_MOTION_MCP_URL_V3", MOTION_MCP_URL_V3)

    steps_taken = 0
    last_model_info: dict[str, Any] | None = None

    async with Client(motion_url) as motion:
        for step in range(max_steps + 1):
            # Check current frame before rotating
            vision_result = await tbot_vision_find_object(object_name_clean)
            last_model_info = vision_result.get("model_info")

            if vision_result["visible"] and (vision_result["confidence"] or 0.0) >= min_confidence:
                try:
                    await motion.call_tool("tbot_motion_stop", {})
                except Exception:
                    pass
                return {
                    "found": True,
                    "position": vision_result["position"],
                    "confidence": vision_result["confidence"],
                    "bbox": vision_result["bbox"],
                    "steps_taken": steps_taken,
                    "degrees_rotated": steps_taken * _SEARCH_STEP_DEG,
                    "model_info": last_model_info,
                }

            if step == max_steps:
                break

            # Rotate 10° left; duration_s causes auto-stop after the step
            await motion.call_tool(
                "tbot_motion_move",
                {"linear": 0.0, "angular": _SEARCH_ANGULAR_SPEED, "duration_s": step_duration_s},
            )
            steps_taken += 1
            # Wait for the camera frame to update after rotation
            await asyncio.sleep(_SEARCH_FRAME_SETTLE_S)

    return {
        "found": False,
        "position": None,
        "confidence": 0.0,
        "bbox": None,
        "steps_taken": steps_taken,
        "degrees_rotated": steps_taken * _SEARCH_STEP_DEG,
        "model_info": last_model_info,
    }


@mcp_vision_v3.tool()
async def tbot_vision_search_and_approach_object(
    object_name: str,
    target_distance_m: float = 0.5,
    stop_distance_m: float = 0.1,
    forward_speed: float = 0.1,
    forward_step_s: float = 5.0,
    min_confidence: float = 0.5,
    initial_search_max_steps: int = 36,
    timeout_s: float = 90.0,
) -> dict[str, Any]:
    """
    Search for a target object, then approach based on LiDAR-estimated forward distance.

    Workflow:
    1) Scan to find object.
    2) Estimate front distance from LiDAR.
    3) Move forward using duration derived from distance and speed.
    4) Check if object is still visible.
    5) If lost, run one broad rescan; if still lost, stop if already <= 10 cm.
    """
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")
    if target_distance_m <= 0:
        raise ValueError("target_distance_m must be > 0")
    if stop_distance_m <= 0:
        raise ValueError("stop_distance_m must be > 0")
    if stop_distance_m > target_distance_m:
        raise ValueError("stop_distance_m must be <= target_distance_m")
    if forward_speed <= 0:
        raise ValueError("forward_speed must be > 0")
    if forward_step_s <= 0:
        raise ValueError("forward_step_s must be > 0")
    if not (0.0 < min_confidence <= 1.0):
        raise ValueError("min_confidence must be between 0 (exclusive) and 1 (inclusive)")
    if initial_search_max_steps < 1:
        raise ValueError("initial_search_max_steps must be >= 1")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")

    close_enough_m = max(0.1, float(stop_distance_m))

    started = time.monotonic()
    motion_url = os.getenv("TBOT_MOTION_MCP_URL_V3", MOTION_MCP_URL_V3)
    lidar_url = os.getenv("TBOT_LIDAR_MCP_URL_V3", LIDAR_MCP_URL_V3)

    counters = {
        "search_steps": 0,
        "forward_segments": 0,
        "rescan_attempts": 0,
        "vision_checks": 0,
    }
    had_lost_target = False
    last_detection: dict[str, Any] | None = None
    errors: list[str] = []

    initial_search = await tbot_vision_search_object(
        object_name=object_name_clean,
        min_confidence=min_confidence,
        max_steps=initial_search_max_steps,
    )
    counters["search_steps"] += int(initial_search.get("steps_taken") or 0)
    if not initial_search.get("found", False):
        return {
            "status": "not_found",
            "object_name": object_name_clean,
            "target_distance_m": float(target_distance_m),
            "final_front_distance_m": None,
            "phases": counters,
            "last_detection": None,
            "history_summary": {
                "had_lost_target": False,
                "errors_count": 0,
            },
            "timing_s": time.monotonic() - started,
        }

    async with Client(motion_url) as motion:
        async with Client(lidar_url) as lidar:
            try:
                while True:
                    elapsed = time.monotonic() - started
                    if elapsed >= timeout_s:
                        await _safe_motion_stop(motion)
                        return {
                            "status": "lost_timeout" if had_lost_target else "timeout",
                            "object_name": object_name_clean,
                            "target_distance_m": float(target_distance_m),
                            "final_front_distance_m": None,
                            "phases": counters,
                            "last_detection": last_detection,
                            "history_summary": {
                                "had_lost_target": had_lost_target,
                                "errors_count": len(errors),
                            },
                            "timing_s": elapsed,
                            "errors": errors or None,
                        }

                    # 1) Estimate front distance first.
                    lidar_raw = await lidar.call_tool(
                        "tbot_lidar_get_obstacle_distances",
                        {"sector": "front"},
                    )
                    lidar_data = _extract_tool_result_dict(lidar_raw)
                    front_distance = _front_distance_from_lidar(lidar_data)
                    if front_distance is not None and front_distance <= target_distance_m:
                        await _safe_motion_stop(motion)
                        return {
                            "status": "reached",
                            "object_name": object_name_clean,
                            "target_distance_m": float(target_distance_m),
                            "final_front_distance_m": front_distance,
                            "phases": counters,
                            "last_detection": last_detection,
                            "history_summary": {
                                "had_lost_target": had_lost_target,
                                "errors_count": len(errors),
                            },
                            "timing_s": time.monotonic() - started,
                            "errors": errors or None,
                        }

                    if front_distance is not None and front_distance <= close_enough_m:
                        await _safe_motion_stop(motion)
                        return {
                            "status": "collision_blocked",
                            "object_name": object_name_clean,
                            "target_distance_m": float(target_distance_m),
                            "final_front_distance_m": front_distance,
                            "phases": counters,
                            "last_detection": last_detection,
                            "history_summary": {
                                "had_lost_target": had_lost_target,
                                "errors_count": len(errors),
                            },
                            "timing_s": time.monotonic() - started,
                            "errors": errors or None,
                        }

                    # 2) Keep low-sensitivity collision guard (~10 cm) right before forward motion.
                    collision_raw = await lidar.call_tool(
                        "tbot_lidar_check_collision",
                        {"front_threshold_m": close_enough_m},
                    )
                    collision_data = _extract_tool_result_dict(collision_raw)
                    if collision_data.get("risk_level") == "stop":
                        blocked_front = None
                        distances = collision_data.get("distances")
                        if isinstance(distances, dict):
                            front_raw = distances.get("front")
                            if isinstance(front_raw, (int, float)):
                                blocked_front = float(front_raw)
                        await _safe_motion_stop(motion)
                        return {
                            "status": "collision_blocked",
                            "object_name": object_name_clean,
                            "target_distance_m": float(target_distance_m),
                            "final_front_distance_m": blocked_front,
                            "phases": counters,
                            "last_detection": last_detection,
                            "history_summary": {
                                "had_lost_target": had_lost_target,
                                "errors_count": len(errors),
                            },
                            "timing_s": time.monotonic() - started,
                            "errors": errors or None,
                        }

                    # 3) Estimate how long to move using LiDAR distance.
                    segment_duration = float(forward_step_s)
                    if front_distance is not None:
                        target_travel_m = max(0.0, front_distance - target_distance_m)
                        duration_to_target = target_travel_m / max(0.01, forward_speed)
                        if duration_to_target <= 0:
                            await _safe_motion_stop(motion)
                            return {
                                "status": "reached",
                                "object_name": object_name_clean,
                                "target_distance_m": float(target_distance_m),
                                "final_front_distance_m": front_distance,
                                "phases": counters,
                                "last_detection": last_detection,
                                "history_summary": {
                                    "had_lost_target": had_lost_target,
                                    "errors_count": len(errors),
                                },
                                "timing_s": time.monotonic() - started,
                                "errors": errors or None,
                            }

                        safe_travel_m = max(0.0, front_distance - close_enough_m)
                        max_duration_by_safety = safe_travel_m / max(0.01, forward_speed)
                        if max_duration_by_safety <= 0:
                            await _safe_motion_stop(motion)
                            return {
                                "status": "collision_blocked",
                                "object_name": object_name_clean,
                                "target_distance_m": float(target_distance_m),
                                "final_front_distance_m": front_distance,
                                "phases": counters,
                                "last_detection": last_detection,
                                "history_summary": {
                                    "had_lost_target": had_lost_target,
                                    "errors_count": len(errors),
                                },
                                "timing_s": time.monotonic() - started,
                                "errors": errors or None,
                            }
                        segment_duration = min(segment_duration, duration_to_target, max_duration_by_safety)
                    segment_duration = max(0.1, segment_duration)

                    forward_raw = await motion.call_tool(
                        "tbot_motion_move_forward",
                        {
                            "speed": forward_speed,
                            "duration_seconds": segment_duration,
                        },
                    )
                    forward_result = _extract_tool_result_dict(forward_raw)
                    counters["forward_segments"] += 1

                    if forward_result.get("status") == "collision_risk":
                        await _safe_motion_stop(motion)
                        return {
                            "status": "collision_blocked",
                            "object_name": object_name_clean,
                            "target_distance_m": float(target_distance_m),
                            "final_front_distance_m": forward_result.get("front_distance"),
                            "phases": counters,
                            "last_detection": last_detection,
                            "history_summary": {
                                "had_lost_target": had_lost_target,
                                "errors_count": len(errors),
                            },
                            "timing_s": time.monotonic() - started,
                            "errors": errors or None,
                        }

                    # 4) After forward motion, verify target is still visible.
                    detection = await tbot_vision_find_object(object_name_clean)
                    counters["vision_checks"] += 1
                    last_detection = {
                        "visible": bool(detection.get("visible", False)),
                        "confidence": float(detection.get("confidence", 0.0) or 0.0),
                        "position": detection.get("position"),
                        "bbox": detection.get("bbox"),
                    }

                    if _is_detection_good(detection, min_confidence):
                        continue

                    had_lost_target = True
                    counters["rescan_attempts"] += 1

                    # User-requested safety fallback: if target is lost, verify if we are already ~10 cm away.
                    lost_lidar_raw = await lidar.call_tool(
                        "tbot_lidar_get_obstacle_distances",
                        {"sector": "front"},
                    )
                    lost_lidar_data = _extract_tool_result_dict(lost_lidar_raw)
                    lost_front_distance = _front_distance_from_lidar(lost_lidar_data)
                    if lost_front_distance is not None and lost_front_distance <= close_enough_m:
                        await _safe_motion_stop(motion)
                        return {
                            "status": "reached",
                            "object_name": object_name_clean,
                            "target_distance_m": float(target_distance_m),
                            "final_front_distance_m": lost_front_distance,
                            "phases": counters,
                            "last_detection": last_detection,
                            "history_summary": {
                                "had_lost_target": had_lost_target,
                                "errors_count": len(errors),
                            },
                            "timing_s": time.monotonic() - started,
                            "errors": errors or None,
                        }

                    # No local reorientation policy: only broad rescan when target is lost.
                    broad_rescan = await tbot_vision_search_object(
                        object_name=object_name_clean,
                        min_confidence=min_confidence,
                        max_steps=initial_search_max_steps,
                    )
                    counters["search_steps"] += int(broad_rescan.get("steps_taken") or 0)
                    if broad_rescan.get("found", False):
                        last_detection = {
                            "visible": True,
                            "confidence": float(broad_rescan.get("confidence", 0.0) or 0.0),
                            "position": broad_rescan.get("position"),
                            "bbox": broad_rescan.get("bbox"),
                        }
                        continue

                    errors.append("reacquire_scan_not_found")
                    await asyncio.sleep(_SEARCH_FRAME_SETTLE_S)
            except Exception as e:
                await _safe_motion_stop(motion)
                return {
                    "status": "error",
                    "object_name": object_name_clean,
                    "target_distance_m": float(target_distance_m),
                    "final_front_distance_m": None,
                    "phases": counters,
                    "last_detection": last_detection,
                    "history_summary": {
                        "had_lost_target": had_lost_target,
                        "errors_count": len(errors) + 1,
                    },
                    "timing_s": time.monotonic() - started,
                    "errors": [*errors, str(e)],
                }


@mcp_vision_v3.tool()
async def tbot_vision_follow_line(
    color: str = "black",
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """
    Follow a colored line on the floor using the camera and LLM perception.

    color: the color of the line to follow (e.g. "black", "red", "yellow").
    timeout_s: stop and return after this many seconds.

    Returns {"status": "line_lost"|"timeout"|"obstacle_reached", "ticks_followed": int,
             "front_distance": float|None}.
    """
    color_clean = color.strip() if isinstance(color, str) else ""
    if not color_clean:
        raise ValueError("color must be a non-empty string")

    timeout_f = float(timeout_s)
    if timeout_f <= 0:
        raise ValueError("timeout_s must be > 0")

    start_mono = time.monotonic()
    ticks_followed = 0
    motion_url = os.getenv("TBOT_MOTION_MCP_URL_V3", MOTION_MCP_URL_V3)
    lidar_url = os.getenv("TBOT_LIDAR_MCP_URL_V3", LIDAR_MCP_URL_V3)

    system_prompt = "Return only JSON with keys: visible (boolean), offset ('left'|'center'|'right'|null)."
    user_prompt = (
        f"Is there a {color_clean} colored line on the floor? "
        "If yes, is the line to the left, center, or right of the image center?"
    )

    async with Client(motion_url) as motion:
        async with Client(lidar_url) as lidar:
            while True:
                if time.monotonic() - start_mono >= timeout_f:
                    try:
                        await motion.call_tool("tbot_motion_stop", {})
                    except Exception:
                        pass
                    return {"status": "timeout", "ticks_followed": ticks_followed, "front_distance": None}

                # Lidar collision check before any motion or LLM call
                try:
                    raw = await lidar.call_tool("tbot_lidar_check_collision", {})
                    collision = _extract_tool_result_dict(raw)
                except Exception:
                    collision = {}

                risk_level = collision.get("risk_level", "clear")
                front_dist = (collision.get("distances") or {}).get("front")

                if risk_level == "stop":
                    try:
                        await motion.call_tool("tbot_motion_stop", {})
                    except Exception:
                        pass
                    return {"status": "obstacle_reached", "ticks_followed": ticks_followed, "front_distance": front_dist}

                frame_path = os.getenv("TBOT_FRAME_PATH", FRAME_PATH)
                host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
                model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)

                try:
                    image_b64 = _load_frame_as_base64(frame_path)
                except Exception:
                    try:
                        await motion.call_tool("tbot_motion_stop", {})
                    except Exception:
                        pass
                    return {"status": "line_lost", "ticks_followed": ticks_followed, "front_distance": front_dist}

                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ],
                    },
                ]

                parsed: dict[str, Any] = {}
                try:
                    client = AsyncOpenAI(base_url=host, api_key=_resolve_api_key(host))
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        timeout=VISION_TIMEOUT_S,
                    )
                    raw_response = completion.choices[0].message.content or ""
                    parsed = _extract_first_json_object(raw_response) or {}
                except (APITimeoutError, OpenAIError, Exception):
                    parsed = {}

                visible = bool(parsed.get("visible", False))
                offset = parsed.get("offset")

                if not visible:
                    try:
                        await motion.call_tool("tbot_motion_stop", {})
                    except Exception:
                        pass
                    return {"status": "line_lost", "ticks_followed": ticks_followed, "front_distance": front_dist}

                if offset == "left":
                    angular = _LINE_FOLLOW_ANGULAR
                elif offset == "right":
                    angular = -_LINE_FOLLOW_ANGULAR
                else:
                    angular = 0.0

                speed = _LINE_FOLLOW_SPEED * 0.5 if risk_level == "caution" else _LINE_FOLLOW_SPEED

                await motion.call_tool(
                    "tbot_motion_move",
                    {"linear": speed, "angular": angular},
                )
                ticks_followed += 1


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18211,
    path: str = "/turtlebot-vision-v3",
    options: dict = {},
) -> None:
    """Run the TurtleBot Vision MCP Server V3."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting TurtleBot Vision MCP V3 model=%s host=%s at %s:%s%s",
        os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL),
        os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST),
        host,
        port,
        path,
    )
    mcp_vision_v3.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
