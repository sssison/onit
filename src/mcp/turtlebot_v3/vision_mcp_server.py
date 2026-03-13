"""
TurtleBot Vision MCP Server V3.

Reads frames directly from /dev/shm/latest_frame.jpg — no ROS dependency.
"""

import base64
import json
import logging
import math
import os
import re
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
DEFAULT_VISION_MODEL = os.getenv("TBOT_VISION_MODEL", "Qwen/Qwen3.5-9B")
DEFAULT_VISION_API_KEY = os.getenv("TBOT_VISION_API_KEY", "EMPTY")
VISION_TIMEOUT_S = _env_float("TBOT_VISION_TIMEOUT_S", 60.0)
VISION_THINKING_ENABLED = os.getenv("TBOT_VISION_THINKING", "false").strip().lower() == "true"
MOTION_MCP_URL_V3 = os.getenv("TBOT_MOTION_MCP_URL_V3", "http://127.0.0.1:18210/turtlebot-motion-v3")
LIDAR_MCP_URL_V3 = os.getenv("TBOT_LIDAR_MCP_URL_V3", "http://127.0.0.1:18212/turtlebot-lidar-v3")
NAV_MCP_URL_V3 = os.getenv("TBOT_NAV_MCP_URL_V3", "http://127.0.0.1:18213/turtlebot-nav-v3")
TBOT_CAMERA_FOV_DEG = _env_float("TBOT_CAMERA_FOV_DEG", 62.0)
VISION_SCAN_STEP_DEG = 10.0
VISION_SCAN_MAX_STEPS = max(1, int(round(360.0 / VISION_SCAN_STEP_DEG)))
VISION_TURN_SPEED_RPS = max(0.05, _env_float("VISION_TURN_SPEED_RPS", 0.30))
VISION_SCAN_BBOX_RETRY_COUNT = max(1, int(_env_float("VISION_SCAN_BBOX_RETRY_COUNT", 2.0)))
VISION_RECENTER_EPS_DEG = max(0.25, _env_float("VISION_RECENTER_EPS_DEG", 1.0))


def _vision_extra_body() -> dict:
    if VISION_THINKING_ENABLED:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _with_no_think(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("/no_think"):
        return text
    return f"/no_think\n{text}"


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


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return None


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


def _extract_tool_result_dict(tool_result: Any) -> dict[str, Any]:
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


def _extract_valid_distance(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _heuristic_distance_from_bbox_m(bbox: Any) -> float | None:
    if not isinstance(bbox, dict):
        return None
    h = bbox.get("h")
    if not isinstance(h, (int, float)) or not math.isfinite(float(h)):
        return None
    h_f = max(0.01, min(1.0, float(h)))
    # Coarse fallback when LiDAR is unavailable: larger bbox height implies a nearer object.
    return max(0.2, min(3.0, 0.9 / h_f))


def _heading_offset_from_bbox_deg(bbox: Any, camera_fov_deg: float = TBOT_CAMERA_FOV_DEG) -> float | None:
    if not isinstance(bbox, dict):
        return None
    cx = bbox.get("cx")
    if not isinstance(cx, (int, float)) or not math.isfinite(float(cx)):
        return None
    cx_f = max(0.0, min(1.0, float(cx)))
    return (cx_f - 0.5) * float(camera_fov_deg)


def _heading_offset_from_detection_deg(detection: dict[str, Any]) -> float:
    heading = _heading_offset_from_bbox_deg(detection.get("bbox"))
    if heading is not None:
        return heading
    position = str(detection.get("position", "")).strip().lower()
    if position == "left":
        return -TBOT_CAMERA_FOV_DEG / 4.0
    if position == "right":
        return TBOT_CAMERA_FOV_DEG / 4.0
    return 0.0


async def _record_scan_detection_to_spatial_map(
    label: str,
    detection: dict[str, Any],
) -> dict[str, Any]:
    heading_offset_deg = _heading_offset_from_detection_deg(detection)
    distance_m = _heuristic_distance_from_bbox_m(detection.get("bbox"))
    distance_source = "bbox_heuristic"

    lidar_result: dict[str, Any] = {}
    try:
        async with Client(LIDAR_MCP_URL_V3) as lidar:
            lidar_raw = await lidar.call_tool(
                "tbot_lidar_get_distance_at_angle",
                {
                    "angle_deg": heading_offset_deg,
                    "half_width_deg": 5.0,
                    "statistic": "min",
                },
            )
            lidar_result = _extract_tool_result_dict(lidar_raw)
            lidar_distance_m = _extract_valid_distance(lidar_result.get("distance_m"))
            if lidar_distance_m is not None:
                distance_m = lidar_distance_m
                distance_source = "lidar"
    except Exception as e:
        lidar_result = {"error": str(e)}

    if distance_m is None:
        return {
            "success": False,
            "error": "distance_unavailable",
            "label": label,
            "bearing_deg": heading_offset_deg,
            "distance_source": distance_source,
            "lidar": lidar_result,
        }

    confidence = _normalize_confidence(detection.get("confidence"))
    try:
        async with Client(NAV_MCP_URL_V3) as nav:
            nav_raw = await nav.call_tool(
                "tbot_spatial_map_record_observation",
                {
                    "label": label,
                    "bearing_deg": heading_offset_deg,
                    "distance_m": distance_m,
                    "confidence": confidence if confidence is not None else 1.0,
                },
            )
            nav_result = _extract_tool_result_dict(nav_raw)
    except Exception as e:
        return {
            "success": False,
            "error": f"spatial_map_update_failed: {e}",
            "label": label,
            "bearing_deg": heading_offset_deg,
            "distance_m": distance_m,
            "distance_source": distance_source,
            "lidar": lidar_result,
        }

    return {
        "success": bool(nav_result.get("success")),
        "label": label,
        "bearing_deg": heading_offset_deg,
        "distance_m": distance_m,
        "distance_source": distance_source,
        "lidar": lidar_result,
        "map_result": nav_result,
    }


async def _turn_by_degrees(motion: Client, turn_deg: float, speed_rps: float = VISION_TURN_SPEED_RPS) -> None:
    direction = "left" if turn_deg >= 0 else "right"
    duration_s = math.radians(abs(turn_deg)) / max(abs(speed_rps), 0.01)
    await motion.call_tool(
        "tbot_motion_turn",
        {
            "direction": direction,
            "speed": max(abs(speed_rps), 0.05),
            "duration_seconds": duration_s,
        },
    )


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


def _build_object_query(
    object_name: str,
    attributes: list[str] | None = None,
    anchor_object: str | None = None,
    relation: str | None = None,
) -> str:
    fragments = [object_name]
    if attributes:
        attrs = [a.strip() for a in attributes if isinstance(a, str) and a.strip()]
        if attrs:
            fragments.append(f"with attributes: {', '.join(attrs)}")
    if anchor_object and isinstance(anchor_object, str) and anchor_object.strip():
        fragments.append(f"near/relative to: {anchor_object.strip()}")
    if relation and isinstance(relation, str) and relation.strip():
        fragments.append(f"spatial relation: {relation.strip()}")
    return "; ".join(fragments)


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
                {"type": "text", "text": _with_no_think(prompt)},
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
            extra_body=_vision_extra_body(),
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


async def _find_object_in_frame(
    object_name: str,
    attributes: list[str] | None = None,
    anchor_object: str | None = None,
    relation: str | None = None,
) -> dict[str, Any]:
    """
    Check if object_name is visible in the current frame.
    Returns {"visible": bool, "confidence": float, "bbox": dict|None, "position": str|None}.
    """
    frame_path = os.getenv("TBOT_FRAME_PATH", FRAME_PATH)
    host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    api_key = _resolve_api_key(host)

    image_b64 = _load_frame_as_base64(frame_path)
    system_prompt = (
        "Return only JSON with keys: matched (boolean), bbox (object with normalized cx, cy, w, h or null). "
        "confidence (0..1) is optional."
    )
    query = _build_object_query(
        object_name=object_name,
        attributes=attributes,
        anchor_object=anchor_object,
        relation=relation,
    )
    user_prompt = _with_no_think(f"Is the target object visible in this image? Target description: {query}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]},
    ]

    parsed: dict[str, Any] = {}
    try:
        client = AsyncOpenAI(base_url=host, api_key=api_key)
        completion = await client.chat.completions.create(
            model=model, messages=messages, timeout=VISION_TIMEOUT_S,
            extra_body=_vision_extra_body(),
        )
        raw = completion.choices[0].message.content or ""
        parsed = _extract_first_json_object(raw) or {}
    except Exception:
        pass

    bbox = _normalize_bbox(parsed.get("bbox"))
    matched = _normalize_bool(parsed.get("matched"))
    if matched is None:
        matched = _normalize_bool(parsed.get("visible"))
    if matched is None:
        matched = bbox is not None
    confidence = _normalize_confidence(parsed.get("confidence")) or 0.0
    position: str | None = None
    if matched and bbox is not None:
        cx = bbox["cx"]
        position = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
    return {"visible": matched, "confidence": confidence, "bbox": bbox, "position": position}


@mcp_vision_v3.tool()
async def tbot_vision_find_object(
    object_name: str,
    attributes: list[str] | None = None,
    anchor_object: str | None = None,
    relation: str | None = None,
) -> dict[str, Any]:
    """
    Find a named object by checking the current frame first, then scanning if needed.
    """
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")

    host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    model_info = {
        "host": host,
        "model": model,
        "timeout_s": VISION_TIMEOUT_S,
    }

    result = await _find_object_in_frame(
        object_name_clean,
        attributes=attributes,
        anchor_object=anchor_object,
        relation=relation,
    )

    scan_steps = 0
    recenter_applied = False
    spatial_map_update: dict[str, Any] = {
        "success": False,
        "reason": "scan_not_performed",
    }
    async with Client(MOTION_MCP_URL_V3) as motion:
        while not bool(result.get("visible")) and scan_steps < VISION_SCAN_MAX_STEPS:
            await _turn_by_degrees(motion, VISION_SCAN_STEP_DEG, VISION_TURN_SPEED_RPS)
            scan_steps += 1
            result = await _find_object_in_frame(
                object_name_clean,
                attributes=attributes,
                anchor_object=anchor_object,
                relation=relation,
            )

        if not bool(result.get("visible")):
            spatial_map_update = {
                "success": False,
                "reason": "target_not_found_after_scan",
                "scan_steps": scan_steps,
            }
            return {
                **result,
                "success": False,
                "scan_steps": scan_steps,
                "degrees_swept": float(scan_steps * VISION_SCAN_STEP_DEG),
                "recenter_applied": False,
                "stopped_reason": "max_retries",
                "model_info": model_info,
                "spatial_map_updated": False,
                "spatial_map_update": spatial_map_update,
            }

        bbox_retries = 0
        bbox = result.get("bbox")
        while bbox is None and bbox_retries < VISION_SCAN_BBOX_RETRY_COUNT:
            bbox_retries += 1
            refreshed = await _find_object_in_frame(
                object_name_clean,
                attributes=attributes,
                anchor_object=anchor_object,
                relation=relation,
            )
            if bool(refreshed.get("visible")):
                result = refreshed
                bbox = result.get("bbox")

        if not isinstance(bbox, dict):
            spatial_map_update = {
                "success": False,
                "reason": "bbox_unavailable",
                "scan_steps": scan_steps,
            }
            return {
                **result,
                "success": False,
                "scan_steps": scan_steps,
                "degrees_swept": float(scan_steps * VISION_SCAN_STEP_DEG),
                "bbox_retries": bbox_retries,
                "recenter_applied": False,
                "stopped_reason": "bbox_unavailable",
                "model_info": model_info,
                "spatial_map_updated": False,
                "spatial_map_update": spatial_map_update,
            }

        heading_offset = _heading_offset_from_bbox_deg(bbox)
        if heading_offset is not None:
            # _heading_offset_from_bbox_deg is image x-offset; negate to get turn-to-center command.
            turn_to_center_deg = -heading_offset
            if abs(turn_to_center_deg) >= VISION_RECENTER_EPS_DEG:
                await _turn_by_degrees(motion, turn_to_center_deg, VISION_TURN_SPEED_RPS)
                recenter_applied = True

                recentered = await _find_object_in_frame(
                    object_name_clean,
                    attributes=attributes,
                    anchor_object=anchor_object,
                    relation=relation,
                )
                if bool(recentered.get("visible")):
                    result = recentered
                    bbox = result.get("bbox")

    if scan_steps > 0 and bool(result.get("visible")):
        spatial_map_update = await _record_scan_detection_to_spatial_map(
            label=object_name_clean,
            detection=result,
        )

    spatial_map_updated = bool(spatial_map_update.get("success"))

    stopped_reason = "found" if isinstance(result.get("bbox"), dict) else "found_bbox_unavailable"

    return {
        **result,
        "success": True,
        "scan_steps": scan_steps,
        "degrees_swept": float(scan_steps * VISION_SCAN_STEP_DEG),
        "bbox_retries": bbox_retries,
        "recenter_applied": recenter_applied,
        "stopped_reason": stopped_reason,
        "model_info": model_info,
        "spatial_map_updated": spatial_map_updated,
        "spatial_map_update": spatial_map_update,
    }


@mcp_vision_v3.tool()
async def tbot_vision_get_object_bbox(
    object_name: str,
    attributes: list[str] | None = None,
    anchor_object: str | None = None,
    relation: str | None = None,
) -> dict[str, Any]:
    """
    Get object visibility and bbox from the current frame only (no scan/turn).
    """
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")

    host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    model_info = {
        "host": host,
        "model": model,
        "timeout_s": VISION_TIMEOUT_S,
    }

    result = await _find_object_in_frame(
        object_name_clean,
        attributes=attributes,
        anchor_object=anchor_object,
        relation=relation,
    )
    has_bbox = isinstance(result.get("bbox"), dict)
    return {
        **result,
        "success": bool(result.get("visible")) and has_bbox,
        "model_info": model_info,
    }


@mcp_vision_v3.tool()
async def tbot_vision_inspect_floor(
    targets: list[str] | None = None,
    region: str = "lower_half",
    near_object: str | None = None,
) -> dict[str, Any]:
    """
    Structured floor inspection for hazards and misplaced objects.
    """
    requested_targets = targets if isinstance(targets, list) and targets else [
        "cables",
        "spill",
        "feet_legs",
        "fallen_objects",
        "power_strip",
        "floor_tape",
    ]
    query = (
        "Inspect the floor region of the scene and return hazards/objects for: "
        f"{', '.join(str(t) for t in requested_targets)}. "
        f"Region focus: {region}. "
        f"Reference object: {near_object if near_object else 'none'}."
    )
    description = await tbot_vision_describe_scene(prompt=query)
    text = (description.get("description") or "").lower()
    detected = [t for t in requested_targets if isinstance(t, str) and t.lower().replace("_", " ") in text]
    return {
        "status": "ok",
        "region": region,
        "near_object": near_object,
        "targets": requested_targets,
        "detected": detected,
        "raw_description": description.get("description"),
        "model_info": description.get("model_info"),
    }


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
