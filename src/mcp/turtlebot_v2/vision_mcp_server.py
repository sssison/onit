"""
TurtleBot Vision MCP Server V2.

Consumes image input from the existing OnIt image flow and returns a structured
scene analysis payload.
"""

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

mcp_vision_v2 = FastMCP("TurtleBot Vision MCP Server V2")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


VISION_TIMEOUT_S = _env_float("TBOT_VISION_TIMEOUT_S", 60.0)
DEFAULT_VISION_HOST = os.getenv("TBOT_VISION_HOST") or os.getenv("ONIT_HOST", "http://202.92.159.240:8001/v1")
DEFAULT_VISION_MODEL = os.getenv("TBOT_VISION_MODEL", "Qwen/Qwen3.5-35B-A3B")
DEFAULT_VISION_API_KEY = os.getenv("TBOT_VISION_API_KEY", "EMPTY")

CAMERA_MCP_URL = os.getenv("TBOT_CAMERA_MCP_URL", "http://127.0.0.1:18206/turtlebot-camera-v2")
MOTION_MCP_URL = os.getenv("TBOT_MOTION_MCP_URL", "http://127.0.0.1:18205/turtlebot-motion-v2")
CAMERA_HFOV_DEG = _env_float("CAMERA_HFOV_DEG", 62.0)


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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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


def _select_best_bbox(parsed: dict[str, Any]) -> dict[str, float] | None:
    raw_detections = parsed.get("detections")
    if isinstance(raw_detections, list):
        best_bbox: dict[str, float] | None = None
        best_conf = -1.0
        for item in raw_detections:
            if not isinstance(item, dict):
                continue
            bbox = _normalize_bbox(item.get("bbox"))
            if bbox is None:
                continue
            conf = _normalize_confidence(item.get("confidence"))
            conf_score = conf if conf is not None else 0.0
            if best_bbox is None or conf_score > best_conf:
                best_bbox = bbox
                best_conf = conf_score
        if best_bbox is not None:
            return best_bbox

    top_level_bbox = _normalize_bbox(parsed.get("bbox"))
    if top_level_bbox is not None:
        return top_level_bbox

    # Backward compatibility with older x_center-only output.
    x_center = _normalize_unit_float(parsed.get("x_center"))
    if x_center is None:
        return None
    return {"cx": x_center, "cy": 0.5, "w": 0.0, "h": 0.0}


def _normalize_output(parsed: dict[str, Any] | None, raw_response: str, model_info: dict[str, Any]) -> dict[str, Any]:
    parsed = parsed or {}

    summary = parsed.get("summary")
    if not isinstance(summary, str) or summary.strip() == "":
        summary = "Scene analysis completed, but response was not fully structured."

    navigation = parsed.get("navigation")
    if navigation is None:
        navigation = {}
    elif not isinstance(navigation, dict):
        navigation = {"notes": str(navigation)}

    return {
        "summary": summary,
        "objects": _as_list(parsed.get("objects")),
        "visible_text": _as_list(parsed.get("visible_text")),
        "navigation": navigation,
        "hazards": _as_list(parsed.get("hazards")),
        "confidence": _normalize_confidence(parsed.get("confidence")),
        "model_info": model_info,
        "raw_response": raw_response,
    }


def _normalize_input_image(image_input: str) -> str:
    if not isinstance(image_input, str) or image_input.strip() == "":
        raise ValueError("images must contain a non-empty base64-encoded image string.")

    value = image_input.strip()
    if value.startswith("data:"):
        marker = ";base64,"
        if marker not in value:
            raise ValueError("Unsupported data URL image format. Expected ';base64,'.")
        value = value.split(marker, 1)[1]

    try:
        base64.b64decode(value, validate=True)
    except Exception as e:
        raise ValueError("images[0] must be valid base64 image data.") from e
    return value


@mcp_vision_v2.tool()
async def tbot_vision_analyze_scene(
    images: list[str],
    task: str = "Analyze scene for navigation, obstacles, objects, and visible text.",
) -> dict[str, Any]:
    """
    Analyze a scene from camera image input.

    Uses only the first image to match current OnIt media handling behavior.
    """
    if not images:
        raise ValueError("images is required and must contain at least one base64 image.")

    first_image = _normalize_input_image(images[0])

    host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    api_key = _resolve_api_key(host)

    model_info = {
        "host": host,
        "model": model,
        "timeout_s": VISION_TIMEOUT_S,
        "requested_images": len(images),
        "used_images": 1,
        "used_index": 0,
    }

    system_prompt = (
        "You are a robot perception assistant. Return only valid JSON with keys: "
        "summary (string), objects (array), visible_text (array), navigation (object), "
        "hazards (array), confidence (0..1 number). No markdown."
    )

    user_content = [
        {"type": "text", "text": task},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_image}"}},
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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
        parsed = _extract_first_json_object(raw_response)
        return _normalize_output(parsed, raw_response, model_info)
    except APITimeoutError:
        raw_response = f"Vision request timed out after {VISION_TIMEOUT_S} seconds."
    except OpenAIError as e:
        raw_response = f"Vision model request failed: {e}"
    except Exception as e:
        raw_response = f"Unexpected vision processing error: {e}"

    return {
        "summary": "Vision analysis failed.",
        "objects": [],
        "visible_text": [],
        "navigation": {},
        "hazards": [raw_response],
        "confidence": None,
        "model_info": model_info,
        "raw_response": raw_response,
    }


def _extract_image_b64(tool_result: Any) -> str:
    """Extract image_base64 from a FastMCP Client.call_tool() ToolResult."""
    if isinstance(tool_result, dict):
        val = tool_result.get("image_base64")
        if isinstance(val, str) and val.strip():
            return val.strip()
    structured = getattr(tool_result, "structured_content", None)
    if isinstance(structured, dict):
        val = structured.get("image_base64")
        if isinstance(val, str) and val.strip():
            return val.strip()
    structured_camel = getattr(tool_result, "structuredContent", None)
    if isinstance(structured_camel, dict):
        val = structured_camel.get("image_base64")
        if isinstance(val, str) and val.strip():
            return val.strip()
    content = getattr(tool_result, "content", None)
    if isinstance(content, list):
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        val = parsed.get("image_base64")
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                except Exception:
                    pass
    raise RuntimeError("Could not extract image_base64 from camera tool response.")


def _normalize_detection(parsed: dict[str, Any] | None, raw_text: str) -> dict[str, Any]:
    parsed = parsed or {}
    matched_raw = parsed.get("matched", False)
    confidence_raw = parsed.get("confidence", 0.0)
    evidence_raw = parsed.get("evidence", "")

    matched = bool(matched_raw)
    confidence = 0.0
    if isinstance(confidence_raw, (int, float)):
        confidence = float(confidence_raw)
    elif isinstance(confidence_raw, str):
        try:
            confidence = float(confidence_raw.strip())
        except ValueError:
            confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence = evidence_raw if isinstance(evidence_raw, str) else str(evidence_raw)
    return {
        "matched": matched,
        "confidence": confidence,
        "evidence": evidence,
        "raw_response": raw_text,
    }


async def _detect_object_in_image(
    vision_client: AsyncOpenAI,
    target: str,
    image_base64: str,
    timeout_s: float,
    model: str,
) -> dict[str, Any]:
    system_prompt = (
        "You are a strict robot object detector. "
        "Return only JSON with keys: matched (boolean), confidence (number 0..1), evidence (string)."
    )
    user_prompt = f"Is the target object '{target}' visible in this image?"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ],
        },
    ]

    raw_response = ""
    try:
        completion = await vision_client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout_s,
        )
        raw_response = completion.choices[0].message.content or ""
        parsed = _extract_first_json_object(raw_response)
        return _normalize_detection(parsed, raw_response)
    except APITimeoutError:
        raw_response = f"Vision request timed out after {timeout_s:.2f}s."
    except OpenAIError as e:
        raw_response = f"Vision request failed: {e}"
    except Exception as e:
        raw_response = f"Unexpected vision error: {e}"

    return _normalize_detection({"matched": False, "confidence": 0.0, "evidence": raw_response}, raw_response)


def _normalize_detection_with_position(parsed: dict[str, Any] | None, raw_text: str) -> dict[str, Any]:
    base = _normalize_detection(parsed, raw_text)
    parsed = parsed or {}
    x_center: float | None = None
    in_frame_offset_deg: float | None = None
    bbox: dict[str, float] | None = None

    if base["matched"]:
        bbox = _select_best_bbox(parsed)
        if bbox is not None:
            x_center = bbox["cx"]
            in_frame_offset_deg = (x_center - 0.5) * CAMERA_HFOV_DEG

    return {**base, "x_center": x_center, "bbox": bbox, "in_frame_offset_deg": in_frame_offset_deg}


async def _detect_object_with_position(
    vision_client: AsyncOpenAI,
    target: str,
    image_base64: str,
    timeout_s: float,
    model: str,
) -> dict[str, Any]:
    system_prompt = (
        "You are a strict robot object detector. "
        "Return only JSON with keys: matched (boolean), confidence (number 0..1), evidence (string), "
        "bbox (object with normalized cx, cy, w, h values in range 0..1 for the matched target, or null if not matched). "
        "If multiple target instances are visible, return detections array and use the highest-confidence instance as bbox. "
        "No markdown."
    )
    user_prompt = f"Is the target object '{target}' visible in this image?"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ],
        },
    ]

    raw_response = ""
    try:
        completion = await vision_client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout_s,
        )
        raw_response = completion.choices[0].message.content or ""
        parsed = _extract_first_json_object(raw_response)
        return _normalize_detection_with_position(parsed, raw_response)
    except APITimeoutError:
        raw_response = f"Vision request timed out after {timeout_s:.2f}s."
    except OpenAIError as e:
        raw_response = f"Vision request failed: {e}"
    except Exception as e:
        raw_response = f"Unexpected vision error: {e}"

    return _normalize_detection_with_position(
        {"matched": False, "confidence": 0.0, "evidence": raw_response}, raw_response
    )


@mcp_vision_v2.tool()
async def tbot_vision_capture_and_analyze(
    prompt: str = "Analyze scene for navigation, obstacles, objects, and visible text.",
) -> dict[str, Any]:
    """Capture a frame from the camera and analyze it in a single call. Returns the same schema as tbot_vision_analyze_scene."""
    async with Client(CAMERA_MCP_URL) as cam:
        frame = await cam.call_tool("tbot_camera_get_decoded_frame", {"include_base64": True})
    image_b64 = _extract_image_b64(frame)
    return await tbot_vision_analyze_scene(images=[image_b64], task=prompt)


@mcp_vision_v2.tool()
async def tbot_vision_scan_for_object(
    object_name: str,
    rotation_budget_degrees: float = 360.0,
    step_degrees: float = 20.0,
    speed: float = 0.3,
    confidence_threshold: float = 0.65,
) -> dict[str, Any]:
    """Rotate the robot in steps and detect a target object in each captured frame.

    When status="found", the robot has ALREADY stopped at heading_offset_deg —
    it is already facing the object. Do NOT rotate by heading_offset_deg again.
    Use tbot_camera_reorient_to_object to fine-tune centering, then move forward.

    heading_offset_deg records how far the robot rotated during the scan (informational).
    """
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")

    rotation_budget = float(rotation_budget_degrees)
    step = float(step_degrees)
    if step <= 0:
        raise ValueError("step_degrees must be > 0")
    threshold = max(0.0, min(1.0, float(confidence_threshold)))

    vision_host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    vision_model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    vision_api_key = _resolve_api_key(vision_host)
    vision_client = AsyncOpenAI(base_url=vision_host, api_key=vision_api_key)

    cumulative_deg = 0.0
    frames_analyzed = 0
    best_confidence = 0.0

    try:
        async with Client(CAMERA_MCP_URL) as cam, Client(MOTION_MCP_URL) as motion:
            while cumulative_deg < rotation_budget:
                this_step = min(step, rotation_budget - cumulative_deg)
                await motion.call_tool("tbot_motion_scan_rotate", {"degrees": this_step, "speed": speed})
                cumulative_deg += this_step

                frame = await cam.call_tool("tbot_camera_get_decoded_frame", {"include_base64": True})
                image_b64 = _extract_image_b64(frame)

                detection = await _detect_object_in_image(
                    vision_client=vision_client,
                    target=object_name_clean,
                    image_base64=image_b64,
                    timeout_s=VISION_TIMEOUT_S,
                    model=vision_model,
                )
                frames_analyzed += 1

                if detection["confidence"] > best_confidence:
                    best_confidence = detection["confidence"]

                if detection["matched"] and detection["confidence"] >= threshold:
                    return {
                        "status": "found",
                        "heading_offset_deg": cumulative_deg,
                        "frames_analyzed": frames_analyzed,
                        "total_degrees_rotated": cumulative_deg,
                        "best_confidence": best_confidence,
                        "object_name": object_name_clean,
                    }
    except Exception as e:
        return {
            "status": "error",
            "heading_offset_deg": None,
            "frames_analyzed": frames_analyzed,
            "total_degrees_rotated": cumulative_deg,
            "best_confidence": best_confidence,
            "object_name": object_name_clean,
            "error": str(e),
        }

    return {
        "status": "not_found",
        "heading_offset_deg": None,
        "frames_analyzed": frames_analyzed,
        "total_degrees_rotated": cumulative_deg,
        "best_confidence": best_confidence,
        "object_name": object_name_clean,
    }


@mcp_vision_v2.tool()
async def tbot_vision_analyze_frames_for_object(
    frames: list[dict],
    object_name: str,
    confidence_threshold: float = 0.65,
) -> dict[str, Any]:
    """Analyze a list of frames with heading metadata (from tbot_camera_capture_frames_during_rotation) to find a target object. Returns the heading offset where the object was found, or not_found."""
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")
    if not frames:
        raise ValueError("frames must be a non-empty list")

    threshold = max(0.0, min(1.0, float(confidence_threshold)))

    vision_host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    vision_model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    vision_api_key = _resolve_api_key(vision_host)
    vision_client = AsyncOpenAI(base_url=vision_host, api_key=vision_api_key)

    best_heading: float | None = None
    best_confidence = 0.0
    best_in_frame_offset_deg: float | None = None
    best_bbox: dict[str, float] | None = None
    frames_analyzed = 0

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        image_b64 = frame.get("image_base64", "")
        heading_offset = frame.get("heading_offset_deg", 0.0)
        if not isinstance(image_b64, str) or not image_b64.strip():
            continue

        detection = await _detect_object_with_position(
            vision_client=vision_client,
            target=object_name_clean,
            image_base64=image_b64.strip(),
            timeout_s=VISION_TIMEOUT_S,
            model=vision_model,
        )
        frames_analyzed += 1

        if detection["confidence"] > best_confidence:
            best_confidence = detection["confidence"]

        if detection["matched"] and detection["confidence"] >= threshold:
            best_heading = float(heading_offset)
            best_in_frame_offset_deg = detection["in_frame_offset_deg"]
            if isinstance(detection.get("bbox"), dict):
                best_bbox = detection["bbox"]
            break

    if best_heading is not None:
        return {
            "status": "found",
            "heading_offset_deg": best_heading,
            "bbox": best_bbox,
            "in_frame_offset_deg": best_in_frame_offset_deg,
            "frames_analyzed": frames_analyzed,
            "best_confidence": best_confidence,
            "object_name": object_name_clean,
        }

    return {
        "status": "not_found",
        "heading_offset_deg": None,
        "bbox": None,
        "in_frame_offset_deg": None,
        "frames_analyzed": frames_analyzed,
        "best_confidence": best_confidence,
        "object_name": object_name_clean,
    }


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18207,
    path: str = "/turtlebot-vision-v2",
    options: dict = {},
) -> None:
    """Run the TurtleBot Vision MCP Server V2."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting TurtleBot Vision MCP V2 model=%s host=%s timeout_s=%.2f at %s:%s%s",
        os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL),
        os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST),
        VISION_TIMEOUT_S,
        host,
        port,
        path,
    )
    mcp_vision_v2.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
