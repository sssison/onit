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
from typing import Any

import httpx
from fastmcp import FastMCP
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


def _vision_extra_body() -> dict:
    if VISION_THINKING_ENABLED:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}

MOTION_BASE_URL = os.getenv("MOTION_SERVER_BASE_URL", "http://10.158.38.26:5001").rstrip("/")
MOTION_ANGULAR_SIGN = _env_float("MOTION_ANGULAR_SIGN", -1.0)
MOTION_HTTP_TIMEOUT_S = _env_float("MOTION_TIMEOUT_S", 5.0)
MOTION_REFRESH_S = 0.05


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


async def _find_object_in_frame(object_name: str) -> dict[str, Any]:
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
        "Return only JSON with keys: matched (boolean), confidence (0..1), "
        "bbox (object with normalized cx, cy, w, h or null)."
    )
    user_prompt = f"Is the target object '{object_name}' visible in this image?"
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

    matched = bool(parsed.get("matched", False))
    confidence = _normalize_confidence(parsed.get("confidence")) or 0.0
    bbox: dict[str, float] | None = None
    position: str | None = None
    if matched:
        bbox = _normalize_bbox(parsed.get("bbox"))
        if bbox is not None:
            cx = bbox["cx"]
            position = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
    return {"visible": matched, "confidence": confidence, "bbox": bbox, "position": position}


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

    host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    model_info = {"host": host, "model": model, "timeout_s": VISION_TIMEOUT_S}

    result = await _find_object_in_frame(object_name_clean)
    return {**result, "model_info": model_info}


async def _turn_step_async(direction: str, speed_rad_s: float, duration_s: float) -> None:
    """Issue a timed turn to the motion HTTP server."""
    # Keep scan turn semantics aligned with motion server direction mapping.
    # For this robot convention, "right" maps to +1 before applying MOTION_ANGULAR_SIGN.
    sign = 1.0 if direction == "right" else -1.0
    angular = sign * speed_rad_s * MOTION_ANGULAR_SIGN
    deadline = asyncio.get_event_loop().time() + duration_s
    try:
        async with httpx.AsyncClient(timeout=MOTION_HTTP_TIMEOUT_S) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    await client.post(f"{MOTION_BASE_URL}/move",
                                      json={"linear": 0.0, "angular": angular})
                except Exception:
                    break
                await asyncio.sleep(MOTION_REFRESH_S)
            try:
                await client.post(f"{MOTION_BASE_URL}/stop", json={})
            except Exception:
                pass
    except Exception:
        pass


@mcp_vision_v3.tool()
async def tbot_vision_scan_for_object(
    object_name: str,
    step_deg: float = 15.0,
    max_rotation_deg: float = 360.0,
    direction: str = "left",
    turn_speed: float = 0.3,
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Scan for a named object by rotating in steps until it is found.

    Checks the current frame first; if not found, turns step_deg and checks again.
    Stops as soon as the object is found (confidence >= confidence_threshold) or
    the full max_rotation_deg budget is used.

    Returns status="found" with bbox (for recentering) or status="not_found".
    After "found", do NOT call tbot_vision_find_object — use the returned bbox directly.

    direction: "left" | "right"
    turn_speed: rotation speed in rad/s (default 0.3)
    max_rotation_deg is clamped to a full 360° sweep.
    """
    object_name_clean = object_name.strip() if isinstance(object_name, str) else ""
    if not object_name_clean:
        raise ValueError("object_name must be a non-empty string")

    step_f = float(step_deg) if isinstance(step_deg, (int, float)) else 10.0
    requested_max_rot = (
        float(max_rotation_deg)
        if isinstance(max_rotation_deg, (int, float))
        else 360.0
    )
    if not math.isfinite(requested_max_rot):
        requested_max_rot = 360.0
    # Enforce full scan coverage even if callers request 180°.
    max_rot_f = 360.0 if requested_max_rot < 360.0 else requested_max_rot
    direction_clean = direction.strip().lower() if isinstance(direction, str) else "left"
    if direction_clean not in ("left", "right"):
        direction_clean = "left"
    speed_f = float(turn_speed) if isinstance(turn_speed, (int, float)) else 0.3
    threshold_f = float(confidence_threshold) if isinstance(confidence_threshold, (int, float)) else 0.5

    step_duration_s = step_f * math.pi / (180.0 * max(speed_f, 0.01))

    total_rotation = 0.0
    steps_taken = 0
    max_steps = int(math.ceil(max_rot_f / step_f)) if step_f > 0 else 0

    for _ in range(max_steps + 1):  # +1: check current frame before any turn
        result = await _find_object_in_frame(object_name_clean)
        if result["visible"] and result["confidence"] >= threshold_f:
            return {
                "status": "found",
                "object_name": object_name_clean,
                "total_rotation_deg": total_rotation,
                "steps_taken": steps_taken,
                "confidence": result["confidence"],
                "bbox": result["bbox"],
                "position": result["position"],
            }
        if steps_taken >= max_steps:
            break
        await _turn_step_async(direction_clean, speed_f, step_duration_s)
        total_rotation += step_f
        steps_taken += 1

    return {
        "status": "not_found",
        "object_name": object_name_clean,
        "total_rotation_deg": total_rotation,
        "steps_taken": steps_taken,
        "confidence": None,
        "bbox": None,
        "position": None,
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
