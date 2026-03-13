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
        "Return only JSON with keys: matched (boolean), confidence (0..1), "
        "bbox (object with normalized cx, cy, w, h or null)."
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
async def tbot_vision_find_object(
    object_name: str,
    attributes: list[str] | None = None,
    anchor_object: str | None = None,
    relation: str | None = None,
) -> dict[str, Any]:
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

    result = await _find_object_in_frame(
        object_name_clean,
        attributes=attributes,
        anchor_object=anchor_object,
        relation=relation,
    )
    return {**result, "model_info": model_info}


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
