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
from typing import Any

from fastmcp import FastMCP
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
DEFAULT_VISION_HOST = os.getenv("TBOT_VISION_HOST") or os.getenv("ONIT_HOST", "http://127.0.0.1:8000/v1")
DEFAULT_VISION_MODEL = os.getenv("TBOT_VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
DEFAULT_VISION_API_KEY = os.getenv("TBOT_VISION_API_KEY", "EMPTY")


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

