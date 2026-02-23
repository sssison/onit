"""
TurtleBot Search MCP Server V2.

Composite tool for continuous rotate-and-detect behavior.
"""

import asyncio
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

mcp_search_v2 = FastMCP("TurtleBot Search MCP Server V2")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float") from e


CAMERA_MCP_URL = os.getenv("TBOT_CAMERA_MCP_URL", "http://127.0.0.1:18206/turtlebot-camera-v2")
MOTION_MCP_URL = os.getenv("TBOT_MOTION_MCP_URL", "http://127.0.0.1:18205/turtlebot-motion-v2")
LIDAR_MCP_URL = os.getenv("TBOT_LIDAR_MCP_URL", "http://127.0.0.1:18208/turtlebot-lidar-v2")

VISION_TIMEOUT_S = _env_float("TBOT_SEARCH_VISION_TIMEOUT_S", 10.0)
MAX_CONSECUTIVE_ERRORS = max(1, int(_env_float("TBOT_SEARCH_MAX_CONSECUTIVE_ERRORS", 3)))
MAX_DETECTION_HISTORY = max(1, int(_env_float("TBOT_SEARCH_HISTORY_SIZE", 20)))

DEFAULT_VISION_HOST = os.getenv("TBOT_VISION_HOST") or os.getenv("ONIT_HOST", "http://127.0.0.1:8000/v1")
DEFAULT_VISION_MODEL = os.getenv("TBOT_VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
DEFAULT_VISION_API_KEY = os.getenv("TBOT_VISION_API_KEY", "EMPTY")


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


def _resolve_api_key(host: str) -> str:
    explicit_key = os.getenv("TBOT_VISION_API_KEY", DEFAULT_VISION_API_KEY)
    if "openrouter.ai" in host:
        if explicit_key and explicit_key != "EMPTY":
            return explicit_key
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("Search tool is using OpenRouter but no API key is configured.")
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
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
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


def _extract_structured_dict(tool_response: Any) -> dict[str, Any]:
    if isinstance(tool_response, dict):
        return tool_response

    structured = getattr(tool_response, "structured_content", None)
    if isinstance(structured, dict):
        return structured

    structured_camel = getattr(tool_response, "structuredContent", None)
    if isinstance(structured_camel, dict):
        return structured_camel

    content = getattr(tool_response, "content", None)
    if isinstance(content, list):
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parsed = _extract_first_json_object(text)
                if isinstance(parsed, dict):
                    return parsed

    return {}


async def _call_mcp_tool(client: Client, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = await client.call_tool(tool_name, arguments)
    parsed = _extract_structured_dict(response)
    if isinstance(parsed, dict):
        return parsed
    return {}


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


async def _detect_target_with_vision(
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


def _append_history(history: list[dict[str, Any]], item: dict[str, Any], max_items: int) -> None:
    history.append(item)
    if len(history) > max_items:
        del history[0 : len(history) - max_items]


@mcp_search_v2.tool()
async def tbot_search_while_rotating(
    target: str,
    timeout_s: float = 30.0,
    angular_speed: float = 0.25,
    linear_speed: float = 0.0,
    sample_period_s: float = 0.7,
    detection_confidence_threshold: float = 0.65,
    max_frames: int = 80,
    stop_on_collision: bool = True,
    collision_front_threshold_m: float = 0.25,
) -> dict[str, Any]:
    """
    Rotate continuously while sensing and stop when target object is detected.
    """
    target_clean = target.strip() if isinstance(target, str) else ""
    if target_clean == "":
        raise ValueError("target must be a non-empty string")

    timeout_value = _ensure_non_negative("timeout_s", timeout_s)
    angular_speed_value = _ensure_finite("angular_speed", angular_speed)
    linear_speed_value = _ensure_finite("linear_speed", linear_speed)
    sample_period_value = _ensure_non_negative("sample_period_s", sample_period_s)
    threshold_value = _ensure_non_negative("detection_confidence_threshold", detection_confidence_threshold)
    threshold_value = min(1.0, threshold_value)
    collision_front_threshold_value = _ensure_non_negative("collision_front_threshold_m", collision_front_threshold_m)

    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if sample_period_value <= 0:
        raise ValueError("sample_period_s must be > 0")

    vision_host = os.getenv("TBOT_VISION_HOST", DEFAULT_VISION_HOST)
    vision_model = os.getenv("TBOT_VISION_MODEL", DEFAULT_VISION_MODEL)
    vision_api_key = _resolve_api_key(vision_host)
    vision_client = AsyncOpenAI(base_url=vision_host, api_key=vision_api_key)

    status = "error"
    stop_reason = "unknown"
    found = False
    best_confidence: float | None = None
    best_frame_index: int | None = None
    frames_processed = 0
    consecutive_errors = 0
    detection_history: list[dict[str, Any]] = []
    collision_events: list[dict[str, Any]] = []
    motion_info: dict[str, Any] = {
        "requested_linear": linear_speed_value,
        "requested_angular": angular_speed_value,
        "angular_sign_applied": None,
    }

    start_mono = time.monotonic()

    try:
        async with Client(CAMERA_MCP_URL) as camera_client, Client(MOTION_MCP_URL) as motion_client:
            lidar_context = Client(LIDAR_MCP_URL) if stop_on_collision else None

            if lidar_context is None:
                lidar_client = None

                move_result = await _call_mcp_tool(
                    motion_client,
                    "tbot_motion_move",
                    {
                        "linear": linear_speed_value,
                        "angular": angular_speed_value,
                        "duration_s": None,
                    },
                )
                motion_info["angular_sign_applied"] = move_result.get("angular_sign")

                while True:
                    elapsed = time.monotonic() - start_mono
                    if elapsed >= timeout_value:
                        status = "timeout"
                        stop_reason = "timeout_reached"
                        break
                    if frames_processed >= max_frames:
                        status = "timeout"
                        stop_reason = "max_frames_reached"
                        break

                    loop_start = time.monotonic()

                    try:
                        camera_result = await _call_mcp_tool(
                            camera_client,
                            "tbot_camera_get_decoded_frame",
                            {
                                "wait_for_new_frame": True,
                                "wait_timeout_s": sample_period_value,
                                "include_base64": True,
                            },
                        )
                        image_base64 = camera_result.get("image_base64")
                        if not isinstance(image_base64, str) or image_base64.strip() == "":
                            raise RuntimeError("Camera tool did not return image_base64.")

                        detection = await _detect_target_with_vision(
                            vision_client=vision_client,
                            target=target_clean,
                            image_base64=image_base64,
                            timeout_s=VISION_TIMEOUT_S,
                            model=vision_model,
                        )

                        frames_processed += 1
                        history_item = {
                            "frame_index": frames_processed,
                            "confidence": detection["confidence"],
                            "matched": detection["matched"],
                        }
                        _append_history(detection_history, history_item, MAX_DETECTION_HISTORY)
                        consecutive_errors = 0

                        detection_confidence = float(detection["confidence"])
                        if best_confidence is None or detection_confidence > best_confidence:
                            best_confidence = detection_confidence
                            best_frame_index = frames_processed

                        if detection["matched"] and detection_confidence >= threshold_value:
                            found = True
                            status = "found"
                            stop_reason = "target_detected"
                            break

                    except Exception as e:
                        consecutive_errors += 1
                        _append_history(
                            detection_history,
                            {
                                "frame_index": frames_processed + 1,
                                "confidence": 0.0,
                                "matched": False,
                            },
                            MAX_DETECTION_HISTORY,
                        )
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            status = "error"
                            stop_reason = f"consecutive_failures: {e}"
                            break

                    remaining = sample_period_value - (time.monotonic() - loop_start)
                    if remaining > 0:
                        await asyncio.sleep(remaining)

            else:
                async with lidar_context as lidar_client:
                    move_result = await _call_mcp_tool(
                        motion_client,
                        "tbot_motion_move",
                        {
                            "linear": linear_speed_value,
                            "angular": angular_speed_value,
                            "duration_s": None,
                        },
                    )
                    motion_info["angular_sign_applied"] = move_result.get("angular_sign")

                    while True:
                        elapsed = time.monotonic() - start_mono
                        if elapsed >= timeout_value:
                            status = "timeout"
                            stop_reason = "timeout_reached"
                            break
                        if frames_processed >= max_frames:
                            status = "timeout"
                            stop_reason = "max_frames_reached"
                            break

                        loop_start = time.monotonic()

                        try:
                            collision_result = await _call_mcp_tool(
                                lidar_client,
                                "tbot_lidar_check_collision",
                                {
                                    "front_threshold_m": collision_front_threshold_value,
                                    "side_threshold_m": collision_front_threshold_value,
                                    "back_threshold_m": collision_front_threshold_value,
                                },
                            )
                            risk_level = collision_result.get("risk_level")
                            if risk_level == "stop":
                                collision_events.append(
                                    {
                                        "frame_index": frames_processed,
                                        "risk_level": risk_level,
                                        "scan_age_s": collision_result.get("scan_age_s"),
                                    }
                                )
                                status = "stopped_collision"
                                stop_reason = "collision_risk_stop"
                                break

                            camera_result = await _call_mcp_tool(
                                camera_client,
                                "tbot_camera_get_decoded_frame",
                                {
                                    "wait_for_new_frame": True,
                                    "wait_timeout_s": sample_period_value,
                                    "include_base64": True,
                                },
                            )
                            image_base64 = camera_result.get("image_base64")
                            if not isinstance(image_base64, str) or image_base64.strip() == "":
                                raise RuntimeError("Camera tool did not return image_base64.")

                            detection = await _detect_target_with_vision(
                                vision_client=vision_client,
                                target=target_clean,
                                image_base64=image_base64,
                                timeout_s=VISION_TIMEOUT_S,
                                model=vision_model,
                            )

                            frames_processed += 1
                            history_item = {
                                "frame_index": frames_processed,
                                "confidence": detection["confidence"],
                                "matched": detection["matched"],
                            }
                            _append_history(detection_history, history_item, MAX_DETECTION_HISTORY)
                            consecutive_errors = 0

                            detection_confidence = float(detection["confidence"])
                            if best_confidence is None or detection_confidence > best_confidence:
                                best_confidence = detection_confidence
                                best_frame_index = frames_processed

                            if detection["matched"] and detection_confidence >= threshold_value:
                                found = True
                                status = "found"
                                stop_reason = "target_detected"
                                break

                        except Exception as e:
                            consecutive_errors += 1
                            _append_history(
                                detection_history,
                                {
                                    "frame_index": frames_processed + 1,
                                    "confidence": 0.0,
                                    "matched": False,
                                },
                                MAX_DETECTION_HISTORY,
                            )
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                status = "error"
                                stop_reason = f"consecutive_failures: {e}"
                                break

                        remaining = sample_period_value - (time.monotonic() - loop_start)
                        if remaining > 0:
                            await asyncio.sleep(remaining)

    except Exception as e:
        status = "error"
        stop_reason = f"setup_or_runtime_error: {e}"
    finally:
        try:
            async with Client(MOTION_MCP_URL) as motion_client:
                await _call_mcp_tool(motion_client, "tbot_motion_stop", {})
        except Exception as stop_error:
            logger.warning("Failed to stop motion after search tool completion: %s", stop_error)

    elapsed_total = time.monotonic() - start_mono

    return {
        "status": status,
        "target": target_clean,
        "found": found,
        "best_confidence": best_confidence,
        "best_frame_index": best_frame_index,
        "frames_processed": frames_processed,
        "elapsed_s": elapsed_total,
        "stop_reason": stop_reason,
        "detection_history": detection_history,
        "collision_events": collision_events,
        "motion": motion_info,
        "final_action": "stopped",
    }


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 18209,
    path: str = "/turtlebot-search-v2",
    options: dict = {},
) -> None:
    """Run the TurtleBot Search MCP Server V2."""
    if "verbose" in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "Starting TurtleBot Search MCP V2 camera_url=%s motion_url=%s lidar_url=%s at %s:%s%s",
        CAMERA_MCP_URL,
        MOTION_MCP_URL,
        LIDAR_MCP_URL,
        host,
        port,
        path,
    )
    mcp_search_v2.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()

