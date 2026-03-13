"""Regression checks for TurtleBot prompt guidance."""

from pathlib import Path

import yaml


def _load_instruction() -> str:
    template_path = Path(__file__).resolve().parents[1] / "mcp" / "prompts" / "prompt_templates" / "assistant_turtlebot.yaml"
    payload = yaml.safe_load(template_path.read_text())
    return payload["instruction_template"]


def test_turtlebot_prompt_includes_sensor_priority_and_collision_guard():
    instruction = _load_instruction()

    assert "## Sensor priority rules" in instruction
    assert "Wall navigation - LiDAR is primary, vision is secondary" in instruction
    assert "### Collision guard - active for forward motion" in instruction
    assert "Before any motion call with a forward component" in instruction
    assert "For pure in-place turning/search rotations, do not call LiDAR collision checks." in instruction


def test_turtlebot_prompt_includes_new_composite_patterns_and_no_removed_tools():
    instruction = _load_instruction()

    assert "PATTERN: FIND_AND_APPROACH <object>" in instruction
    assert "PATTERN: WALL_FOLLOW to <destination>" in instruction
    assert "tbot_navigate_to_object(" in instruction
    assert "Use tbot_vision_find_object as the primary object finder." in instruction
    assert "It always checks the current frame first, then scans if needed." in instruction
    assert "tbot_estimate_object_pose(" in instruction
    assert "Find the destination landmark first" in instruction
    assert "fixed 15 deg steps" in instruction
    assert "Repeat up to 24 steps (full 360 deg sweep)" in instruction
    assert "If still not found after full sweep: stop and abort (no forward motion)" in instruction
    assert "Move toward the wall until near it" in instruction
    assert "Approach destination while verifying wall proximity with LiDAR" in instruction
    assert "Do not execute forward motion while the target object is not yet confirmed in frame." in instruction
    assert "If tbot_vision_find_object reports the target is visible, lock the target." in instruction
    assert "Do not run another search sweep after lock." in instruction
    assert "Rescan ONLY IF the object is not present in frame." in instruction
    assert "If the object is present but off-center, recenter using the bounding box and continue approach." in instruction
    assert "tbot_vision_get_object_bbox" in instruction
    assert "If nav.stopped_reason == \"target_lost\":" in instruction
    assert "Call tools only when the result changes the next action." in instruction
    assert "Avoid repeated identical calls without movement or state change between calls." in instruction

    assert "tbot_vision_health" not in instruction
    assert "tbot_lidar_health" not in instruction
    assert "tbot_motion_move_timed" not in instruction
    assert "tbot_lidar_is_path_clear" not in instruction
    assert "tbot_vision_scan_for_object" not in instruction
    assert "tbot_vision_search_and_approach_object" not in instruction
