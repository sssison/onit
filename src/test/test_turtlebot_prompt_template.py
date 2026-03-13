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
    assert "### Collision guard - always active" in instruction
    assert "Before every tbot_motion_* call" in instruction


def test_turtlebot_prompt_includes_new_composite_patterns_and_no_removed_tools():
    instruction = _load_instruction()

    assert "PATTERN: FIND_AND_APPROACH <object>" in instruction
    assert "PATTERN: WALL_FOLLOW to <destination>" in instruction
    assert "tbot_navigate_to_object(" in instruction
    assert "tbot_estimate_object_pose(" in instruction
    assert "Find the destination landmark first" in instruction
    assert "fixed 15 deg steps" in instruction
    assert "Move toward the wall until near it" in instruction
    assert "Approach destination while verifying wall proximity with LiDAR" in instruction

    assert "tbot_vision_health" not in instruction
    assert "tbot_lidar_health" not in instruction
    assert "tbot_motion_move_timed" not in instruction
    assert "tbot_lidar_is_path_clear" not in instruction
    assert "tbot_vision_scan_for_object" not in instruction
    assert "tbot_vision_search_and_approach_object" not in instruction
