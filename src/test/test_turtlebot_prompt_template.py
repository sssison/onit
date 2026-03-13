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
    assert "### Arrival lock and hard stop" in instruction
    assert "If the target was visible before motion and front LiDAR is now within stop_distance," in instruction
    assert "Do not re-run search/scan after arrival unless the user explicitly asks to continue exploring." in instruction


def test_turtlebot_prompt_includes_new_composite_patterns_and_no_removed_tools():
    instruction = _load_instruction()

    assert "PATTERN: FIND_AND_APPROACH <object>" in instruction
    assert "PATTERN: WALL_FOLLOW to <destination>" in instruction
    assert "tbot_navigate_to_object(" in instruction
    assert "tbot_nav_go_to_midpoint_between_objects(" in instruction
    assert "Use tbot_vision_find_object as the primary object finder." in instruction
    assert "It always checks the current frame first, then scans if needed." in instruction
    assert "Use explicit stop_distance when requested by the task (example: soccer ball at 0.50 m)." in instruction
    assert "tbot_estimate_object_pose(" in instruction
    assert "Find the destination landmark first" in instruction
    assert "fixed 15 deg steps" in instruction
    assert "Repeat up to 24 steps (full 360 deg sweep)" in instruction
    assert "If still not found after full sweep: stop and abort (no forward motion)" in instruction
    assert "Move toward the wall until near it" in instruction
    assert "Approach destination while verifying wall proximity with LiDAR" in instruction
    assert "Do not execute forward motion while the target object is not yet confirmed in frame." in instruction
    assert "If navigating toward an object and the object is visible with collision status clear," in instruction
    assert "execute tbot_navigate_to_object once, then call tbot_motion_stop and end that action cycle." in instruction
    assert "If a task includes directional language like \"on the left\" or \"on the right\" for the target:" in instruction
    assert "First do one in-place turn toward that side (left/right) before any forward motion." in instruction
    assert "Then run visual QA with tbot_vision_describe_scene to confirm the side context." in instruction
    assert "Then run FIND_AND_APPROACH for the target (find first, then navigate)." in instruction
    assert "If tbot_vision_find_object reports the target is visible, lock the target." in instruction
    assert "Do not run another search sweep after lock." in instruction
    assert "Rescan ONLY IF the object is not present in frame." in instruction
    assert "If the object is present but off-center, recenter using the bounding box and continue approach." in instruction
    assert "tbot_vision_get_object_bbox" in instruction
    assert "If nav.stopped_reason == \"target_lost\":" in instruction
    assert "Close-range fallback: if object was recently locked and front LiDAR <= stop_distance," in instruction
    assert "Immediately call tbot_motion_stop() when reached." in instruction
    assert "Do not restart search after reached_target." in instruction
    assert "Call tools only when the result changes the next action." in instruction
    assert "Avoid repeated identical calls without movement or state change between calls." in instruction
    assert "Never simulate or guess sensor readings. Always call LiDAR/vision/nav tools for measurements." in instruction
    assert "PATTERN: OBJECT_STATE_CHECK <object>" in instruction
    assert "PATTERN: QUALIFIED_FIND_AND_APPROACH <object>" in instruction
    assert "PATTERN: OBSTACLE_AROUND_TARGET <object>" in instruction
    assert "PATTERN: STOP_ON_PERSON_FEET" in instruction
    assert "PATTERN: UNDER_OBJECT_CHECK <object>" in instruction
    assert "PATTERN: VISUAL_QA_ONLY" in instruction
    assert "### Task Routing Guide" in instruction
    assert "Door open check: OBJECT_STATE_CHECK + FIND_AND_APPROACH" in instruction
    assert "Trashcan or cable bypass: OBSTACLE_AROUND_TARGET" in instruction
    assert "Two-object midpoint tasks (chairs, trashcan + soccer ball): MIDPOINT_NAVIGATE" in instruction
    assert "If the task asks to confirm \"no spills\" on the floor:" in instruction
    assert "Perform a full 360 deg spill sweep before concluding clear." in instruction
    assert "Use fixed 15 deg turn steps for 24 steps total:" in instruction
    assert "tbot_vision_inspect_floor(targets=[\"spill\"], region=\"lower_half\")" in instruction
    assert "Report \"no spills detected\" only if all 24 checks report no spill." in instruction

    assert "tbot_vision_health" not in instruction
    assert "tbot_lidar_health" not in instruction
    assert "tbot_motion_move_timed" not in instruction
    assert "tbot_lidar_is_path_clear" not in instruction
    assert "tbot_vision_scan_for_object" not in instruction
    assert "tbot_vision_search_and_approach_object" not in instruction
    assert "tbot_get_spatial_map" not in instruction
    assert "PATTERN: FIND_UPDATES_MAP" not in instruction
