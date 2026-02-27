"""Regression checks for TurtleBot prompt guidance."""

from pathlib import Path

import yaml


def test_turtlebot_prompt_includes_bbox_reorient_and_cautious_branch():
    template_path = Path(__file__).resolve().parents[1] / "mcp" / "prompts" / "prompt_templates" / "assistant_turtlebot.yaml"
    payload = yaml.safe_load(template_path.read_text())
    instruction = payload["instruction_template"]

    assert "tbot_camera_reorient_to_object" in instruction
    assert "bounding box" in instruction
    assert "ready_to_approach=true" in instruction
    assert "cautious approach is allowed" in instruction
