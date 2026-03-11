"""Regression checks for TurtleBot prompt guidance."""

from pathlib import Path

import yaml


def test_turtlebot_prompt_includes_bbox_reorient_and_cautious_branch():
    template_path = Path(__file__).resolve().parents[1] / "mcp" / "prompts" / "prompt_templates" / "assistant_turtlebot.yaml"
    payload = yaml.safe_load(template_path.read_text())
    instruction = payload["instruction_template"]

    assert "tbot_vision_scan_for_object" in instruction
    assert "bbox" in instruction
    assert "Approach with Standoff" in instruction
    assert "standoff_m = 0.20" in instruction


def test_turtlebot_prompt_uses_single_instruction_template_with_default_standoff():
    template_path = Path(__file__).resolve().parents[1] / "mcp" / "prompts" / "prompt_templates" / "assistant_turtlebot.yaml"
    payload = yaml.safe_load(template_path.read_text())
    instruction = payload["instruction_template"]

    assert "v3_instruction_template" not in payload
    assert "You are a TurtleBot robot agent. Complete the following task:" in instruction
    assert "standoff_m = 0.20" in instruction
    assert "(V3)" not in instruction
