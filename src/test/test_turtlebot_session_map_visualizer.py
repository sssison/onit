"""Tests for src/mcp/turtlebot_v3/session_map_visualizer.py."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import session_map_visualizer as vis


def test_resolve_snapshot_path_uses_default_location_when_env_missing(monkeypatch):
    monkeypatch.delenv("TBOT_SESSION_MAP_SNAPSHOT_PATH", raising=False)
    monkeypatch.delenv("TBOT_SESSION_MAP_SNAPSHOT_DIR", raising=False)

    path = vis.resolve_snapshot_path(session_id="demo")
    assert str(path).endswith("/tmp/onit/tbot_session_map_demo.json")


def test_load_snapshot_extract_state_and_render_text(tmp_path: Path):
    snapshot_path = tmp_path / "session.json"
    payload = {
        "session_id": "demo",
        "initial_pose": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        "current_pose": {"x_m": 1.0, "y_m": 0.5, "yaw_rad": 0.1},
        "robot_trail": [{"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}],
        "objects": [
            {
                "object_name": "chair",
                "qualifier": None,
                "estimated_object_pose": {"x_m": 2.0, "y_m": 1.0},
                "distance_m": 1.8,
                "confidence": "high",
            }
        ],
    }
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = vis.load_snapshot(snapshot_path)
    assert loaded is not None
    state = vis.extract_render_state(loaded)
    text = vis.render_text(state)
    assert "session_id: demo" in text
    assert "initial_pose" in text
    assert "chair" in text


def test_load_snapshot_returns_none_for_missing_or_invalid(tmp_path: Path):
    missing = tmp_path / "missing.json"
    assert vis.load_snapshot(missing) is None

    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    assert vis.load_snapshot(bad) is None
