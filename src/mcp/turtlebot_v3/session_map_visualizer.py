"""
Session map visualizer for TurtleBot Nav V3.

Reads session map snapshots written by nav_mcp_server and renders a live view.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _sanitize_session_fragment(session_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_id)


def resolve_snapshot_path(session_id: str, snapshot_path: str | None = None) -> Path:
    if isinstance(snapshot_path, str) and snapshot_path.strip():
        return Path(snapshot_path.strip())
    env_path = os.getenv("TBOT_SESSION_MAP_SNAPSHOT_PATH", "").strip()
    if env_path:
        safe_id = _sanitize_session_fragment(session_id)
        if "{session_id}" in env_path:
            return Path(env_path.format(session_id=safe_id))
        if safe_id == "default":
            return Path(env_path)
        base, ext = os.path.splitext(env_path)
        suffix = ext if ext else ".json"
        return Path(f"{base}_{safe_id}{suffix}")
    snapshot_dir = os.getenv("TBOT_SESSION_MAP_SNAPSHOT_DIR", "/tmp/onit").strip() or "/tmp/onit"
    safe_id = _sanitize_session_fragment(session_id)
    return Path(snapshot_dir) / f"tbot_session_map_{safe_id}.json"


def load_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def extract_render_state(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {
            "session_id": None,
            "initial_pose": None,
            "current_pose": None,
            "robot_trail": [],
            "objects": [],
        }
    return {
        "session_id": snapshot.get("session_id"),
        "initial_pose": snapshot.get("initial_pose"),
        "current_pose": snapshot.get("current_pose"),
        "robot_trail": snapshot.get("robot_trail", []) if isinstance(snapshot.get("robot_trail"), list) else [],
        "objects": snapshot.get("objects", []) if isinstance(snapshot.get("objects"), list) else [],
    }


def render_text(state: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"session_id: {state.get('session_id')}")
    lines.append(f"initial_pose: {state.get('initial_pose')}")
    lines.append(f"current_pose: {state.get('current_pose')}")
    objects = state.get("objects", [])
    if isinstance(objects, list) and objects:
        lines.append("objects:")
        for item in objects:
            if not isinstance(item, dict):
                continue
            pose = item.get("estimated_object_pose")
            lines.append(
                f"- {item.get('object_name')} ({item.get('qualifier')}): "
                f"pose={pose}, distance={item.get('distance_m')}, confidence={item.get('confidence')}"
            )
    else:
        lines.append("objects: []")
    return "\n".join(lines)


def run_text_loop(path: Path, refresh_s: float) -> None:
    last_mtime: float | None = None
    while True:
        snapshot = load_snapshot(path)
        mtime = path.stat().st_mtime if path.exists() else None
        if mtime != last_mtime:
            last_mtime = mtime
            state = extract_render_state(snapshot)
            print("\n" + "=" * 72)
            print(render_text(state))
        time.sleep(refresh_s)


def run_matplotlib_loop(path: Path, refresh_s: float) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    last_mtime: float | None = None

    def _draw() -> None:
        nonlocal last_mtime
        mtime = path.stat().st_mtime if path.exists() else None
        if mtime == last_mtime:
            return
        last_mtime = mtime
        snapshot = load_snapshot(path)
        state = extract_render_state(snapshot)
        ax.clear()
        ax.set_title(f"TurtleBot Session Map ({state.get('session_id')})")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        initial_pose = state.get("initial_pose")
        if isinstance(initial_pose, dict):
            ix = initial_pose.get("x_m")
            iy = initial_pose.get("y_m")
            if isinstance(ix, (int, float)) and isinstance(iy, (int, float)):
                ax.scatter([ix], [iy], marker="*", s=180, c="tab:green", label="initial")

        trail = state.get("robot_trail", [])
        trail_x: list[float] = []
        trail_y: list[float] = []
        if isinstance(trail, list):
            for point in trail:
                if not isinstance(point, dict):
                    continue
                x = point.get("x_m")
                y = point.get("y_m")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    trail_x.append(float(x))
                    trail_y.append(float(y))
        if trail_x and trail_y:
            ax.plot(trail_x, trail_y, "-", color="tab:blue", alpha=0.5, label="trail")

        current_pose = state.get("current_pose")
        if isinstance(current_pose, dict):
            cx = current_pose.get("x_m")
            cy = current_pose.get("y_m")
            yaw = current_pose.get("yaw_rad")
            if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
                ax.scatter([cx], [cy], marker="o", s=100, c="tab:blue", label="robot")
                if isinstance(yaw, (int, float)):
                    ax.arrow(
                        float(cx),
                        float(cy),
                        0.20 * float(math.cos(float(yaw))),
                        0.20 * float(math.sin(float(yaw))),
                        head_width=0.06,
                        head_length=0.08,
                        color="tab:blue",
                        length_includes_head=True,
                    )

        objects = state.get("objects", [])
        if isinstance(objects, list):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                pose = item.get("estimated_object_pose")
                if not isinstance(pose, dict):
                    continue
                ox = pose.get("x_m")
                oy = pose.get("y_m")
                if not isinstance(ox, (int, float)) or not isinstance(oy, (int, float)):
                    continue
                ax.scatter([ox], [oy], marker="x", s=100, c="tab:red")
                label = str(item.get("object_name", "object"))
                qual = item.get("qualifier")
                if isinstance(qual, str) and qual.strip():
                    label = f"{label} ({qual.strip()})"
                ax.text(float(ox) + 0.05, float(oy) + 0.05, label, fontsize=9, color="tab:red")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right")

    timer = fig.canvas.new_timer(interval=max(50, int(refresh_s * 1000)))
    timer.add_callback(_draw)
    timer.start()
    _draw()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live visualizer for TurtleBot session map snapshots.")
    parser.add_argument("--session-id", default="default", help="Session id used in snapshot file naming.")
    parser.add_argument("--snapshot-path", default="", help="Explicit snapshot json path (overrides env/default).")
    parser.add_argument("--refresh-ms", type=int, default=250, help="Refresh interval in milliseconds.")
    parser.add_argument("--text", action="store_true", help="Force text-only mode.")
    args = parser.parse_args()

    refresh_s = max(0.05, float(args.refresh_ms) / 1000.0)
    snapshot_path = resolve_snapshot_path(session_id=str(args.session_id), snapshot_path=args.snapshot_path or None)
    if args.text:
        run_text_loop(snapshot_path, refresh_s)
        return

    try:
        run_matplotlib_loop(snapshot_path, refresh_s)
    except Exception:
        run_text_loop(snapshot_path, refresh_s)


if __name__ == "__main__":
    main()
