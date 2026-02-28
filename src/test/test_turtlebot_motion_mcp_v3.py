"""Tests for src/mcp/turtlebot_v3/motion_mcp_server.py."""

import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, patch

import pytest


def _install_motion_test_stubs() -> None:
    """Install lightweight module stubs when ROS/FastMCP deps are unavailable."""
    if "rclpy" not in sys.modules:
        rclpy_mod = types.ModuleType("rclpy")
        rclpy_mod._ok = False

        def _ok() -> bool:
            return bool(rclpy_mod._ok)

        def _init() -> None:
            rclpy_mod._ok = True

        rclpy_mod.ok = _ok
        rclpy_mod.init = _init
        sys.modules["rclpy"] = rclpy_mod

    if "rclpy.node" not in sys.modules:
        rclpy_node_mod = types.ModuleType("rclpy.node")

        class _DummyPublisher:
            def __init__(self):
                self.messages = []

            def publish(self, msg):
                self.messages.append(msg)

        class _DummyNode:
            def __init__(self, _name: str):
                self._name = _name

            def create_publisher(self, _msg_type, _topic: str, _qos):
                return _DummyPublisher()

        rclpy_node_mod.Node = _DummyNode
        sys.modules["rclpy.node"] = rclpy_node_mod

    if "geometry_msgs.msg" not in sys.modules:
        geometry_msgs_mod = types.ModuleType("geometry_msgs")
        geometry_msgs_msg_mod = types.ModuleType("geometry_msgs.msg")

        class _Vec:
            def __init__(self):
                self.x = 0.0
                self.y = 0.0
                self.z = 0.0

        class Twist:
            def __init__(self):
                self.linear = _Vec()
                self.angular = _Vec()

        geometry_msgs_msg_mod.Twist = Twist
        geometry_msgs_mod.msg = geometry_msgs_msg_mod
        sys.modules["geometry_msgs"] = geometry_msgs_mod
        sys.modules["geometry_msgs.msg"] = geometry_msgs_msg_mod

    if "fastmcp" not in sys.modules:
        fastmcp_mod = types.ModuleType("fastmcp")

        class Client:
            def __init__(self, _url: str):
                self._url = _url

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def call_tool(self, _tool_name: str, _args: dict):
                return {}

        class FastMCP:
            def __init__(self, _name: str):
                self._name = _name

            def tool(self):
                def _decorator(fn):
                    return fn

                return _decorator

            def run(self, **_kwargs):
                return None

        fastmcp_mod.Client = Client
        fastmcp_mod.FastMCP = FastMCP
        sys.modules["fastmcp"] = fastmcp_mod


_install_motion_test_stubs()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mcp.turtlebot_v3 import motion_mcp_server as motion_v3


@pytest.mark.asyncio
async def test_move_without_duration_streams_until_stop():
    await motion_v3._set_continuous_motion(None)
    publishes: list[tuple[float, float]] = []

    async def fake_publish(linear: float, angular: float):
        publishes.append((linear, angular))
        return {"linear": linear, "angular": angular, "topic": motion_v3.CMD_VEL_TOPIC}

    try:
        with patch.object(motion_v3, "CONTROL_HZ", 200), patch.object(
            motion_v3, "_publish_twist", side_effect=fake_publish
        ):
            result = await motion_v3.tbot_motion_move(linear=0.1, angular=0.2)
            await asyncio.sleep(0.035)

            assert result["status"] == "streaming"
            assert motion_v3._continuous_motion_task is not None
            assert publishes.count((0.1, 0.2)) >= 2

            await motion_v3.tbot_motion_stop()
            assert motion_v3._continuous_motion_task is None
            assert publishes[-1] == (0.0, 0.0)

            count_after_stop = len(publishes)
            await asyncio.sleep(0.03)
            assert len(publishes) == count_after_stop
    finally:
        await motion_v3._set_continuous_motion(None)


@pytest.mark.asyncio
async def test_move_with_duration_preempts_continuous_stream():
    await motion_v3._set_continuous_motion(None)

    async def fake_publish(linear: float, angular: float):
        return {"linear": linear, "angular": angular, "topic": motion_v3.CMD_VEL_TOPIC}

    guard = AsyncMock(return_value={"status": "completed"})

    try:
        with patch.object(motion_v3, "_publish_twist", side_effect=fake_publish), patch.object(
            motion_v3, "_run_forward_with_collision_guard", guard
        ):
            await motion_v3.tbot_motion_move(linear=0.1, angular=0.0)
            assert motion_v3._continuous_motion_task is not None

            result = await motion_v3.tbot_motion_move(
                linear=0.05,
                angular=0.01,
                duration_s=0.5,
            )

            assert result["status"] == "completed"
            assert motion_v3._continuous_motion_task is None
            guard.assert_awaited_once_with(0.05, 0.01, 0.5)
    finally:
        await motion_v3._set_continuous_motion(None)


@pytest.mark.asyncio
async def test_move_rejects_non_positive_duration_when_provided():
    await motion_v3._set_continuous_motion(None)
    with pytest.raises(ValueError, match="duration_s must be > 0 when provided"):
        await motion_v3.tbot_motion_move(linear=0.1, angular=0.0, duration_s=0.0)


@pytest.mark.asyncio
async def test_turn_cancels_existing_continuous_stream():
    await motion_v3._set_continuous_motion(None)
    publishes: list[tuple[float, float]] = []

    async def fake_publish(linear: float, angular: float):
        publishes.append((linear, angular))
        return {"linear": linear, "angular": angular, "topic": motion_v3.CMD_VEL_TOPIC}

    try:
        with patch.object(motion_v3, "CONTROL_HZ", 200), patch.object(
            motion_v3, "_publish_twist", side_effect=fake_publish
        ):
            await motion_v3.tbot_motion_move(linear=0.1, angular=0.0)
            assert motion_v3._continuous_motion_task is not None

            result = await motion_v3.tbot_motion_turn(
                direction="left",
                speed=0.5,
                duration_seconds=0.03,
            )

            assert result["status"] == "completed"
            assert motion_v3._continuous_motion_task is None
            assert publishes[-1] == (0.0, 0.0)
    finally:
        await motion_v3._set_continuous_motion(None)
