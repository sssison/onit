# TurtleBot V3 MCP Obstacle-Bypass Plan

## Goal
Add a motion MCP tool that can route around blocking objects using LiDAR, for tasks like:
- Patrol around chairs
- Navigate around a trash can blocking an aisle

## New Tool
- `tbot_motion_bypass_obstacle(...)` in `src/mcp/turtlebot_v3/motion_mcp_server.py`

## Maneuver Logic (Implemented)
1. Read LiDAR distances (`front`, `left`, `right`, `rear`).
2. Choose bypass side (`left` / `right` / `auto`).
3. Compute turn angle from distance geometry:
   - Use front distance and side clearance deficit.
   - Clamp angle to configurable min/max.
4. Rotate toward chosen side.
5. Move forward until robot is parallel to the obstacle:
   - Front is clear enough.
   - LiDAR confirms stable forward progress.
6. Rotate back to original heading.
7. Move forward again until front path is clear past the obstacle.
8. Stop and return a structured status payload.

## Safety Behavior
- Uses collision interrupt distance while moving.
- Stops and returns status on:
  - `collision_risk`
  - `lidar_unavailable`
  - leg timeout
- Always sends stop command in cleanup.

## Scan-Step Update
- Vision search scan step reduced from 20° to 15° (`_SEARCH_STEP_DEG = 15.0`).
- Full 360° default scan now uses 24 steps.

## Suggested Next Iteration
- Add object-specific clearance profiles (chair, trash can, cart) and a planner wrapper that picks defaults by object class.
