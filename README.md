# AI 361 - Autonomous Robots Documentation (forked from onit)

## Pre-requisites

Follow the guide provided by [Robotis](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/).
- For Turtlebot3, focus on SBC setup. Make sure to create swap memory before building.
- For Jetson Orin, focus on PC setup. No need to install gazebo simulator.

## Basics - Bringup and Teleop

Once you were able to install the prerequisites, you can run the following commands to control the robot remotely:
1. SSH to the turtlebot `ssh sssison@10.158.38.26`
2. Run:
```
source ~/.bashrc
ros2 launch turtlebot3_bringup robot.launch.py
ros2 run turtlebot3_teleop teleop_keyboard
```

## Running the Video Server and Motion Server

1. SSH to the turtlebot.
2. Run bringup and camera, then run the motion server.
```
source ~/.bashrc
ros2 launch turtlebot3_bringup robot.launch.py
ros2 run camera\_ros camera\_node --ros-args -p width:=640 -p height:=480 -p format:="YUYV" -p rotation:=90
python3 motion_server/motion_server.py
```
3. Run the video server and motion server in jetson orin.
```
source ~/.bashrc
cd ~
cd onit-robot/
python video_server/video_server.py
```
4. Verify the video server through 'http://localhost:5000/video_feed'
5. Verify the motion server by running the command
```
curl -X POST http://10.158.38.26:5001/move -H "Content-Type: application/json" -d '{"linear": 0.5, "angular": 0.2}'
```
> Note: When verifying the motion server, make sure to use small velocities only ($<0.01$).
6. [Optional] Run the dashboard.
```
python3 -m http.server 8000
```
> Make sure to check the ports: 5001 for motion server, 5000 for video server, and 8000 for dashboard.

## Using On-it as the brain

1. Install on-it using this forked version. You can follow the instructions in the original versions.
2. Modify `configs/default.yaml` to use model served in DGX.
3. Add `motion_mcp_server.py` and `camera_mcp_server.py` to `src/mcp/turtlebot`.
4. Modify `src/mcp/servers/configs/default.yaml` to include new mcp servers as well as to `configs/default.yaml`.
5. Re-run bringup, camera, and motion server.
6. Run `onit --mcp`, then run `onit --text-show-logs`.

TODO:
- Fix MCP servers
- Add MCP server for LiDAR + depth estimation.
