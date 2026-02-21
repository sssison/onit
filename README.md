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

## Using OnIt as the brain (TurtleBot MCP V2)

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Start TurtleBot bringup and camera publisher (robot side)
```bash
source ~/.bashrc
ros2 launch turtlebot3_bringup robot.launch.py
ros2 run camera_ros camera_node --ros-args -p width:=640 -p height:=480 -p format:="YUYV" -p rotation:=90
```

### 3) Start motion HTTP server (robot side)
```bash
source ~/.bashrc
cd ~/onit-robot
python3 motion_server/motion_server_tbot.py
```

### 4) Set environment variables for MCP V2 services (OnIt host)
```bash
export MOTION_SERVER_BASE_URL="http://10.158.38.26:5001"
export CAMERA_TOPIC="/camera/image_raw/compressed"

# Vision server settings (OpenAI-compatible endpoint)
export TBOT_VISION_HOST="http://127.0.0.1:8000/v1"
export TBOT_VISION_MODEL="Qwen/Qwen3-VL-8B-Instruct"
export TBOT_VISION_API_KEY="EMPTY"
export TBOT_VISION_TIMEOUT_S="60"
```

Notes:
- `TBOT_VISION_API_KEY` can be omitted for local vLLM-style endpoints.
- For OpenRouter, set `TBOT_VISION_API_KEY` or `OPENROUTER_API_KEY`.

### 5) Start MCP servers
```bash
onit --mcp --config src/mcp/servers/configs/default.yaml
```

### 6) Start OnIt client
```bash
onit --config configs/default.yaml --text-show-logs
```

### 7) Sample tool-chain flow (camera -> vision -> motion)
1. `tbot_camera_get_decoded_frame(wait_for_new_frame=true, wait_timeout_s=1.0)`
2. `tbot_vision_analyze_scene(images=[<image_from_step_1>], task="Analyze scene for navigation and hazards.")`
3. `tbot_motion_move(linear=0.03, angular=0.0, duration_s=1.5)`
4. `tbot_motion_stop()`

The V2 configuration keeps legacy TurtleBot MCP entries disabled and enables:
- `TurtlebotMotionServerV2`
- `TurtlebotCameraServerV2`
- `TurtlebotVisionServerV2`
