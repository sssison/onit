#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import threading
import uvicorn
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for the dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the target velocity
target_linear = 0.0
target_angular = 0.0

class MotionPublisher(Node):
    def __init__(self):
        super().__init__('motion_server_node')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # 10Hz Heartbeat to keep the TurtleBot from timing out
        self.timer = self.create_timer(0.1, self.publish_continuous)
        self.get_logger().info('FastAPI Motion Server: Continuous Publisher Started')

    def publish_continuous(self):
        msg = Twist()
        msg.linear.x = float(target_linear)
        msg.angular.z = float(target_angular)
        self.publisher_.publish(msg)

@app.post("/move")
async def handle_move(data: dict = Body(...)):
    global target_linear, target_angular
    
    # Update global targets (the timer thread picks these up)
    target_linear = data.get('linear', 0.0)
    target_angular = data.get('angular', 0.0)
    
    return {
        "status": "updated", 
        "linear": target_linear, 
        "angular": target_angular
    }

@app.post("/stop")
async def handle_stop():
    global target_linear, target_angular
    target_linear = 0.0
    target_angular = 0.0
    return {"status": "stopped"}

@app.get("/health")
async def health_check():
    return {"status": "online", "linear": target_linear, "angular": target_angular}

def run_ros_spin():
    if not rclpy.ok():
        rclpy.init()
    ros_node = MotionPublisher()
    try:
        rclpy.spin(ros_node)
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # Start ROS in background
    threading.Thread(target=run_ros_spin, daemon=True).start()
    
    # Run FastAPI
    # Port 5001 to match your previous dashboard configuration
    uvicorn.run(app, host='0.0.0.0', port=5001)
