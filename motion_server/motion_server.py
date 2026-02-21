#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- Add this

app = Flask(__name__)
CORS(app)  # <--- Add this line right after creating 'app'

# Global variables for the target velocity
target_linear = 0.0
target_angular = 0.0

class MotionPublisher(Node):
    def __init__(self):
        super().__init__('motion_server_node')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # Create a timer that runs 10 times per second (0.1s)
        self.timer = self.create_timer(0.1, self.publish_continuous)
        self.get_logger().info('Motion Server: Continuous Publisher Started')

    def publish_continuous(self):
        # This function runs automatically every 0.1 seconds
        msg = Twist()
        msg.linear.x = float(target_linear)
        msg.angular.z = float(target_angular)
        self.publisher_.publish(msg)

# Global ROS Node reference
ros_node = None

@app.route('/move', methods=['POST'])
def handle_move():
    global target_linear, target_angular
    data = request.get_json()
    
    # Update the global target values
    # The background timer will pick these up automatically
    target_linear = data.get('linear', 0.0)
    target_angular = data.get('angular', 0.0)

    return jsonify({"status": "updated", "linear": target_linear, "angular": target_angular}), 200

@app.route('/stop', methods=['POST'])
def handle_stop():
    global target_linear, target_angular
    # Reset targets to zero
    target_linear = 0.0
    target_angular = 0.0
    return jsonify({"status": "stopped"}), 200

@app.get("/health")
async def health_check():
    return {"status": "online", "linear": target_linear, "angular": target_angular}

def run_ros_spin():
    rclpy.init()
    global ros_node
    ros_node = MotionPublisher()
    rclpy.spin(ros_node)
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    threading.Thread(target=run_ros_spin, daemon=True).start()
    app.run(host='0.0.0.0', port=5001)
