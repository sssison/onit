#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import threading
import os
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS so your HTML dashboard can view the feed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

last_frame = None
lock = threading.Lock()

class JetsonVideoServer(Node):
    def __init__(self):
        super().__init__('jetson_video_server')
        
        # Ensure this topic matches your TurtleBot's output
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed', 
            self.listener_callback,
            10)
        self.get_logger().info('FastAPI Jetson Server: Listening for Compressed Images...')

    def listener_callback(self, msg):
        global last_frame
        try:
            # 1. Decode compressed data
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 2. Rotate 90° clockwise
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)

            # 3. Re-encode to MJPEG (.jpg)
            # Lowering quality to 40-50 helps with latency over Wi-Fi
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            ret, buffer = cv2.imencode('.jpg', cv_image, encode_param)
            
            if ret:
                with lock:
                    last_frame = buffer.tobytes()
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

def generate_frames():
    """Generator function for the video stream."""
    while True:
        with lock:
            if last_frame is None:
                continue
            frame = last_frame
        
        # Standard MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def frame_saver_loop():
    output_path = '/dev/shm/latest_frame.jpg'
    tmp_path = '/dev/shm/latest_frame.jpg.tmp'
    period = 1.0 / 30.0
    next_tick = time.monotonic()

    while True:
        with lock:
            frame = last_frame

        if frame is not None:
            with open(tmp_path, 'wb') as f:
                f.write(frame)
            os.replace(tmp_path, output_path)

        next_tick += period
        now = time.monotonic()
        sleep_time = next_tick - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            missed = int((now - next_tick) // period) + 1
            next_tick += missed * period

@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get('/')
async def index():
    return {"status": "Jetson Video Server Online", "endpoint": "/video_feed"}

def run_ros_spin():
    if not rclpy.ok():
        rclpy.init()
    node = JetsonVideoServer()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"ROS Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # Start ROS in background
    threading.Thread(target=run_ros_spin, daemon=True).start()
    threading.Thread(target=frame_saver_loop, daemon=True).start()
    
    # Start FastAPI with Uvicorn
    # host '0.0.0.0' allows access from your dashboard laptop
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level="info")
