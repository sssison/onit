import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class CompressedVideoViewer(Node):
    def __init__(self):
        super().__init__('compressed_video_viewer')
        
        # CONFIGURATION
        # Check your topic list! Common defaults:
        # '/camera/image_raw/compressed'
        # '/raspicam_node/image/compressed'
        # '/oakd/rgb/preview/image_raw/compressed'
        self.topic_name = '/camera/image_raw/compressed'
        
        # SUBSCRIPTION
        # We subscribe to CompressedImage, not standard Image
        self.subscription = self.create_subscription(
            CompressedImage,
            self.topic_name,
            self.listener_callback,
            10)
            
        self.get_logger().info(f'✅ Video Viewer (Compressed) started on {self.topic_name}')

    def listener_callback(self, msg):
        try:
            # 1. CONVERT ROS COMPRESSED MESSAGE TO OPENCV
            # The 'data' field in CompressedImage is just a list of bytes (jpeg/png)
            # We map this list to a numpy array of integers
            np_arr = np.frombuffer(msg.data, np.uint8)
            
            # 2. DECODE THE IMAGE
            # We let OpenCV decode the jpeg/png data back into a raw image matrix
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 3. DISPLAY
            if frame is not None:
                cv2.imshow("Jetson Compressed Feed", frame)
                
                # Check for 'q' key to quit manually
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.get_logger().info("Quit key pressed.")
                    rclpy.shutdown()
            else:
                self.get_logger().warn("Received empty frame!")

        except Exception as e:
            self.get_logger().error(f'Failed to process frame: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CompressedVideoViewer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
