#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA
import cv2
from cv_bridge import CvBridge
import numpy as np
from dt_apriltags import Detector

# Define colors
RED = ColorRGBA(1.0, 0.0, 0.0, 1.0)    # Red for Stop Sign
BLUE = ColorRGBA(0.0, 0.0, 1.0, 1.0)   # Blue for T-Intersection
GREEN = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green for UofA Tag
WHITE = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # White for No Detection

class AprilTagLEDNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(AprilTagLEDNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        # Get vehicle name from environment
        self._vehicle_name = os.environ['VEHICLE_NAME']
        
        # Topics
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self._led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        
        # Bridge between OpenCV and ROS
        self._bridge = CvBridge()
        
        # Variables for camera calibration
        self._camera_matrix = None
        self._distortion_coeffs = None
        
        # Initialize AprilTag Detector
        self._detector = Detector(
            families='tag36h11',  # Duckietown standard
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        # Subscribers
        self.camera_info_sub = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback)
        
        # Publisher for LED control
        self.led_pub = rospy.Publisher(self._led_topic, LEDPattern, queue_size=1)
        
        # Default LED state (White - No Detection)
        self.set_led_color(WHITE, "WHITE")

    def camera_info_callback(self, msg):
        # Extract camera matrix and distortion coefficients
        self._camera_matrix = np.array(msg.K).reshape(3, 3)
        self._distortion_coeffs = np.array(msg.D)

    def image_callback(self, msg):
        if self._camera_matrix is None or self._distortion_coeffs is None:
            rospy.logwarn("Waiting for camera calibration parameters...")
            return
        
        # Convert compressed image to OpenCV format
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
        # Undistort the image
        undistorted_image = cv2.undistort(image, self._camera_matrix, self._distortion_coeffs)
        
        # Convert to grayscale for AprilTag detection
        bw_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
        
        # Extract camera parameters
        fx = self._camera_matrix[0, 0]  # Focal length x
        fy = self._camera_matrix[1, 1]  # Focal length y
        cx = self._camera_matrix[0, 2]  # Optical center x
        cy = self._camera_matrix[1, 2]  # Optical center y
        camera_params = (fx, fy, cx, cy)
        
        # Detect AprilTags
        tags = self._detector.detect(
            bw_image,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=0.065  # Duckietown tag size
        )
        
        # Process detected tags and update LEDs
        if tags:
            for tag in tags:
                tag_id = tag.tag_id
                # Map tag IDs to specific meanings (adjust IDs based on your setup)
                if tag_id == 21:  # Stop Sign
                    self.set_led_color(RED, "RED")
                    rospy.logwarn(f"Detected Stop Sign (ID: {tag_id}) - LEDs set to RED")
                elif tag_id == 59:  # T-Intersection 
                    self.set_led_color(BLUE, "BLUE")
                    rospy.logwarn(f"Detected T-Intersection (ID: {tag_id}) - LEDs set to BLUE")
                elif tag_id == 8:  # UofA Tag
                    self.set_led_color(GREEN, "GREEN")
                    rospy.logwarn(f"Detected UofA Tag (ID: {tag_id}) - LEDs set to GREEN")
                else:
                    self.set_led_color(WHITE, "WHITE")
                    rospy.logwarn(f"Unknown tag (ID: {tag_id}) - LEDs set to WHITE")
                break  # Process only the first detected tag for simplicity
        else:
            # No tags detected, set LEDs to white
            self.set_led_color(WHITE, "WHITE")
            # rospy.loginfo("No AprilTags detected - LEDs set to WHITE")

    def set_led_color(self, color, color_name):
        """Helper function to set LED color"""
        pattern = LEDPattern()
        pattern.color_list = [color_name] * 5  # Apply to all 5 LEDs
        pattern.rgb_vals = [color] * 5         # Set RGB values for all LEDs
        pattern.color_mask = [1, 1, 1, 1, 1]  # Affect all LEDs
        pattern.frequency = 0.0                # Steady light (no blinking)
        pattern.frequency_mask = [0, 0, 0, 0, 0]  # No frequency applied
        self.led_pub.publish(pattern)

    def on_shutdown(self):
        # Set LEDs to off (black) on shutdown
        off_color = ColorRGBA(0.0, 0.0, 0.0, 1.0)
        self.set_led_color(off_color, "OFF")
        rospy.loginfo("Shutting down AprilTagLEDNode - LEDs turned OFF")
        super(AprilTagLEDNode, self).on_shutdown()

if __name__ == '__main__':
    # Create and run the node
    node = AprilTagLEDNode(node_name='apriltag_led_node')
    rospy.spin()