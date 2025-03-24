#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo

import cv2
from cv_bridge import CvBridge
import numpy as np

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self._undistorted_topic = f"/{self._vehicle_name}/camera_node/image/distorted_image/compressed"

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # variables to store camera matrix and distortion coefficients
        self._camera_matrix = None
        self._distortion_coeffs = None

        # construct subscriber for camera_info intrinsic parameters
        self.camera_info_sub = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.camera_info_callback)
        
        # construct subscriber for image topics
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        # Publisher for the undistorted image
        self.image_pub = rospy.Publisher(self._undistorted_topic, CompressedImage, queue_size=10)

    def camera_info_callback(self, msg):
        # Extract camera matrix (K) and distortion coefficients (D)
        self._camera_matrix = np.array(msg.K).reshape(3, 3)
        self._distortion_coeffs = np.array(msg.D)

    def callback(self, msg):
        if self._camera_matrix is None or self._distortion_coeffs is None:
            rospy.logwarn("Waiting for camera calibration parameters...")
            return
        
        # Convert JPEG bytes to OpenCV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Undistort the image using the camera calibration parameters
        undistorted_image = cv2.undistort(image, self._camera_matrix, self._distortion_coeffs)
        
        # Get image dimensions
        height, width = undistorted_image.shape[:2]
        
        # Define the desired aspect ratio (width:height). For example, 4:3.
        desired_aspect_ratio = 2.0 / 3.0
        desired_width = int(height * desired_aspect_ratio)
        
        if desired_width < width:
            # Calculate horizontal offset to center the crop
            offset = (width - desired_width) // 2
            cropped_image = undistorted_image[:, offset:offset+desired_width]
        else:
            # If the image is not wider than the desired aspect ratio, use the undistorted image as is.
            cropped_image = undistorted_image

        # Resize the cropped image to a fixed size (e.g., 100x100)
        resized_image = cv2.resize(cropped_image, (100, 100))
        
        # Convert to black and white (grayscale)
        bw_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Convert the processed image back to a ROS CompressedImage message
        undistorted_msg = self._bridge.cv2_to_compressed_imgmsg(bw_image)

        # Publish the processed image
        self.image_pub.publish(undistorted_msg)


if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()
