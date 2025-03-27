#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo

import cv2
from cv_bridge import CvBridge
import numpy as np
from dt_apriltags import Detector

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._camera_info_topic = f"/{self._vehicle_name}/camera_node/camera_info"
        self._undistorted_topic = f"/{self._vehicle_name}/camera_node/image/distorted_image/compressed"
        # New topic for augmented image with AprilTag detections
        self._apriltag_topic = f"/{self._vehicle_name}/camera_node/image/apriltag_detections/compressed"

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # variables to store camera matrix and distortion coefficients
        self._camera_matrix = None
        self._distortion_coeffs = None

        # Initialize dt_apriltags Detector
        self._detector = Detector(
            families='tag36h11',  # Duckietown uses tag36h11 (Part 1.3.e)
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            searchpath=['apriltags'],
            debug=0
        )

        # construct subscriber for camera_info intrinsic parameters
        self.camera_info_sub = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.camera_info_callback)
        
        # construct subscriber for image topics
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        # Publisher for the processed image with AprilTag detections
        self.image_pub = rospy.Publisher(self._undistorted_topic, CompressedImage, queue_size=10)
        
        # New publisher for the augmented image with AprilTag detections
        self.apriltag_pub = rospy.Publisher(self._apriltag_topic, CompressedImage, queue_size=10)

    def camera_info_callback(self, msg):
        # Extract camera matrix (K) and distortion coefficients (D)
        self._camera_matrix = np.array(msg.K).reshape(3, 3)
        self._distortion_coeffs = np.array(msg.D)

    def callback(self, msg):
        if self._camera_matrix is None or self._distortion_coeffs is None:
            rospy.logwarn("Waiting for camera calibration parameters...")
            return
        
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
        # Undistort the image using the camera calibration parameters
        undistorted_image = cv2.undistort(image, self._camera_matrix, self._distortion_coeffs)
        
        # Resize the image to a fixed size
        # resized_image = cv2.resize(undistorted_image, (320, 240))
        
        # Convert to black and white (grayscale)
        bw_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

        # Extract camera parameters for dt_apriltags
        fx = self._camera_matrix[0, 0]  # focal length x
        fy = self._camera_matrix[1, 1]  # focal length y

        cx = self._camera_matrix[0, 2]  # optical center x
        cy = self._camera_matrix[1, 2]  # optical center y
        camera_params = (fx, fy, cx, cy)

        ############################ Part 1.3.a ############################
        # Detect AprilTags in the grayscale image 
        tags = self._detector.detect(
            bw_image,
            estimate_tag_pose=True,  # Enable pose estimation
            camera_params=camera_params,
            tag_size=0.065  # Duckietown tags are typically 6.5cm, adjust if different
        )
        ####################################################################

        ############################ Part 1.3.b ############################
        # Draw bounding boxes and tag IDs on the image
        output_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored drawings
        for tag in tags:
            # Draw bounding box
            for i in range(4):
                pt1 = (int(tag.corners[i-1][0]), int(tag.corners[i-1][1]))
                pt2 = (int(tag.corners[i][0]), int(tag.corners[i][1]))
                cv2.line(output_image, pt1, pt2, (0, 255, 0), 2)
        ####################################################################

        ############################ Part 1.3.c ############################
            # Draw tag ID
            tag_center = (int(tag.center[0]), int(tag.center[1]))  # Use tag.center for center position
            cv2.putText(output_image, str(tag.tag_id), tag_center, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # Convert the processed image back to a ROS CompressedImage message
        processed_msg = self._bridge.cv2_to_compressed_imgmsg(output_image)

        # Publish the processed image with AprilTag detections
        self.image_pub.publish(processed_msg)

        # Part 1.3.d: Publish the augmented image with AprilTag detections to the new topic
        self.apriltag_pub.publish(processed_msg)

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()