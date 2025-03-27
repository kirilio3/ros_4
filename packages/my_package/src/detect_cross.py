#!/usr/bin/env python3

import os
import math
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import ColorRGBA, Float64
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, LEDPattern
from sensor_msgs.msg import CompressedImage, CameraInfo
import signal
import sys
import cv2
import numpy as np
from cv_bridge import CvBridge
from dt_apriltags import Detector
from mother_of_all import MotherOfAll as MOA


class Detect_Corss(MOA):

    def __init__(self, 
                 vehicle_name, 
                 distortion_coeffs, 
                 camera_matrix,
                 debugger=False):
        # super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = vehicle_name
        self.debug = debugger
        # Lane detection variables
        self.bridge = CvBridge()
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.detect_corss_reached = False
        # Publishers

        self.image_pub = rospy.Publisher(f"/{self.vehicle_name}/camera_node/image/distorted_image/compressed", CompressedImage, queue_size=10)
        self.corss_line_detect_pub = rospy.Publisher(f"/{self.vehicle_name}/corss_line_detect", Float64, queue_size=1)
        self.pedestrian_detect_pub = rospy.Publisher(f"/{self.vehicle_name}/pedestrian_detect", Float64, queue_size=1)

        # Subscribers

        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"


        self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.cb_camera)

        # self.yellow_lower = np.array([20, 100, 100], np.uint8)
        # self.yellow_upper = np.array([30, 255, 255], np.uint8)
        # self.white_lower = np.array([0, 0, 200], np.uint8)
        # self.white_upper = np.array([180, 30, 255], np.uint8)

        self.cross_blue_lower = np.array([100, 150, 50], np.uint8)
        self.cross_blue_upper = np.array([130, 255, 255], np.uint8)

        self.duck_color_lower = np.array([14, 100, 175], np.uint8)
        self.duck_color_upper = np.array([19, 210, 255], np.uint8)
    
        self.has_pedestrian = False


    def camera_info_setter(self, camera_matrix, distortion_coeffs):
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs

    def cb_camera(self, msg):
        if self.camera_matrix is None or self.distortion_coeffs is None:
            return
            
        # Process image for lane detection and visualization
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        # undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        undistorted_image = cv2.GaussianBlur(image, (9, 9), 0)
        
        # Crop the image (e.g., lower half of 640x480 image)
        height, width = undistorted_image.shape[:2]
        crop_top = height // 2  # Start from halfway down (240 for 480 height)
        crop_bottom = height    # Go to the bottom (480)
        crop_left = 0           # Start from the left edge
        crop_right = width      # Go to the right edge (640)
        cropped_image = undistorted_image[crop_top:crop_bottom, crop_left:crop_right]
        # Detect lanes and mark centers on the cropped image
        # yellow_pos, white_pos, processed_image = self.detect_lanes(cropped_image)
        image = self.detect_corss(cropped_image)
        distorted_msg = self.bridge.cv2_to_compressed_imgmsg(image)
        if self.debug: self.debugger(distorted_msg)
        

            

    def detect_corss(self, image):
        total_con =0
        numberOfCon = 0
        detected = False
        # Convert image to HSV for blue shape detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the blue shape using your defined blue range
        blue_mask1 = cv2.inRange(hsv, self.cross_blue_lower, self.cross_blue_upper)
        
        # Dilate to fill gaps in the mask
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.dilate(blue_mask1, kernel, iterations=2)

        # Find contours from the mask
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        
        height, width, _ = image.shape
        
        # Loop over each contour found
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Check for significant contours
                total_con += area
                numberOfCon +=1
                x, y, w, h = cv2.boundingRect(contour)
                # Check that the bounding box is within image bounds
                if y + h < height and x + w < width and x > 0 and y > 0 and area/numberOfCon >1000:
                    # Draw a red rectangle around the detected blue shape
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # rospy.loginfo(area)
                    # Additional processing (e.g., distance calculation)
                    focal_length = 50  # Adjust based on your camera calibration
                    real_height_meters = 0.1  # Estimated height of the shape
                    pixel_height = h

                    # self.detect_corss_reached = True
                    # detected = True
                    # self.corss_line_detect_pub.publish(Float64(1))
                    # rospy.loginfo("corss detected, stopping the robot.")
                    if pixel_height > 0:
                        distance = abs((real_height_meters * focal_length) / pixel_height)
                        # rospy.loginfo(distance)
                        if distance < 0.1:  # If blue shape is close
                            self.detect_pedestrian(image)
                            self.detect_corss_reached = True
                            detected = True
                            self.corss_line_detect_pub.publish(Float64(1))
                            if not self.has_pedestrian: 
                                self.pedestrian_detect_pub.publish(Float64(0))
                            rospy.loginfo("corss detected, stopping the robot.")
        
        # If no contour met the condition, publish that no cross is detected.
        if not detected:
            self.corss_line_detect_pub.publish(Float64(0))
            self.detect_corss_reached = False
        
        return image  # Return the modified image


    def detect_pedestrian(self,image):

        detected = False
        # Convert image to HSV for blue shape detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the blue shape using your defined blue range
        duck_mask = cv2.inRange(hsv, self.duck_color_lower, self.duck_color_upper)
        
        # Dilate to fill gaps in the mask
        kernel = np.ones((5, 5), np.uint8)
        duck_mask = cv2.dilate(duck_mask, kernel, iterations=2)

        # Find contours from the mask
        contours, _ = cv2.findContours(duck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        
        height, width, _ = image.shape
        
        # Loop over each contour found
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Check for significant contours
                x, y, w, h = cv2.boundingRect(contour)
                # Check that the bounding box is within image bounds
                if y + h < height and x + w < width and x > 0 and y > 0:
                    # Draw a red rectangle around the detected blue shape
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    self.has_pedestrian = True
                    self.pedestrian_detect_pub.publish(Float64(1))



        
        return image  # Return the modified image

    def reached_corss_getter(self):
        return self.detect_corss_reached
    
    def debugger(self, undistorted_msg):
        self.image_pub.publish(undistorted_msg)

        