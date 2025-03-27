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
import lane_following as lf
import detect_cross as dc
import time





class I_HATE_THE_DUCKIEBOT_(DTROS):
    def __init__(self, vehicle_name):
        self.vehicle_name = vehicle_name
        
        # Lane detection variables
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Encoder variables
        self.last_left_ticks = None
        self.last_right_ticks = None
        self._left_distance_traveled = 0.0
        self._right_distance_traveled = 0.0
        
        # Parameters
        self.TICKS_PER_REV = 135
        self.WHEEL_RADIUS = 0.0318
        self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
        self.BASELINE = 0.077
        self.VELOCITY = 0.15
        self.OMEGA_SPEED = 2.5
        self.angular_vel = 2.6
        
        # Control parameters
        self.KP = 0.025  # Proportional gain
        self.KI = 0.0001  # Integral gain
        self.KD = 0.01    # Derivative gain
        self.TARGET_DISTANCE = 20  # meters
        
        # Variables for PID terms
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        
        # Publishers
        twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

        
        self.yellow_lane_pub = rospy.Publisher(f"/{self.vehicle_name}/yellow_lane", Float64, queue_size=1)
        self.white_lane_pub = rospy.Publisher(f"/{self.vehicle_name}/white_lane", Float64, queue_size=1)
        self.corss_line_detect_pub = rospy.Publisher(f"/{self.vehicle_name}/corss_line_detect", Float64, queue_size=1)
        self.color = rospy.Publisher(f"/{self.vehicle_name}/color", Float64, queue_size=1)
        self.image_pub = rospy.Publisher(f"/{self.vehicle_name}/camera_node/image/distorted_image/compressed", CompressedImage, queue_size=10)

        # Subscribers
        self.left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
        self.right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"

        
        self.sub_left_enc = rospy.Subscriber(self.left_encoder_topic, WheelEncoderStamped, self.cb_left_encoder)
        self.sub_right_enc = rospy.Subscriber(self.right_encoder_topic, WheelEncoderStamped, self.cb_right_encoder)
        self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.cb_camera)

        self.yellow_lower = np.array([25, 100, 100], np.uint8)
        self.yellow_upper = np.array([30, 255, 255], np.uint8)
        self.white_lower = np.array([0, 0, 200], np.uint8)
        self.white_upper = np.array([180, 30, 255], np.uint8)
        self.detect_cross = False



    def cb_camera(self, msg):
        if self.camera_matrix is None or self.distortion_coeffs is None:
            return
            
        # Process image for lane detection and visualization
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        undistorted_image = cv2.GaussianBlur(undistorted_image, (5, 5), 0)
        
        # Crop the image (e.g., lower half of 640x480 image)
        height, width = undistorted_image.shape[:2]
        crop_top = height // 2  # Start from halfway down (240 for 480 height)
        crop_bottom = height    # Go to the bottom (480)
        crop_left = 0           # Start from the left edge
        crop_right = width      # Go to the right edge (640)
        cropped_image = undistorted_image[crop_top:crop_bottom, crop_left:crop_right]
        # Detect lanes and mark centers on the cropped image
        yellow_pos, white_pos, processed_image = self.detect_lanes(cropped_image)
        
        # Adjust lane positions to account for cropping offset (only x-coordinate matters here)
        if yellow_pos is not None:
            yellow_pos += crop_left  # Adjust x-coordinate if crop_left != 0
        if white_pos is not None:
            white_pos += crop_left   # Adjust x-coordinate if crop_left != 0
        
        # Publish the processed cropped image
        undistorted_msg = self.bridge.cv2_to_compressed_imgmsg(processed_image)
        # self.image_pub.publish(undistorted_msg)
        
        # Publish lane detection results
        if yellow_pos is not None:
            self.yellow_lane_pub.publish(Float64(yellow_pos))
        if white_pos is not None:
            self.white_lane_pub.publish(Float64(white_pos))
    
    
    
    def detect_lanes(self, image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)
        
        yellow_center = None
        white_center = None

        # Detect yellow lane
        yellow_mask = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for contour in yellow_contours:
            if cv2.contourArea(contour) > 300:  # Adjust threshold based on dot size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
                    # Optionally draw each dot's center:
                    #cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)

        if centers:
            avg_cx = int(sum(pt[0] for pt in centers) / len(centers))
            avg_cy = int(sum(pt[1] for pt in centers) / len(centers))
            yellow_center = (avg_cx, avg_cy)
            # Draw the averaged center
            yellow_pos = avg_cx
            cv2.circle(image, yellow_center, 5, (0, 255, 255), -1)  # Yellow marker
        else:
            yellow_pos = None
            yellow_center = None

        # Now detect white lane and only consider white contours to the right of the yellow lane
        white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        white_pos = None
        white_y_value = None
        for contour in white_contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                center_white = x + w // 2
                # Only accept white contour if its center is to the right of the yellow lane's center
                if yellow_pos is None or center_white > yellow_pos:
                    white_pos = center_white
                    white_y_value = int(y + h // 2)
                    white_center = (center_white, white_y_value)
                    cv2.circle(image, white_center, 5, (255, 255, 255), -1)  # White marker
                    break  # Use the first valid white contour
                else:
                    white_pos = yellow_pos + 500   # Assume a fixed offset if white lane is to the left of yellow lane 

        if yellow_center is not None and white_pos is None:
            white_pos = yellow_pos + 500
        # Calculate and draw the lane center if both lane centers are detected
        if yellow_center is not None and white_center is not None:
            lane_center_x = (yellow_center[0] + white_center[0]) // 2
            lane_center_y = (yellow_center[1] + white_center[1]) // 2
            lane_center = (lane_center_x, lane_center_y)
            cv2.circle(image, lane_center, 5, (0, 0, 255), -1)  # Red marker for lane center
            cv2.putText(image, f"Line Center: {lane_center}", (lane_center_x + 10, lane_center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # Draw the screen center
        h, w = image.shape[:2]
        screen_center = (w // 2, h // 2)
        cv2.circle(image, screen_center, 5, (255, 0, 0), -1)  # Blue marker for screen center
        cv2.putText(image, f"Screen Center: {screen_center}", (screen_center[0] + 10, screen_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return yellow_pos, white_pos, image