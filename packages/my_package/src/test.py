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

class DShapeNode(DTROS):
    def __init__(self, node_name):
        super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
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
        self.VELOCITY = 0.3
        self.OMEGA_SPEED = 2.5
        self.angular_vel = 2.6
        self.bias = 1
        
        # Control parameters
        self.KP = 0.035  # Proportional gain for lane following
        self.TARGET_DISTANCE = 10  # meters

        # Publishers
        twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        
        self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        
        # Lane detection publishers
        self.yellow_lane_pub = rospy.Publisher(f"/{self.vehicle_name}/yellow_lane", Float64, queue_size=1)
        self.white_lane_pub = rospy.Publisher(f"/{self.vehicle_name}/white_lane", Float64, queue_size=1)
        
        # Subscribers
        self.left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
        self.right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.camera_info_topic = f"/{self.vehicle_name}/camera_node/camera_info"
        
        self.sub_left_enc = rospy.Subscriber(self.left_encoder_topic, WheelEncoderStamped, self.cb_left_encoder)
        self.sub_right_enc = rospy.Subscriber(self.right_encoder_topic, WheelEncoderStamped, self.cb_right_encoder)
        self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.cb_camera)
        self.sub_camera_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_camera_info)

        # Lane detection variables
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.yellow_lower = np.array([20, 100, 100], np.uint8)
        self.yellow_upper = np.array([30, 255, 255], np.uint8)
        self.white_lower = np.array([0, 0, 200], np.uint8)
        self.white_upper = np.array([180, 30, 255], np.uint8)
        
        # Red line detection variables
        self.red_line_reached = False
        self.red_lower1 = np.array([0, 150, 50], np.uint8)   # Lower bound for red
        self.red_upper1 = np.array([10, 255, 255], np.uint8) # Upper bound for red
        self.red_lower2 = np.array([170, 150, 50], np.uint8) # Second lower bound for red
        self.red_upper2 = np.array([180, 255, 255], np.uint8) # Second upper bound for red
    
        signal.signal(signal.SIGINT, self.signal_handler)

    def cb_camera_info(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)

    def cb_camera(self, msg):
        if self.camera_matrix is None or self.distortion_coeffs is None:
            return
            
        # Process image for lane and red line detection
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        
        # Detect lanes
        yellow_pos, white_pos = self.detect_lanes(undistorted_image)
        
        # Detect red line
        self.detect_red_line(undistorted_image)
        
        # Publish lane detection results
        if yellow_pos is not None:
            self.yellow_lane_pub.publish(Float64(yellow_pos))
        if white_pos is not None:
            self.white_lane_pub.publish(Float64(white_pos))

    def detect_lanes(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect yellow lane
        yellow_mask = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        yellow_pos = None
        for contour in yellow_contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                yellow_pos = x + w/2  # Center of yellow lane
                
        # Detect white lane
        white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        white_pos = None
        for contour in white_contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                white_pos = x + w/2  # Center of white lane
        
        return yellow_pos, white_pos

    def detect_red_line(self, image):
        # Convert image to HSV for red line detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create two masks for red and combine them
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Dilate to fill gaps in the mask
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest red object
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Check if the red line is fully in view and significant
            height, width, _ = image.shape
            if y + h < height and x + w < width and x > 0 and y > 0 and cv2.contourArea(largest_contour) > 300:
                # Calculate distance (simplified version)
                focal_length = 50  # Adjust based on your camera calibration
                real_height_meters = 0.1  # Estimated height of the red line
                pixel_height = h
                
                if pixel_height > 0:
                    distance = abs((real_height_meters * focal_length) / pixel_height)
                    if distance < 0.1:  # Stop if red line is close (adjust threshold as needed)
                        self.red_line_reached = True
                        rospy.loginfo("Red line detected, stopping the robot.")

    def cb_left_encoder(self, msg):
        current_ticks = msg.data
        if self.last_left_ticks is None:
            self.last_left_ticks = current_ticks
            return
        delta_ticks = current_ticks - self.last_left_ticks
        if delta_ticks > self.TICKS_PER_REV / 2:
            delta_ticks -= self.TICKS_PER_REV
        elif delta_ticks < -self.TICKS_PER_REV / 2:
            delta_ticks += self.TICKS_PER_REV
        self.last_left_ticks = current_ticks
        distance = (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC
        self._left_distance_traveled += distance

    def cb_right_encoder(self, msg):
        current_ticks = msg.data
        if self.last_right_ticks is None:
            self.last_right_ticks = current_ticks
            return
        delta_ticks = current_ticks - self.last_right_ticks
        if delta_ticks > self.TICKS_PER_REV / 2:
            delta_ticks -= self.TICKS_PER_REV
        elif delta_ticks < -self.TICKS_PER_REV / 2:
            delta_ticks += self.TICKS_PER_REV
        self.last_right_ticks = current_ticks
        distance = (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC
        self._right_distance_traveled += distance

    def set_led_color(self, color):
        pattern = LEDPattern()
        colors = {
            'GREEN': ColorRGBA(0, 1, 0, 1),
            'RED': ColorRGBA(1, 0, 0, 1),
            'CYAN': ColorRGBA(0, 1, 1, 1)
        }
        selected_color = colors.get(color, ColorRGBA(0.5, 0, 0.5, 1))
        pattern.color_list = [color] * 5
        pattern.rgb_vals = [selected_color] * 5
        pattern.color_mask = [1] * 5
        pattern.frequency = 1.0
        pattern.frequency_mask = [1] * 5
        self.led_pub.publish(pattern)

    def stop(self):
        msg = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd.publish(msg)

    def lane_follow(self):
        rospy.loginfo("Starting lane following until red line or 10 meters...")
        self.set_led_color('CYAN')
        
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
            # Stop if red line is reached or target distance exceeded
            if self.red_line_reached or avg_distance >= self.TARGET_DISTANCE:
                self.stop()
                self.set_led_color('RED')
                rospy.loginfo("Stopping: Red line reached or distance limit hit.")
                break
                
            # Get latest lane positions
            try:
                yellow_msg = rospy.wait_for_message(f"/{self.vehicle_name}/yellow_lane", Float64, timeout=1.0)
                white_msg = rospy.wait_for_message(f"/{self.vehicle_name}/white_lane", Float64, timeout=1.0)
            except rospy.ROSException as e:
                rospy.logwarn(f"Failed to get lane messages: {e}")
                yellow_msg = None
                white_msg = None
            
            if yellow_msg is not None and white_msg is not None:
                image_center = 320  # Assuming 640x480 image
                lane_center = (yellow_msg.data + white_msg.data) / 2
                error = image_center - lane_center
                rospy.loginfo(f"Error: {error}")
                omega = self.KP * error  # Proportional control
                
                if abs(error) > 42.5:
                    whiteLine = white_msg.data
                    lane_center = (yellow_msg.data + whiteLine) / 2
                    error = image_center - lane_center
                    rospy.loginfo(f"Turning Error: {error}")
                    omega = self.KP * error
                    omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                    cmd = Twist2DStamped(v=self.VELOCITY * 0.8, omega=omega)
                    self.pub_cmd.publish(cmd)
                else:
                    omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                    cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
                    self.pub_cmd.publish(cmd)
            else:
                # If lanes not detected, go straight slowly
                cmd = Twist2DStamped(v=self.VELOCITY / 3, omega=self.OMEGA_SPEED)
                self.pub_cmd.publish(cmd)
                
            rate.sleep()

    def on_shutdown(self):
        rospy.loginfo("Shutting down node...")
        self.stop()
        super(DShapeNode, self).on_shutdown()

    def signal_handler(self, sig, frame):
        rospy.loginfo("Ctrl+C detected, shutting down...")
        self.on_shutdown()
        sys.exit(0)

    def run(self):
        rospy.sleep(1)
        self.lane_follow()

if __name__ == "__main__":
    node = DShapeNode(node_name="d_shape_node")
    node.run()
    rospy.loginfo("DShapeNode main finished.")