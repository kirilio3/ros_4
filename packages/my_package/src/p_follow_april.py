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
        
        self.left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
        self.right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"

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
        self.VELOCITY = 0.2
        self.OMEGA_SPEED = 2.5
        self.angular_vel = 2.6
        self.bias = 1
        
        # Control parameters
        self.KP = 0.025  # Proportional gain for lane following
        # self.KP = 0.035  # Proportional gain for lane following
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
        # self.left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
        # self.right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
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
    
        signal.signal(signal.SIGINT, self.signal_handler)

    def cb_camera_info(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)

    def cb_camera(self, msg):
        if self.camera_matrix is None or self.distortion_coeffs is None:
            return
            
        # Process image for lane detection
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        yellow_pos, white_pos = self.detect_lanes(undistorted_image)
        
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
        rospy.loginfo("Starting lane following for 1.5 meters...")
        self.set_led_color('CYAN')
        
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
            if avg_distance >= self.TARGET_DISTANCE:
                self.stop()
                break
                
            # Get latest lane positions (these will be updated by cb_camera)
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
                # Calculate error and control signal    
                error = image_center - lane_center
                rospy.loginfo(f"Error: {error}")
                omega = self.KP * error  # Proportional control
                if abs(error) > 140:
                # if abs(error) > 42.5 :
                    whiteLine = white_msg.data
                    lane_center = (yellow_msg.data + whiteLine) / 2
                    # Calculate error and control signal    
                    error = image_center - lane_center
                    rospy.loginfo(f"turning Error: {error}")
                    omega = self.KP * error  # Proportional control

                    omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                    
                    # Publish control command
                    cmd = Twist2DStamped(v=self.VELOCITY * 1.6 , omega=omega*1.4)
                    self.pub_cmd.publish(cmd)

                # elif error > 100 and error < 130:
                # # if abs(error) > 42.5 :
                #     whiteLine = white_msg.data
                #     lane_center = (yellow_msg.data + whiteLine) / 2
                #     # Calculate error and control signal    
                #     error = image_center - lane_center
                #     rospy.loginfo(f"turning Error: {error}")
                #     omega = self.KP * error  # Proportional control

                #     omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                    
                #     # Publish control command
                #     cmd = Twist2DStamped(v=self.VELOCITY * 1.8 , omega=omega*1.3)
                #     self.pub_cmd.publish(cmd)

                elif error < 0 and error > -5:
                    whiteLine = white_msg.data
                    lane_center = (yellow_msg.data + whiteLine) / 2
                    # Calculate error and control signal    
                    error = image_center - lane_center
                    rospy.loginfo(f"turning Error: {error}")
                    omega = self.KP * error  # Proportional control

                    omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                    
                    # Publish control command
                    cmd = Twist2DStamped(v=self.VELOCITY*1.5 , omega=omega*1.5)
                    self.pub_cmd.publish(cmd)
                
                elif error < -6 and error > -30:
                    whiteLine = white_msg.data
                    lane_center = (yellow_msg.data + whiteLine) / 2
                    # Calculate error and control signal    
                    error = image_center - lane_center
                    rospy.loginfo(f"turning Error: {error}")
                    omega = self.KP * error  # Proportional control

                    omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                    
                    # Publish control command
                    cmd = Twist2DStamped(v=self.VELOCITY*2 , omega=omega*2)
                    self.pub_cmd.publish(cmd)
                    
                else:
                    # Bound the angular velocity
                    omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                    
                    # Publish control command
                    cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
                    self.pub_cmd.publish(cmd)
            else:
                # If lanes not detected, go straight slowly
                cmd = Twist2DStamped(v=self.VELOCITY/3, omega=self.OMEGA_SPEED)
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
