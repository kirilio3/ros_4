#!/usr/bin/env python3

import os
import math
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, VehicleCorners
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import numpy as np
import signal
import sys

class DuckiebotManeuverNode(DTROS):
    def __init__(self, node_name):
        super(DuckiebotManeuverNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']
        # https://github.com/jihoonog/CMPUT-503-Exercise-4/blob/main/packages/duckiebot_detection/src/duckiebot_detection_node.py
        
        # Initialize CvBridge for image processing
        self.bridge = CvBridge()

        # Encoder variables
        self.last_left_ticks = None
        self.last_right_ticks = None
        self._left_distance_traveled = 0.0
        self._right_distance_traveled = 0.0
        self.TICKS_PER_REV = 135
        self.WHEEL_RADIUS = 0.0318
        self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS

        # Detection variables
        self.circle_matrix = [7, 3]  # 7x3 circle pattern on Duckiebot back
        self.blobdetector_min_area = 10
        self.blobdetector_min_dist_between_blobs = 2
        self.simple_blob_detector = self.setup_blob_detector()
        self.duckiebot_detected = False
        self.duckiebot_distance = None

        # Control parameters
        self.VELOCITY = 0.2  # m/s
        self.OMEGA_SPEED = 2.8  # rad/s for turning
        self.STOP_DISTANCE = 0.10  # Stop 0.10m from detected Duckiebot
        self.POST_PASS_DISTANCE = 0.6  # Drive 0.3m after passing
        self.state = "DRIVE_STRAIGHT"  # FSM states: DRIVE_STRAIGHT, STOP, MANEUVER, DRIVE_PASS, RETURN, LANE_FOLLOW, STOPPED

        # PID parameters for lane following
        self.KP = 0.025  # Proportional gain
        self.KI = 0.0001  # Integral gain
        self.KD = 0.01   # Derivative gain
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None

        # Lane detection variables
        self.yellow_lower = np.array([20, 100, 100], np.uint8)
        self.yellow_upper = np.array([30, 255, 255], np.uint8)
        self.white_lower = np.array([0, 0, 200], np.uint8)
        self.white_upper = np.array([180, 30, 255], np.uint8)

        # Publishers
        self.pub_cmd = rospy.Publisher(f"/{self.vehicle_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)

        # Subscribers
        self.sub_left_enc = rospy.Subscriber(f"/{self.vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_left_encoder)
        self.sub_right_enc = rospy.Subscriber(f"/{self.vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_right_encoder)
        self.sub_image = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_image, queue_size=1)

        self.log("Maneuver Node Initialized.")
        signal.signal(signal.SIGINT, self.signal_handler)

    def setup_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = self.blobdetector_min_area
        params.minDistBetweenBlobs = self.blobdetector_min_dist_between_blobs
        return cv2.SimpleBlobDetector_create(params)

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

    def cb_image(self, image_msg):
        image_cv = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
        (detection, centers) = cv2.findCirclesGrid(
            image_cv,
            patternSize=tuple(self.circle_matrix),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.simple_blob_detector,
        )

        self.duckiebot_detected = detection > 0
        if self.duckiebot_detected and centers is not None:
            focal_length = 50  # Adjust based on camera calibration
            real_height_meters = 0.1  # Approx height of Duckiebot pattern (10cm)
            pixel_height = max([c[0, 1] for c in centers]) - min([c[0, 1] for c in centers])
            if pixel_height > 0:
                self.duckiebot_distance = (real_height_meters * focal_length) / pixel_height
            else:
                self.duckiebot_distance = None
        else:
            self.duckiebot_distance = None

    def detect_lanes(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)

        # Detect yellow lane
        yellow_mask = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        yellow_pos = None
        if yellow_contours:
            largest_contour = max(yellow_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 300:
                x, y, w, h = cv2.boundingRect(largest_contour)
                yellow_pos = x + w // 2

        # Detect white lane
        white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        white_pos = None
        for contour in white_contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                center_white = x + w // 2
                if yellow_pos is None or center_white > yellow_pos:
                    white_pos = center_white
                    break
                else:
                    white_pos = yellow_pos + 500

        if yellow_pos is not None and white_pos is None:
            white_pos = yellow_pos + 500

        return yellow_pos, white_pos

    def stop(self):
        cmd = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd.publish(cmd)

    def drive_straight(self, velocity):
        cmd = Twist2DStamped(v=velocity, omega=0.0)
        self.pub_cmd.publish(cmd)

    def turn(self, omega, duration):
        cmd = Twist2DStamped(v=0.0, omega=omega)
        self.pub_cmd.publish(cmd)
        rospy.sleep(duration)
        self.stop()

    def lane_follow(self, target_distance, start_distance):
        rate = rospy.Rate(20)  # 20 Hz
        self.prev_time = rospy.get_time()

        while not rospy.is_shutdown():
            avg_distance = self.get_average_distance()
            if avg_distance - start_distance >= target_distance:
                return True

            image_msg = rospy.wait_for_message(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, timeout=1.0)
            image_cv = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
            yellow_pos, white_pos = self.detect_lanes(image_cv)

            if yellow_pos is not None and white_pos is not None:
                lane_center = (yellow_pos + white_pos) / 2
                image_center = image_cv.shape[1] // 2  # Assuming width of image

                error = image_center - lane_center
                current_time = rospy.get_time()
                dt = current_time - self.prev_time if self.prev_time is not None and current_time > self.prev_time else 0.001

                self.integral += error * dt
                error_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

                omega = (self.KP * error) + (self.KI * self.integral) + (self.KD * error_derivative)
                omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
                self.pub_cmd.publish(cmd)

                self.prev_error = error
                self.prev_time = current_time
            else:
                self.drive_straight(self.VELOCITY / 1.5)  # Slow down if lanes not detected

            rate.sleep()
        return False

    def maneuver(self):
        rate = rospy.Rate(20)  # 20 Hz
        maneuver_start_distance = self.get_average_distance()

        while not rospy.is_shutdown():
            avg_distance = self.get_average_distance()
            rospy.loginfo(f"State: {self.state}, Distance Traveled: {avg_distance:.2f}m")

            if self.state == "DRIVE_STRAIGHT":
                self.drive_straight(self.VELOCITY)
                if self.duckiebot_detected and self.duckiebot_distance is not None and self.duckiebot_distance <= self.STOP_DISTANCE:
                    self.state = "STOP"
                    self.stop()
                    rospy.loginfo(f"Duckiebot detected at {self.duckiebot_distance:.2f}m, stopping.")

            elif self.state == "STOP":
                rospy.sleep(3.0)  # Wait 3 seconds
                self.state = "MANEUVER"
                maneuver_start_distance = self.get_average_distance()
                rospy.loginfo("Waited 3 seconds, starting maneuver.")
                rospy.sleep(0.5)

            elif self.state == "MANEUVER":
                self.turn(self.OMEGA_SPEED * 1.3, 1.2)  # Turn left for 1.2sec
                self.state = "DRIVE_PASS"
                rospy.loginfo("Turned left, driving straight to pass.")

            elif self.state == "DRIVE_PASS":
                self.drive_straight(self.VELOCITY)
                rospy.sleep(2.0)  # Drive straight for 2s
                self.state = "RETURN"
                rospy.loginfo("Drove straight, returning to original lane.")

            elif self.state == "RETURN":
                self.turn(-self.OMEGA_SPEED, 3.0)  # Turn right for 3sec
                self.state = "LANE_FOLLOW"
                maneuver_end_distance = self.get_average_distance()
                rospy.loginfo("Turned right, starting lane following.")

            elif self.state == "LANE_FOLLOW":
                if self.lane_follow(self.POST_PASS_DISTANCE, maneuver_end_distance):
                    self.state = "STOPPED"
                    self.stop()
                    rospy.loginfo("Lane followed for 0.3m, stopping.")

            elif self.state == "STOPPED":
                rospy.loginfo("Task completed, node shutting down.")
                break

            rate.sleep()

    def get_average_distance(self):
        return (self._left_distance_traveled + self._right_distance_traveled) / 2

    def on_shutdown(self):
        self.stop()
        super(DuckiebotManeuverNode, self).on_shutdown()

    def signal_handler(self, sig, frame):
        rospy.loginfo("Ctrl+C detected, shutting down...")
        self.on_shutdown()
        sys.exit(0)

    def run(self):
        rospy.sleep(0.5)  # Brief delay to ensure subscribers are ready
        self.maneuver()

if __name__ == "__main__":
    node = DuckiebotManeuverNode(node_name="duckiebot_maneuver_node")
    node.run()
