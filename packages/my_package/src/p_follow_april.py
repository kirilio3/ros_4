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

class DShapeNode(DTROS):
    def __init__(self, node_name):
        super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        # Lane detection variables
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.red_line_reached = False
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
        
        self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        
        self.yellow_lane_pub = rospy.Publisher(f"/{self.vehicle_name}/yellow_lane", Float64, queue_size=1)
        self.white_lane_pub = rospy.Publisher(f"/{self.vehicle_name}/white_lane", Float64, queue_size=1)
        self.color = rospy.Publisher(f"/{self.vehicle_name}/color", Float64, queue_size=1)
        self.image_pub = rospy.Publisher(f"/{self.vehicle_name}/camera_node/image/distorted_image/compressed", CompressedImage, queue_size=10)

        # Subscribers
        self.left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
        self.right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.camera_info_topic = f"/{self.vehicle_name}/camera_node/camera_info"
        
        self.sub_left_enc = rospy.Subscriber(self.left_encoder_topic, WheelEncoderStamped, self.cb_left_encoder)
        self.sub_right_enc = rospy.Subscriber(self.right_encoder_topic, WheelEncoderStamped, self.cb_right_encoder)
        self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.cb_camera)
        self.sub_camera_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_camera_info)

        self.yellow_lower = np.array([20, 100, 100], np.uint8)
        self.yellow_upper = np.array([30, 255, 255], np.uint8)
        self.white_lower = np.array([0, 0, 200], np.uint8)
        self.white_upper = np.array([180, 30, 255], np.uint8)

        self.average_yellow_center = []
        self.average_white_center = []
        self.red_lower1 = np.array([0, 150, 50], np.uint8)   # Lower bound for red
        self.red_upper1 = np.array([10, 255, 255], np.uint8) # Upper bound for red
        self.red_lower2 = np.array([170, 150, 50], np.uint8) # Second lower bound for red
        self.red_upper2 = np.array([180, 255, 255], np.uint8) # Second upper bound for red




    ##################################### apriltage part ########################################s
        # self.apriltag_topic = f"/{self.vehicle_name}/camera_node/image/apriltag_detections/compressed"


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
        # self.apriltag_pub = rospy.Publisher(self.apriltag_topic, CompressedImage, queue_size=10)
        self.apriltag_number = 0
        self.apriltag_detected = False
        self.apriltag_dic = {"UoA": ['GREEN',94.0], "STOP": ['RED',163.0], "T-tran": ['BLUE', 15.0]}
        self.counter = 0
        self.last_color = 0
        signal.signal(signal.SIGINT, self.signal_handler)

    def cb_camera_info(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)


    def update_lane_centers(self,new_white, new_yellow):

        # Append new values
        self.average_white_center.append(new_white)
        self.average_yellow_center.append(new_yellow)
        
        # Ensure that each list has at most 10 elements by popping the oldest value
        if len(self.average_white_center) > 5:
            self.average_white_center.pop(0)
        if len(self.average_yellow_center) > 5:
            self.average_yellow_center.pop(0)

        #return the average of the lists yellow first and white
        return sum(self.average_yellow_center)/len(self.average_yellow_center), sum(self.average_white_center)/len(self.average_white_center)

    def cb_camera(self, msg):
        if self.camera_matrix is None or self.distortion_coeffs is None:
            return
            
        # Process image for lane detection and visualization
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        self.apriltage(undistorted_image)
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
        self.detect_red_line(cropped_image)
        
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

    def apriltage(self, undistorted_image):
        #image = self.bridge.compressed_imgmsg_to_cv2(msg)
        
        # Undistort the image using the camera calibration parameters
        #undistorted_image = cv2.undistort(image, self._camera_matrix, self._distortion_coeffs)
        
        # Resize the image to a fixed size
        # resized_image = cv2.resize(undistorted_image, (320, 240))
        
        # Convert to black and white (grayscale)
        bw_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

        # Extract camera parameters for dt_apriltags
        fx = self.camera_matrix[0, 0]  # focal length x
        fy = self.camera_matrix[1, 1]  # focal length y

        cx = self.camera_matrix[0, 2]  # optical center x
        cy = self.camera_matrix[1, 2]  # optical center y
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
        # output_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored drawings
        for tag in tags:
            # Draw bounding box
            # for i in range(4):
            #     pt1 = (int(tag.corners[i-1][0]), int(tag.corners[i-1][1]))
            #     pt2 = (int(tag.corners[i][0]), int(tag.corners[i][1]))
            #     cv2.line(output_image, pt1, pt2, (0, 255, 0), 2)
        ####################################################################
            self.apriltag_detected = True
            self.apriltag_number = tag.tag_id
            
        ############################ Part 1.3.c ############################
            #Draw tag ID
            # tag_center = (int(tag.center[0]), int(tag.center[1]))  # Use tag.center for center position
            # cv2.putText(output_image, str(tag.tag_id), tag_center, 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # # Convert the processed image back to a ROS CompressedImage message
        # processed_msg = self.bridge.cv2_to_compressed_imgmsg(output_image)

        # # Publish the processed image with AprilTag detections
        # self.image_pub.publish(processed_msg)

        # # Part 1.3.d: Publish the augmented image with AprilTag detections to the new topic
        # self.apriltag_pub.publish(processed_msg)
 
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

    def detect_lanes(self, image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)
        
        yellow_center = None
        white_center = None


        # # Detect white lane
        # white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        # white_contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # white_pos = None
        # white_y_value = None
        # for contour in white_contours:
        #     if cv2.contourArea(contour) > 300:
        #         x, y, w, h = cv2.boundingRect(contour)
        #         white_pos = x + w // 2
        #         white_y_value = int(y + h // 2)
        #         white_center = (int(x + w // 2), int(y + h // 2))
        #         cv2.circle(image, white_center, 5, (255, 255, 255), -1)  # White marker
        #         break  # Use the first large enough contour

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
            'CYAN': ColorRGBA(0, 1, 1, 1),
            'WHITE': ColorRGBA(1, 1, 1, 1),
            'BLUE': ColorRGBA(0, 0, 1, 1),
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

    def update_led(self, new_led):
        color = new_led
        color_map = {
            94: 'GREEN',
            163: 'RED',
            15: 'BLUE'
        }
        print(f"{color} {self.last_color}")
        if color != self.last_color:
            self.set_led_color(color_map.get(color, 'WHITE'))
            self.last_color = color

    def lane_follow(self):
        # while (True):
        #     pass
        rospy.loginfo("Starting lane following for 1.3 meters with PID control...")
        
        rate = rospy.Rate(20)  # 20 Hz
        self.prev_time = rospy.get_time()
        
        while not rospy.is_shutdown():
            avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2

            if self.counter >= 6:
                rospy.sleep(1.5)
                self.stop()
                self.set_led_color('WHITE')
                rospy.loginfo("Target distance reached!")
                break
            
            if self.red_line_reached:
                self.counter+=1
                stop_time = 0.5
                if self.apriltag_detected:
                    if self.apriltag_number == self.apriltag_dic["UoA"][1]:
                        stop_time = 1
                    elif self.apriltag_number == self.apriltag_dic["STOP"][1]:
                        stop_time = 3
                    elif self.apriltag_number == self.apriltag_dic["T-tran"][1]:
                        stop_time = 2
                rospy.sleep(1.5)
                self.stop()
                rospy.sleep(stop_time)
                # cmd = Twist2DStamped(v=self.VELOCITY*1.5, omega=-0.3)
                # self.pub_cmd.publish(cmd)
                self.set_led_color('WHITE')
                # rospy.sleep(2)
                self.red_line_reached = False
                self.apriltag_number = 0
                self.apriltag_detected = False

            

            self.update_led(self.apriltag_number)
            try:
                yellow_msg = rospy.wait_for_message(f"/{self.vehicle_name}/yellow_lane", Float64, timeout=1.0)
                white_msg = rospy.wait_for_message(f"/{self.vehicle_name}/white_lane", Float64, timeout=1.0)
                # yellow_value, white_value = self.update_lane_centers(yellow_msg.data, white_msg.data)
                # print(yellow_value, white_value)
            except rospy.ROSException as e:
                rospy.logwarn(f"Failed to get lane messages: {e}")
                yellow_msg = None
                white_msg = None
            
            if yellow_msg is not None and white_msg is not None:
                lane_center = (yellow_msg.data + white_msg.data) / 2
                image_center = 320  # Assuming 640x480 image

                # Calculate error
                error = image_center - lane_center
                
                # Calculate time difference
                current_time = rospy.get_time()
                dt = current_time - self.prev_time if self.prev_time is not None and current_time > self.prev_time else 0.001
                
                # Calculate integral term
                self.integral += error * dt
                
                # Calculate derivative term
                error_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
                
                # PID control signal
                # KP = 0.0075, KI = 0.0001, KD = 0.01
                omega = (self.KP * error) + (self.KI * self.integral) + (self.KD * error_derivative)
                rospy.loginfo(f"Error: {error}, Integral: {self.integral}, Derivative: {error_derivative}, Omega: {omega}")

                # Conditional velocity adjustment based on error magnitude
                # if error > 90:
                #     omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                #     cmd = Twist2DStamped(v=self.VELOCITY * 2, omega=omega)
                # else:
                #     omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                #     cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
                omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)

                # Publish control command
                
                self.pub_cmd.publish(cmd)
                
                # Update previous values
                self.prev_error = error
                self.prev_time = current_time
            else:
                # If lanes not detected, go straight slowly
                cmd = Twist2DStamped(v=self.VELOCITY/1.5 , omega=self.OMEGA_SPEED*2)
                self.pub_cmd.publish(cmd)
                rospy.logwarn("Lanes not detected, moving straight slowly")
                
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
        rospy.sleep(0.5)
        self.lane_follow()

if __name__ == "__main__":
    node = DShapeNode(node_name="d_shape_node")
    node.run()
    rospy.loginfo("DShapeNode main finished.")