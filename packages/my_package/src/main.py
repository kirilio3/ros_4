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
class main(DTROS):

    def __init__(self, node_name):
        super(main, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']

        
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None


        ##############following functiinalities are added#####################
        self.LF = lf.Lane_Following(self.vehicle_name)
        self.DC = dc.Detect_Corss(self.vehicle_name, None,None, debugger=True)

        self.distance = 2
        
        
        # Publishers
        self.image_pub = rospy.Publisher(f"/{self.vehicle_name}/camera_node/image/distorted_image/compressed", CompressedImage, queue_size=10)
        
        # Subscribers

        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.camera_info_topic = f"/{self.vehicle_name}/camera_node/camera_info"

        # self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.cb_camera)
        self.sub_camera_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_camera_info)


    '''
        please set camera info here for every camera related node
    '''
    def cb_camera_info(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)
        self.DC.camera_info_setter(camera_matrix=self.camera_matrix, distortion_coeffs=self.distortion_coeffs)
        self.LF.camera_info_setter(camera_matrix=self.camera_matrix, distortion_coeffs=self.distortion_coeffs)



    
    

    def run(self):
        rate = 20
        #rospy.spin()
        done = self.LF.lane_follow(10,rate)

    def on_shutdown(self):
        self.LF.stop()
        super(main, self).on_shutdown()



if __name__ == '__main__':
    node = main(node_name="main")
    node.run()