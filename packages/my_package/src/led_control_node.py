#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA
from std_srvs.srv import SetBool, SetBoolResponse  # Simple service for toggling LED states

# Define colors
BLUE = ColorRGBA(0, 0, 1, 1)   # Blue
GREEN = ColorRGBA(0, 1, 0, 1)  # Green
CYAN = ColorRGBA(0.0, 1.0, 1.0, 1.0)
PURPLE = ColorRGBA(0.5, 0.0, 0.5, 1.0)

class LEDControlNode(DTROS):
    def __init__(self, node_name):
        super(LEDControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        vehicle_name = os.environ['VEHICLE_NAME']
        self.led_topic = f"/{vehicle_name}/led_emitter_node/led_pattern"

        # Publisher for LED control
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)

        # ROS Service to change LED state
        self.srv = rospy.Service('set_led_state', SetBool, self.handle_led_request)

    def handle_led_request(self, req):
        """ Callback for the service, toggles LED state on all LEDs """
        
        # Define colors for both states
        selected_color = CYAN if req.data else PURPLE

        pattern = LEDPattern()
        pattern.color_list = ["CYAN" if req.data else "PURPLE"] * 5  # 5 LEDs
        pattern.rgb_vals = [selected_color] * 5  # Apply color to all LEDs
        pattern.color_mask = [1, 1, 1, 1, 1]  # Affect all LEDs
        pattern.frequency = 1.0  # Optional: Blinking speed
        pattern.frequency_mask = [1, 1, 1, 1, 1]  # Apply frequency to all LEDs

        self.led_pub.publish(pattern)

        return SetBoolResponse(success=True, message=f"LEDs set to {'CYAN' if req.data else 'PURPLE'}")

    def run(self):
        # rospy.stop()
        rospy.spin()
    def on_shutdown(self):
        super(LEDControlNode, self).on_shutdown()

if __name__ == '__main__':
    node = LEDControlNode(node_name="led_control_node")
    node.run()

