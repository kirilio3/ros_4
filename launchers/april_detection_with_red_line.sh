#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch publisher
rosrun my_package april_detection_with_red_line.py

# wait for app to end
dt-launchfile-join