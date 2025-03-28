# Apriltag Detection & Safety on Robots

This lab focused on several key aspects of mobile robotics, specifically Apriltag detection, pedestrian crosswalk interaction, and safe autonomous navigation for Duckiebots


## Part:

### 1. AprilTag Detection

Objectives here were to familiarize ourselves with the Apriltag library and understand how to detect and read these visual markers

This involved several steps, starting with subscribing to the robot's camera feed and undistorting the image using our camera calibration file. The same Distorted_Camera.py has been used that can be ran with the "camera-distorted" launcher. 

Image preprocessing was another important aspect, where we had to consider whether cropping the image or converting it to black and white would improve detection, and justify that choice in the report. 

The implementation required us to detect Apriltags (specifically from the tag36h11 family), draw bounding boxes around them, print the tag number, and publish a new image topic with these augmentations

Steps 1.2 and 1.3 of the assignment were combined in the April_Detection.py which can be ran with "april" launcher. 

Part 1.4 which was completed in Sign_Led.py and can be ran with "sign-led" launcher had a task of changing the Duckiebot's LEDs based on the detected Apriltag: red for a Stop Sign, blue for a T-Intersection, green for a UofA Tag, and white as the default state for no detection. The LEDs were to change as soon as a detection was made. All of that was done as requested.

Part 1.5: makeing the bot following the lane while detecting the apriltag. If the bot sees a red line it will stop for few seconds in front of it before it keeps going. The pausing second was determined by the apriltag it sees. 
1. stop sign for 3 second
2. cross sign for 2 second
3. UofA sign for 1 second
4. 0.5 second if no sign was seen

to run this code simply run "dts devel run -R csc22911 -L p-april"

### 2. PeDuckstrian Crosswalks
In this part, our bot will detect the cross-road and stop in front of it for one second if nothing is in the cross-road. However, if there are peDuckstrains it will not move until the peDuckstrains cross the road. The part was built with three files that are detect_cross.py, lane_following.py and main.py. The Detect_corss will detect the cross-road and peDuckstrains, the lane_following.py handles lane_following, and the main.py is for all the basic setup. To run this part, simply run "dts devel run -R csc22911 -L main" and it will handle everything from there. 


### 3. Safe Navigation

In this part we were tasked to handle hazards on the road, specifically a broken-down Duckiebot. The scenario involved approaching the broken bot from the rear at a distance of approximately 30 cm. We needed to detect the broken-down bot and explain our detection method (which we did in the report), including any other methods we tried. Upon detection, the robot was to pause to assess the situation by stopping at a "safe distance" for 3 seconds. Following this pause, we had to implement a method to maneuver around the broken-down bot by turning into the opposing lane, ensuring no contact was made. After successfully passing, the Duckiebot should transition back into the proper lane and continue driving for about 30 cm. 

This task was completed in maneuver.py and can be called with 'maneuver' launcher.

Dependency:
`dependencies-py3.txt` (dt_apriltags).

