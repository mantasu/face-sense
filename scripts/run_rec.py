#!/usr/bin/env python

import cv2
import rospy
from face_sense.utils import verify_path, load_dict
from face_sense.recognize.recognizer import Recognizer

def run_camera_viewer(config, freq=1):
    # Publisher rate and camera
    interval = rospy.Rate(freq)
    recognizer = Recognizer(config)

    while not rospy.is_shutdown():
        # While not shutdown
        recognizer.publish()
        interval.sleep()

    # Destroy opencv window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = load_dict(verify_path("config.json"))

    # Initialize and run the camera viewer node
    rospy.init_node("recognizer", anonymous=True)
    run_camera_viewer(config)