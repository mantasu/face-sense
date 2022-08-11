#!/usr/bin/env python

import cv2
import rospy
import argparse
from face_sense.utils import verify_path, load_dict
from face_sense.recognize.recognize_viewer import CameraViewer

def run_camera_viewer(config, freq=30):
    # Publisher rate and camera
    interval = rospy.Rate(freq)
    camera_viewer = CameraViewer(config)

    while not rospy.is_shutdown():
        # While not down
        interval.sleep()

    # Destroy opencv window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = load_dict(verify_path("config.json"))

    # Initialize and run the camera viewer node
    rospy.init_node("camera_viewer", anonymous=True)
    run_camera_viewer(config)
    