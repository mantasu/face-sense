#!/usr/bin/env python

import rospy
import argparse

from face_sense.recognize import RecognizeServer, RecognizeClient

DEFAULT_CONFIG_PATH = "config.json"

def parse_args():
    """Parses command line arguments

    Gets the location for the JSON configuration file for face sense
    properties, the argument telling if the path is relative to this ros
    package location, i.e., is inside this package or should be treated
    as an absolute path, and the frequency at which to publish
    information about the processed images.

    Returns:
        tuple: A tuple containing config.json file path, whether the
            path is relative to the package location and the frequency
            at which the information about the faces should be published
    """
    # Create a parser for command args
    parser = argparse.ArgumentParser()

    # Add an argument for the JSON configuration path
    parser.add_argument("-p", "--config-path", type=str,
        default=DEFAULT_CONFIG_PATH, help="The path to JSON config file")

    # Add an argument for whether the path is in pkg
    parser.add_argument("--is-relative", type=bool,
        default=True, help="Whether the path is relative to this package")
    
    # Add an argument for the frequency value
    parser.add_argument("--frequency", type=int,
        default=30, help="Frequency at which to publish ROS messages")

    # Parse known command-line arguments
    args, _ = parser.parse_known_args()

    return args.config_path, args.is_relative, args.frequency

def run_recognition(config_path, is_relative, frequency):
    # Publisher rate and camera
    interval = rospy.Rate(frequency)
    server = RecognizeServer(config_path, is_relative)
    client = RecognizeClient(config_path, is_relative)
    client.start()

    while not rospy.is_shutdown():
        # While not down
        server.publish()
        interval.sleep()

        if rospy.is_shutdown():
            # If server stopped
            client.stop()
        
        if server.goal_handler.end:
            # If client stopped
            break

if __name__ == "__main__":
    # Get cmd-line args
    args = parse_args()

    rospy.init_node("recognition", anonymous=True)
    run_recognition(*args)
