# ROS package for sensing faces with InsightFace

## Overview

This package provides a client-serer interface via ROS to control face-sensing features like face detection or face recognition. There must be a topic to which raw camera images compressed/uncompressed are published which are taken to process face features in real time.

> Note: ROS Noetic and Python 3.8 was used to build this package

> Please install `onnxruntime-gpu` if cuda and cudnn versions are compatible