# ROS package for sensing faces with InsightFace

## Overview

This package provides a client-serer interface via ROS to control face-sensing features like face detection or face recognition. There must be a topic to which raw camera images compressed/uncompressed are published which are taken to process face features in real time.

> Note: ROS Noetic and Python 3.8 was used to build this package

> Please install `onnxruntime-gpu` if cuda and cudnn versions are compatible

## Config

* `model_dir` - the directory where the embeddings model is present or should be downloaded. Note that within this directory `models` directory should exist or will be created automatically where the actual model should be located.
* `model_name` - the name of the model to use for embeddings. If it is not present within `models` subdirectory, it will be downloaded automatically.