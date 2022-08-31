# ROS package for sensing faces with InsightFace

## Overview

This package provides a client-serer interface via ROS to control face-sensing features like face detection or face recognition. There must be a topic to which raw camera images compressed/uncompressed are published which are taken to process face features in real time.

> Note: ROS Noetic and Python 3.8 was used to build this package

> Please install `onnxruntime-gpu` if cuda and cudnn versions are compatible

## Structure

```
data
    identities
        embeds
            *.pkl
        photos
            Name Surname/*.jpg
        test
            *.jpg
        videos_input
            *.mp4
        videos_output
            *.mp4
    models
        buffalo_l
            *.onnx
        recognizer
            *.pth
    performance
        *.json
```

## Config
* `camera_topic` - the name of the topic to which the camera images are published
* `is_compressed` - whether the image is sent as compressed. If true, [CompressedImage](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CompressedImage.html) message will be expected, otherwise a regular [Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) message
* `service_name` - the name of the server which listens for client commands (e.g., to train a model, to perform face recognition)
* `info_topic` - the name of the topic to which the information generated from recognizing the face (as part of the server response) is published to. The type of the published messaged is [FaceInfo.msg](msg/FaceInfo.msg)
* `model_dir` - the directory where the embeddings model is present or should be downloaded. Note that within this directory `models` directory should exist or will be created automatically where the actual model should be located.
* `model_name` - the name of the model to use for embeddings. If it is not present within `models` subdirectory, it will be downloaded automatically.


Face analysis:
* `model_name` - Model specified for Face App should be trained separately, should it be used for commercial purposes. Otherwise, any valid specification from [model zoo](link) is fine as the model will be downloaded automatically.

Tunable:
* `sim_threshold` - threshold for similarity value. It is a minimum value the similarity function should yield when comparing the identified face with its counterparts in the face database. Otherwise, the detected face will be labeled as "Unknown".
* `prob_threshold` - threshold for probability value. It is a minimum value the model should achieve when classifying which identity the captured face belongs to. Otherwise, the detected face will be labeled as "Unknown".
* `num_to_compare` - the number of counterpart faces in the database the detected face to compare with to determine the mean similarity value.
