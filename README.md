# ROS package for sensing faces with InsightFace

## Overview

This package provides a client-serer interface via ROS to control face-sensing features like face detection and face recognition. There must be a topic to which raw camera images (compressed/uncompressed) are published which are taken to process face features in real time.

> This work is influenced by [Face Recognition](https://github.com/procrob/face_recognition) and [Face Recognition with InsightFace](https://github.com/tuna-date/Face-Recognition-with-InsightFace) repositories.

## Structure

### Node Structure

The topic structure is customizable but the default topic names are as follows:
* `/camera/raw_image` - the name of the topic to which raw camera frames are published to. Message can come in the for of either [CompressedImage](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CompressedImage.html) or [Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)
* `/recognizer/processed_image` - the name of the topic to which the processed images are published at a fixed interval (specified by launching the node). The image may contain a bounding box around the face, landmarks, name and confidence score, age and gender. The message can also be published either as [CompressedImage](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CompressedImage.html) or [Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)
* `/recognizer/face_info` - the name of the topic to which the face information is published. See [FaceInfo](msg/FaceInfo.msg) for more information

### Data Structure

This is the default data + dataset structure used (paths can be specified elsewhere in `config.json` file). In the _default_ case, the `data` folder contains photos, embeddings, model, performance etc files in a structured manner.

```bash
├── data
│   ├── identities      # People identity data
│   │   ├── embeds      # Face embedding files (.pkl)
│   │   └── photos      # Sub-directories with identity photos
│   │
│   ├── models          # Model folders
│   │   ├── model_1     # First model directory with model files (e.g., .onnx)
│   │   ├── ...
│   │   └── model_N     # Nth model directory with model files (e.g., .pth)
│   │
│   └── performance     # Training + validation performance files (.json)       
```

> **Note**: the current implementation expects `data/identities/photos` to contain sub-folders where each sub-folder represents a single person and is named accordingly.

## Requirements

### Environment

To install the dependencies, it is recommended that you set-up a virtual environment. Note that you can install these dependencies using `conda` as well (see [this](https://stackoverflow.com/questions/51042589/conda-version-pip-install-r-requirements-txt-target-lib)). For `pip` users, first install the **venv** package:

```shell
$ sudo apt-get install python3-venv
```

Then, within the cloned `scf4-control` directory, create the virtual environment and source into it (all the packages will then be installed there and if something gets messed up, you can safely remove the `venv` directory and recreate it):

```shell
$ python -m venv venv
$ source venv/bin/activate
```

Of course, if it is a product, just install the dependencies globally, `venv` is suggested to ensure everything runs smoothly and to test versions are compatibile.


### Apt Packages

For image messages, [cv_bridge](http://wiki.ros.org/cv_bridge) is required, please install the package:

```shell
$ sudo apt-get install ros-noetic-cv-bridge
```

To compile [onnx-runtime](https://onnxruntime.ai/docs/) please install:
```shell
$ sudo apt install protobuf-compiler
```

### Pip Packages

Just run the following to install the dependent libraries (note: please don't ignore the first line as there are some cases when packages cannot be successfully installed if **pip** version is not the newest):

```shell
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

> During the installation of _insightface_, for systems with less recent version, installation will take quite a bit of time but it will install everything

In some cases, the node may launch but then exit by itself saying there is an error producing an empty log file. If you experience this, delete the installed `opencv-python` and `numpy` packages and reinstall lower versions manually:

```shell
$ pip uninstall opencv-python numpy
$ pip install numpy==1.19.4
$ pip install opencv-python==4.6.0.66
```

If you want GPU support for _onnx_ (if _cuda_ and _cudnn_ versions are compatible), please uninstall `onnxruntime` and install `onnxruntime-gpu`. Just note that for some systems like **Jetson** there may be no repositories to install from in which case follow the installation procedure [here](https://elinux.org/Jetson_Zoo#ONNX_Runtime) (with manual wheel but note that for some technical reasons and poor version compatibility you may need to reinstall `numpy==1.19.4`).

### System

Please, set the python node scripts to be executable once cloned:
```
$ chmod +x scripts/recognition_node.py
```

Reminder: once cloned into `catkin_ws/src/face-sense`, don't forget to run `catkin_make` in `catkin_ws` directory!

> Note: ROS Noetic and Python 3.8 was used to build this package

## Running

### Starting the Node

To launch the node, just run with the optional parameters
```shell
$ roslaunch face_sense recognition.launch config_path:=config.json is_relative:=true frequency:=30
```

> For older systems, it may take a lot of time to load _onnx runtime_ environment (upt to 10 mins).

### Interacting
The node `recognition_node.py` launches **recognition** node which provides client-service interaction. If the `worker` in `config.json` section `node` is specified as `command_line`, all the requests and responses, as well as inputs, are printed in the
terminal. The following goals are accepted:
* `0` - generates a set of images and stores them in the identity folder. This order also takes an additional string argument which should be the name of the identity to capture the photos of.
* `1` - generates a face embeddings file which contains face embeddings from the identity folder. It is based on the face analysis app which is based on the specified face detection/analysis model.
* `2` - trains the face classifier to recognize face embedding given the embeddings dataset.
* `3` - recognizes face from the given image once. Given the face app specification and the classifier specification, it extracts information about the face, such as name, bounding box, age etc. Multiple faces are supported.
* `4` - recognizes the face in the renewing images continuously. Given the face app specification and the classifier specification, a thread is created where every specified interval of time information about the face is extracted. Multiple faces are supported.
* `5` - exits the recognition process. It is only an indicator for upper classes that there should be no more goals handled.

## Config

<details><summary><b><code>config.json</code></b></summary>

### General

* `camera_topic` - the name of the topic to which the camera images are published
* `is_compressed` - whether the received is sent as compressed. If true, [CompressedImage](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CompressedImage.html) message will be expected, otherwise a regular [Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) message

#### Node
* `photo_dir` - the path to the directory of sub-directories with identity photos. Each sub-directory corresponds to a single identity and is named accordingly. Each sub-directory contains _1 or more_ pictures of that identity's face
* `is_relative` - whether the `photo_dir` path is relative to the _face-sense_ package's path, i.e., is inside the package, or is an absolute path
* `num_photos` - the number of photos containing a single face to take when the client issues command to generate identity pictures
* `service_name` - the name of the server which listens for client commands (e.g., to train a model, to perform face recognition)
* `worker_type` - the type of the client worker which is run on a separate thread to issue commands for the server. Currently, only `command_line` is supported - a user is asked to input goal IDs in the terminal
* `info_topic` - the name of the topic to which the information generated from recognizing the face (as part of the server response) is published to. The type of the published messaged is [FaceInfo.msg](msg/FaceInfo.msg)
* `view_topic` - the name of the topic to which the processed images, i.e., raw camera images where bounding boxes, landmarks, names etc are drawn for each face, are published
* `drawable` - the list of drawable items on a frame that was processed to recognize faces. A list can consist of `["box", "marks", "name", "bio"]`
* `is_compressed` - whether the processed image is sent as compressed. If true, [CompressedImage](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CompressedImage.html) message will be sent, otherwise a regular [Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) message
* `format` - the format of the processed image to send. Only works if `is_compressed` is set to `true`
* `process_interval` - the interval (in seconds) at which a current received frame is processed by the recognition methods

### Inference

**Data**
* `embed_dir` - the path to the directory which contains `.pkl` files of face embeddings (generated through face analysis app)
* `model_dir` - the path to the directory which contains `.pth` files of model parameters (generated by training a face classifier)
* `embed_name` - the name of the embeddings file in the `embed_dir`. It could also be set to either `"newest"` or `"oldest"` in which case the most or least recent modified file will be chosen
* `model_name` - the name of the model file in the `model_dir`. It could also be set to either `"newest"` or `"oldest"` in which case the most or least recent modified file will be chosen
* `is_relative` - whether `embed_dir` and `model_dir` paths are relative to this package's path or are absolute paths

**Face analysis**
* `model_dir` - the directory where the embeddings model is present or should be downloaded. Note that within this directory `models` directory should exist or will be created automatically where the actual model should be located.
* `model_name` - the name of the model to use for embeddings. Model specified for Face App should be trained separately, should it be used for commercial purposes. Otherwise, any valid specification from [model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) is fine because the model, if not present in `models` subdirectory, will be downloaded automatically
* `is_relative` - whether `model_dir` path is relative to this package's path
* `ctx_id` - the ID of the device context to use for computation. Anything below `0` will result in _CPU_ context
* `det_size` - the window size [`width`, `height`] at which the face should be detected (in pixels)

**Model**
* `name` - the name of the face classifier to use. The current supported one is `"FaceClassifierBasic"`. Note that the subsequent parameters must match with the model that was trained, otherwise the parameters for this specified model will not be loaded
* `device` - the device on which to load the classifier to perform inference. Typical choices are either `cpu` or `cuda:0`
* `in_shape` - the input shape of the face embedding. This depends on the _Face Analysis_ app that generated the embeddings
* `num_classes` - the number of identities with sets of face pictures. This corresponds to the number of sub-folders in `photo_dir`
* `hidden_shape` - the list of the number of neurons in each hidden layer

**Tunable**
* `sim_threshold` - threshold for similarity value. It is a minimum value the similarity function should yield when comparing the identified face with its counterparts in the face database. Otherwise, the detected face will be labeled as "Unknown".
* `prob_threshold` - threshold for probability value. It is a minimum value the model should achieve when classifying which identity the captured face belongs to. Otherwise, the detected face will be labeled as "Unknown".
* `num_to_compare` - the number of counterpart faces in the database the detected face to compare with to determine the mean similarity value.

### Learn

**Data**
* `photo_dir`: the path to the directory of sub-directories with identity photos. Each sub-directory corresponds to a single identity and is named accordingly. Each sub-directory contains _1 or more_ pictures of that identity's face,
* `embed_dir`: the path to the directory which contains `.pkl` files of face embeddings (generated through face analysis app),
* `model_dir`: the path to the directory where the trained model (classifier) parameters should be saved (`.pth` file)
* `performance_dir`: the path to the performance directory. The training and validation performance over time will be saved there in `.json` format
* `embed_name`: the name of the embeddings file in the `embed_dir` to use for training. It could also be set to either `"newest"` or `"oldest"` in which case the most or least recent modified file will be chosen
* `is_relative`: whether `photo_dir`, `embed_dir`, `model_dir` and `performance_dir` paths are relative to this package's path or are absolute paths

**Face Analysis**
* `model_dir` - the directory where the embeddings model is present or should be downloaded. Note that within this directory `models` directory should exist or will be created automatically where the actual model should be located.
* `model_name` - the name of the model to use for embeddings. Model specified for Face App should be trained separately, should it be used for commercial purposes. Otherwise, any valid specification from [model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) is fine because the model, if not present in `models` subdirectory, will be downloaded automatically
* `is_relative` - whether `model_dir` path is relative to this package's path
* `ctx_id` - the ID of the device context to use for computation. Anything below `0` will result in _CPU_ context
* `det_size` - the window size [`width`, `height`] at which the face should be detected (in pixels)

**Specs**
* `accuracy_name`- the name of the accuracy to use for training. The current available choice is `"total"` which simply computes the average of the correctly predicted faces
* `seed` - the random seed to use for training (to mix up training data)
* `epochs` - the number of training iterations
* `k_folds` - the number of folds to use for training. There will be `k` trainings performed with one fold representing validation data and the other `k-1` folds representing training data. Note that the model parameters remain updated throughout all folds rather than being reset on each fold
* `batch_size` - the number of samples to process at each iteration
* `shuffle` - whether to shuffle tha training dataset,
* `device` - the device on which to load the classifier to perform training. Typical choices are either `cpu` or `cuda:0`

**Params**
* `model` - the dictionary containing model specification parameters, such as `name`, `in_shape`, `num_classes` and `hidden_shape`.
* `optimizer` - the dictionary specifying the optimizer parameters, such as `name`, `lr` etc.
* `criterion` - the dictionary containing the specification of the loss function and its parameters

</details>