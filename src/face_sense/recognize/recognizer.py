import cv2
import rospy
import torch
import numpy as np
import mxnet as mx

from insightface.app import FaceAnalysis
from insightface.app.common import Face
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from face_sense.utils import load_dict, save_dict, verify_path
from face_sense.learn.tools.helper.config import build_model
from face_sense.mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector

#from torch_mtcnn import detect_faces
from PIL import Image

class Recognizer:
    def __init__(self, config):
        camera_topic = config["camera_topic"]
        image_type = CompressedImage if config["is_compressed"] else Image
        
        self.app = FaceAnalysis(model_name="buffalo_s", root=verify_path("data"))
        self.model = build_model(config["recognize"]["model"])
        self.model.load_state_dict(torch.load(verify_path(config["recognize"]["model_path"])))
        self.app.prepare(ctx_id=0, det_size=(160, 160))

        self.detector = MtcnnDetector(model_folder=verify_path("src/face_sense/mxnet_mtcnn_face_detection/model"), ctx=mx.gpu(0), num_worker = 4 , accurate_landmark = False)

        self.win_name = "Camera Frames"

        # A communication bridge
        self.bridge = CvBridge()

        self.msg_callback = self.bridge.compressed_imgmsg_to_cv2 if config["is_compressed"] else self.bridge.imgmsg_to_cv2

        self.cam_subscriber = rospy.Subscriber(
            camera_topic, image_type, self.cam_callback, queue_size=1)
    


    def cam_callback(self, data):
        try:
            # Retrieve the compressed, image and resize it
            frame = self.bridge.compressed_imgmsg_to_cv2(data)
            # frame_scaled = cv2.resize(frame, (self.width, self.height))
            img = self.process_frame(frame)

            # Create a resizable window and load the frame/image
            cv2.imshow(self.win_name, frame)
        except CvBridgeError as e:
            # Log the error
            rospy.logerr(e)

        if cv2.waitKey(1) & 0xFF in [27, 113] or \
           cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
            # If `Q`|`ESC` key or window exit button clicked 
            rospy.signal_shutdown("Quit button clicked")
    
    def process_frame(self, frame):

        # faces = self.app.get(frame)
        bounding_boxes, landmarks = self.detector.detect_face(frame)
        
        #new_img = self.app.draw_on(frame, faces)

        # bboxes = self.detector.detect_faces(frame)

        # for bboxe in bboxes:
        #     bbox = bboxe['box']
        #     bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
        #     landmarks = bboxe['keypoints']
        #     landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
        #                             landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
        #     landmarks = landmarks.reshape((2,5)).T

        #     face = Face(bbox=bbox, kps=landmarks)

        # for face in faces:
        #     y_pred = self.model(face.embedding)
        #     print(y_pred)

        return 0
