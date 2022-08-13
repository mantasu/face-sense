import cv2
import rospy
import torch
import numpy as np

from insightface.app import FaceAnalysis
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from face_sense.utils import load_dict, save_dict, verify_path
from face_sense.learn.tools.helper.config import build_model

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

class Recognizer:
    def __init__(self, config):
        camera_topic = config["camera_topic"]
        image_type = CompressedImage if config["is_compressed"] else Image
        
        self.app = FaceAnalysis(model_name="buffalo_s", root=verify_path("data"))
        self.model = build_model(config["recognize"]["model"])
        self.model.load_state_dict(torch.load(verify_path(config["recognize"]["model_path"])))
        self.model.eval()
        self.app.prepare(ctx_id=0, det_size=(160, 160))

        self.win_name = "Camera Frames"

        # A communication bridge
        self.bridge = CvBridge()

        self.msg_callback = self.bridge.compressed_imgmsg_to_cv2 if config["is_compressed"] else self.bridge.imgmsg_to_cv2

        self.cam_subscriber = rospy.Subscriber(
            camera_topic, image_type, self.cam_callback, queue_size=1)

        self.timer = rospy.get_time()

        # ENCODER
        data = load_dict(verify_path("data/faces/embeds/2022-08-07.pkl"))

        self.label_encoder = LabelEncoder()
        self.embeddings = np.array(data["embeds"])
        self.labels = self.label_encoder.fit_transform(data["labels"])
    


    def cam_callback(self, data):
        if rospy.get_time() - self.timer < 3:
            print("No", rospy.get_time() - self.timer, end='\r')
            return
        else:
            print()
            self.timer = rospy.get_time()

        try:
            # Retrieve the compressed, image and resize it
            frame = self.bridge.compressed_imgmsg_to_cv2(data)
            # frame_scaled = cv2.resize(frame, (self.width, self.height))
            img = self.process_frame(frame)


            frame = img if img is not None else frame

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
        faces = self.app.get(frame)

        labels = self.labels
        embeddings = self.embeddings
        comparing_num = 5
        cosine_threshold = 0.8
        proba_threshold = 0.85

        with torch.no_grad():
            for face in faces:
                y_pred = self.model(torch.from_numpy(face.embedding))
                y_pred = y_pred.flatten()
                j = np.argmax(y_pred).item()
                proba = y_pred[j]

                match_class_idx = (labels == j)
                match_class_idx = np.where(match_class_idx)[0]

                print(labels)
                print(j)
                print(match_class_idx, comparing_num)
                selected_idx = np.random.choice(match_class_idx, comparing_num)
                compare_embeddings = embeddings[selected_idx]
                # Calculate cosine similarity
                print(compare_embeddings.shape, face.embedding.shape)
                cos_similarity = cosine_similarity(compare_embeddings, [face.embedding]).mean()

                if cos_similarity < cosine_threshold and proba > proba_threshold:
                    name = self.label_encoder.classes_[j]
                    text = "{}".format(name)
                    print("Recognized: {} <{:.2f}>".format(name, proba*100))
            
            print(text)

        return None
