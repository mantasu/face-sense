import cv2
import rospy
import numpy as np

from sensor_msgs.msg import CompressedImage, Image
from face_sense.msg import FaceInfo
from cv_bridge import CvBridge, CvBridgeError

class CameraViewer:
    def __init__(self, config, win_name="Camera Frames"):
        # Dims & win name
        camera_topic = config["camera_topic"]
        # face_info_topic = config["face_info_topic"]
        face_info_topic = "/recognizer/face_info"
        image_type = CompressedImage if config["is_compressed"] else Image
        self.win_name = win_name

        self.identities = {}
        self.drawable = ["box", "marks", "name", "bio"]

        # A communication bridge
        self.bridge = CvBridge()

        self.cam_subscriber = rospy.Subscriber(
            camera_topic, image_type, self.cam_callback, queue_size=1)
        self.info_subscriber = rospy.Subscriber(
            face_info_topic, FaceInfo, self.info_callback, queue_size=1)
    
    def cam_callback(self, data):
        try:
            # Retrieve the compressed, image and resize it
            frame = self.bridge.compressed_imgmsg_to_cv2(data)
            img = self.draw_on_frame(frame)

            # Create a resizable window and load the frame/image
            cv2.imshow(self.win_name, img)
        except CvBridgeError as e:
            # Log the error
            rospy.logerr(e)

        if cv2.waitKey(1) & 0xFF in [27, 113] or \
           cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
            # If `Q`|`ESC` key or window exit button clicked 
            rospy.signal_shutdown("Quit button clicked")
    
    def info_callback(self, data):
        self.identities["boxes"] = np.array(data.boxes).reshape(-1, 4)
        self.identities["marks"] = np.array(data.marks).reshape(-1, 5, 2)
        self.identities["names"] = data.names
        self.identities["scores"] = data.scores
        self.identities["genders"] = data.genders
        self.identities["ages"] = data.ages
    
    def draw_on_frame(self, frame):
        if len(self.identities) == 0:
            return frame
        
        img = frame.copy()
        
        for i in range(len(self.identities["boxes"])):
            bbox = self.identities["boxes"][i]
            marks = self.identities["marks"][i]
            name = self.identities["names"][i]
            score = self.identities["scores"][i]
            gender = self.identities["genders"][i]
            age = self.identities["ages"][i]

            x1 = x2 = bbox[0]
            y1 = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
            y2 = bbox[3] + 20 if bbox[3] + 20 > 20 else bbox[3] - 20

            self.draw_box(img, bbox)
            self.draw_marks(img, marks)
            self.draw_name(img, name, score, (x1, y1))
            self.draw_bio(img, gender, age, (x2, y2))
        
        return img
    
    def draw_box(self, img, box):
        if "box" not in self.drawable:
            return
        
        color = (0, 0, 255)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    
    def draw_marks(self, img, marks):
        if "marks" not in self.drawable:
            return
        
        for i, mark in enumerate(marks):
            if i == 0 or i == 3:
                color = (0, 255, 0)
            elif i == 1 or i == 4:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.circle(img, (mark[0], mark[1]), 1, color, 2)
    
    def draw_name(self, img, name, prob, coords):
        if "name" not in self.drawable:
            return

        text = f"{name} ({prob*100:.2f}%)"
        cv2.putText(img, text, coords, cv2.FONT_HERSHEY_COMPLEX, 0.45, (75, 255, 0), 1)
    
    def draw_bio(self, img, gender, age, coords):
        if "bio" not in self.drawable:
            return
        
        text = f"{'Male' if gender else 'Female'}, {age}"
        cv2.putText(img, text, coords, cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 40), 1)