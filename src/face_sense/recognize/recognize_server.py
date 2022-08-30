import rospy
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image

from face_sense.msg import FaceInfo
from face_sense.utils import load_dict, verify_path
from face_sense.srv import FRClientGoal, FRClientGoalResponse
from face_sense.recognize.components import RecognitionGoalHandler
from face_sense.recognize.components import RecognitionViewHandler

class RecognizeServer:
    def __init__(self, config_path, is_relative=True):
        # Read the `config.json` file from the provided path
        self.config = load_dict(verify_path(config_path, is_relative))
        
        # Init goal & view handlers based on config
        self.goal_handler = RecognitionGoalHandler(
            self.config["recognize"])
        self.view_handler = RecognitionViewHandler(
            self.config["recognize"]["node"]["drawable"])
        
        # Init communication between topics
        self._init_communication_attributes()
        self._init_communication()
    
    def _init_communication_attributes(self):
        # A communication bridge
        self.bridge = CvBridge()

        # Topic/service names to communicate
        self.cam_topic = self.config["camera_topic"]
        self.info_topic = self.config["recognize"]["node"]["info_topic"]
        self.view_topic = self.config["recognize"]["node"]["view_topic"]
        self.service_name = self.config["recognize"]["node"]["service_name"]

        if self.config["is_compressed"]:
            # If compressed image message
            self.image_type_sub = CompressedImage
            self.msg2img = self.bridge.compressed_imgmsg_to_cv2
        else:
            # If non-compressed
            self.image_type_sub = Image
            self.msg2img = self.bridge.imgmsg_to_cv2

        if self.config["node"]["is_compressed"]:
            # If compressed image message
            self.image_type_pub = CompressedImage
            self.img2msg = lambda x: self.bridge.cv2_to_compressed_imgmsg(
                x, dst_format=self.config["node"]["format"])
        else:
            # If non-compressed image
            self.image_type_pub = Image
            self.img2msg = self.bridge.cv2_to_imgmsg
    
    def _init_communication(self):
        # Initialize the whole service
        self.fr_server = rospy.Service(
            self.service_name, FRClientGoal, self.goal_callback)

        # Initialize the stream frames subscriber
        self.cam_subscriber = rospy.Subscriber(
            self.cam_topic, self.image_type_sub, self.cam_callback, queue_size=1)

        # Initialize publisher for face info
        self.info_publisher = rospy.Publisher(
            self.info_topic, FaceInfo, queue_size=1)
        
        # Initialize publisher for view
        self.view_publisher = rospy.Publisher(
            self.view_topic, self.image_type_pub, queue_size=1)
    
    def cam_callback(self, img_msg):
        try:
            # Retrieve image from the message
            self.frame = self.msg2img(img_msg)
        except CvBridgeError as e:
            # Log the error
            rospy.logerr(e)

    def goal_callback(self, goal):
        # Extract order ID & argument and call goal handler to handle
        order_id, order_argument = goal.order_id, goal.order_argument
        response = self.goal_handler.handle_order(order_id, order_argument)

        return FRClientGoalResponse(response)
    
    def publish_info(self):
        # Get the identities and initialize face info msg
        identities = self.goal_handler.get_identities()
        face_info = FaceInfo()

        if identities is not None:
            # Assign bounding boxes, marks, names, scores and bio info
            face_info.boxes = np.array(identities["boxes"]).flatten().tolist()
            face_info.marks = np.array(identities["marks"]).flatten().tolist()
            face_info.names = identities["names"]
            face_info.scores = identities["name_scores"]
            face_info.genders = identities["genders"]
            face_info.ages = identities["ages"]

        # Publish the created face info object
        self.info_publisher.publish(face_info)
    
    def publish_view(self):
        # Get the identities and draw on current frame
        identities = self.goal_handler.get_identities()
        img = self.view_handler.draw_on_frame(self.frame, identities)

        try:
            # Compute img msg, publish
            img_msg = self.img2msg(img)
            self.view_publisher.publish(img_msg)
        except CvBridgeError as e:
            # If any error
            rospy.logerr(e)
    
    def publish(self):
        # Publish info+view
        self.publish_info()
        self.publish_view()