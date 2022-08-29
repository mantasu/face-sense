import rospy
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from threading import Thread, Event

from face_sense.msg import FaceInfo
from face_sense.srv import FRClientGoal, FRClientGoalResponse
from face_sense.recognize import Recognizer
from face_sense.learn.general import Trainer
from face_sense.learn.specific import FaceDataset
from face_sense.utils import load_dict, verify_path

class RecognizeServer:
    def __init__(self, config_path, is_relative=True):

        self.config = load_dict(verify_path(config_path, is_relative))
        self.recognizer = Recognizer(self.config["recognize"]["inference"])

        self._event_stop = Event()
        self.thread = None
        self.frame = None
        self.end = False

        self._init_communication()

    def _init_communication(self):
        # A communication bridge
        self.bridge = CvBridge()
        self.win_name = "Camera Frames"
        self.start_time = rospy.get_time()
        self.interval = self.config["recognize"]["node"]["process_interval"]

        # Topic/service names to communicate
        camera_topic = self.config["camera_topic"]
        info_topic = self.config["recognize"]["node"]["info_topic"]
        service_name = self.config["recognize"]["node"]["service_name"]

        if self.config["is_compressed"]:
            # If compressed image message
            image_type = CompressedImage
            self.read_msg = self.bridge.compressed_imgmsg_to_cv2
        else:
            # If non-compressed
            image_type = Image
            self.read_msg = self.bridge.imgmsg_to_cv2
        
        # Initialize the whole service
        self.fr_server = rospy.Service(
            service_name, FRClientGoal, self.req_callback)

        # Initialize the stream frames subscriber
        self.cam_subscriber = rospy.Subscriber(
            camera_topic, image_type, self.cam_callback, queue_size=1)

        # Initialize publisher for face info
        self.info_publisher = rospy.Publisher(
            info_topic, FaceInfo, queue_size=1)
    
    def cam_callback(self, data):
        try:
            # Retrieve the current stream frame
            self.frame = self.read_msg(data)
        except CvBridgeError as e:
            # Log the error
            rospy.logerr(e)
    
    def publish(self, identities):
        if identities is None:
            return
        
        # Publishable object
        face_info = FaceInfo()
        face_info.boxes = np.array(identities["boxes"]).flatten().tolist()
        face_info.marks = np.array(identities["marks"]).flatten().tolist()
        face_info.names = identities["names"]
        face_info.scores = identities["name_scores"]
        face_info.genders = identities["genders"]
        face_info.ages = identities["ages"]

        # Publish the created face info object
        self.info_publisher.publish(face_info)

    def req_callback(self, goal):
        response = self.cleanup(goal.order_id)

        if response != "":
            return FRClientGoalResponse(response)

        if goal.order_id == 0:
            response = self.generate_identities(goal.order_argument)
        elif goal.order_id == 1:
            response = self.generate_embeddings()
        elif goal.order_id == 2:
            response = self.train_model()
        elif goal.order_id == 3:
            response = self.recognize_once()
        elif goal.order_id == 4:
            response = self.recognize_continuous()
        elif goal.order_id == 5:
            response = self.exit()
        
        return FRClientGoalResponse(response)
    
    def cleanup(self, order_id):
        if order_id == 2:
            # Check if embeddings file is available
            pass
        
        if order_id == 3 or order_id == 4:
            if self.recognizer.embeddings is None:
                self.recognizer.init_data(verbose=True)

                if self.recognizer.embeddings is None:
                    return "Cannot load embeddings file."
            
            if self.recognizer.app is None or self.recognizer.model is None:
                self.recognizer.init_models(verbose=True)

                if self.recognizer.model is None:
                    return "Cannot load recognizer model."

        if order_id != 4 and self.thread is not None and self.thread.is_alive():
            self._event_stop.set()
            self.thread.join()
        
        return ""

    def generate_identities(self, name, num_faces=10):
        return "Functionality is not yet available!"

    def generate_embeddings(self):
        config = self.config["recognize"]["learn"]
        app = config["face_analysis"]
        photo_dir = config["data"]["photo_dir"]
        embed_dir = config["data"]["embed_dir"]
        is_relative = config["data"]["is_relative"]

        path = FaceDataset.gen_embeds(app, photo_dir, embed_dir, is_relative)
        
        return f"Embeddings file generated in {path}"

    def train_model(self, embed_name=None):
        config = self.config["recognize"]["learn"]

        embed_dir = config["data"]["embed_dir"]
        embed_name = config["data"]["embed_name"]
        is_relative = config["data"]["is_relative"]
        dataset = FaceDataset(embed_dir, embed_name, is_relative)

        trainer = Trainer(config, dataset)
        trainer.run()

        return f"Training finished successfully"
    
    def recognize_once(self):
        name = "Unknown"

        if self.frame is not None:
            identities = self.recognizer.process(self.frame)

            if len(identities["names"]) != 0:
                name = identities["names"][0]
                self.publish(identities)

        return f"Recognized {name}"

    def recognize_continuous(self):
        def thread_fn():
            while True:
                if self._event_stop.is_set():
                    self._event_stop.reset()
                    break
                
                if rospy.get_time() - self.start_time < self.interval:
                    continue
                else:
                    self.start_time = rospy.get_time()
                
                if self.frame is not None:
                    identities = self.recognizer.process(self.frame)
                    self.publish(identities)
        
        self.thread = Thread(target=thread_fn)
        self.thread.start()

        return "Started recognizing continuously."

    def exit(self):
        self.end = True

        return "Exiting..."


if __name__ == "__main__":
    freq = 30
    config_path = "config.json"
    is_relative = True

    rospy.init_node("fr_client_goal_server")
    
    # Publisher rate and server
    interval = rospy.Rate(freq)
    recognize_server = RecognizeServer(config_path, is_relative)

    while not rospy.is_shutdown():
        if recognize_server.end:
            break
        
        # While not shutdown
        interval.sleep()