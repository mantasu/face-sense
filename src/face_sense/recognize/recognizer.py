import rospy
import torch
import random
import warnings
import numpy as np

from face_sense.msg import FaceInfo

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from face_sense.utils import load_dict, join_by_kwd, get_app
from face_sense.learn.tools import build_model

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class Recognizer:
    def __init__(self, config):        
        self._init_tunable(config["recognize"]["inference"])
        self._init_models(config["recognize"]["inference"])
        self._init_data(config["recognize"]["inference"])
        self._init_communication(config)

        self.timer = rospy.get_time()
        self.identities = None
        
        self.info_publisher = rospy.Publisher(
            "/recognizer/face_info", FaceInfo, queue_size=1
        )
    
    def _init_communication(self, config):
        # A communication bridge
        self.bridge = CvBridge()
        self.win_name = "Camera Frames" 
        camera_topic = config["camera_topic"]

        if config["is_compressed"]:
            # If compressed image message
            image_type = CompressedImage
            self.read_msg = self.bridge.compressed_imgmsg_to_cv2
        else:
            # If non-compressed
            image_type = Image
            self.read_msg = self.bridge.imgmsg_to_cv2
        
        self.cam_subscriber = rospy.Subscriber(
            camera_topic, image_type, self.cam_callback, queue_size=1)
    
    def _init_tunable(self, config):
        for key, val in config["tunable"].items():
            # Set key-val attribute
            setattr(self, key, val)
    
    def _init_models(self, config):
        # Set the device if it is provided as one of model parameters
        self.device = torch.device(config["model"].pop("device", "cpu"))

        # Get the face analysis app and ID classifier
        self.app = get_app(config["face_analysis"])
        self.model = build_model(config["model"])

        # Get the path to saved model params and load model
        state_dict_path = join_by_kwd(config["data"], "model")
        self.model.load_state_dict(torch.load(state_dict_path))
        
        # Set correct device, eval
        self.model.to(self.device)
        self.model.eval()
        
    def _init_data(self, config):
        # Get the path to the embeddings file and load embeds
        data = load_dict(join_by_kwd(config["data"], "embed"))
        self.embeddings = torch.tensor(data["embeds"]).to(self.device)

        # Init and fit the label encoder
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(data["labels"])
        self.labels = torch.tensor(self.labels).to(self.device)

    def compute_similarity(self, compare_embeddings, embedding):
        """Computes average similarity score between embeddings.

        Takes embeddings of shape (N, D) to compare against a single
        embedding of shape (D,). It obtains N scores and computes the
        mean.

        Note: the embeddings come in form of tensors thus they may be on
            device other than cpu. Since calculations in this method do
            not involve GPU, the tensors must be converted to CPU.

        Args:
            compare_embeddings (torch.Tensor): The embeddings tensor to
                compare one embedding against.
            embedding (torch.Tensor): The single face embedding to check
                how similar it is against every other embedding in
                `compare_embeddings`.

        Returns:
            float: A similarity score
        """
        # Prepare the inputs as numpy arrays
        X = compare_embeddings.cpu().numpy()
        Y = [embedding.cpu().numpy()]

        return cosine_similarity(X, Y).mean()
    
    def compute_score(self, similarity, probability):
        return ((similarity + 1) / 2 + probability) / 2
    
    def cam_callback(self, data):
        if rospy.get_time() - self.timer < 1:
            print("No", rospy.get_time() - self.timer, end='\r')
            return
        else:
            print()
            self.timer = rospy.get_time()

        try:
            # Retrieve the image
            frame = self.read_msg(data)
        except CvBridgeError as e:
            # Log the error
            rospy.logerr(e)

        self.identities = self.process_frame(frame)
    
    def process_frame(self, frame):
        # Get faces and disable grad
        torch.set_grad_enabled(False)
        faces = self.app.get(frame)

        # Create a list of keys that identities will have, init lists
        keys = ["boxes", "marks", "names", "name_scores", "genders", "ages"]
        identities = {k: [] for k in keys}

        for face in faces:
            # Get face embedding and convert it to a tensor on device
            face_embed = torch.tensor(face.embedding).to(self.device)

            # Get output and predicted prob
            output = self.model(face_embed)
            prob, i = torch.max(torch.softmax(output, 0), 0)
            
            # Get all the indices of the class that matches prediction
            matched_class_idx = torch.nonzero(self.labels == i).squeeze()
            num_matched = len(matched_class_idx)

            # Select only a certain number of indices and their embeds
            selected_idx = random.sample(range(num_matched), self.num_to_compare)
            compare_embeddings = self.embeddings[matched_class_idx[selected_idx]]

            # Calculate the average similarity score between the embeds
            sim = self.compute_similarity(compare_embeddings, face_embed)
            score = self.compute_score(sim, prob.item())

            if sim > self.sim_threshold and prob.item() > self.prob_threshold:
                # Append known name to the identities collection
                identities["names"].append(self.label_encoder.classes_[i.item()])
                identities["name_scores"].append(score)
            else:
                # Append the unknown identity as well
                identities["names"].append("Unknown")
                identities["name_scores"].append(1 - score)

            # Append properties to the identities collection
            identities["boxes"].append(face.bbox.astype(np.int))
            identities["marks"].append(face.kps.astype(np.int))
            identities["genders"].append(face.gender)
            identities["ages"].append(face.age)

        return identities
    
    def publish(self):
        if self.identities is None:
            return
        
        face_info = FaceInfo()
        face_info.boxes = np.array(self.identities["boxes"]).flatten().tolist()
        face_info.marks = np.array(self.identities["marks"]).flatten().tolist()
        face_info.names = self.identities["names"]
        face_info.scores = self.identities["name_scores"]
        face_info.genders = self.identities["genders"]
        face_info.ages = self.identities["ages"]

        self.info_publisher.publish(face_info)
