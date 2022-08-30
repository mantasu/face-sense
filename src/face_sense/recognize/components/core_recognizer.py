import rospy
import torch
import random
import warnings
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from face_sense.learn.tools import build_model
from face_sense.utils import load_dict, join_by_kwd, get_app

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class CoreRecognizer:
    """Main recognizer which can process the image with faces

    This class is the main one to process the image to detect and
    recognize the identities. Its main method ~CoreRecognizer.process
    helps to do that. It is able to extract bounding boxes, landmarks,
    predict gender and age as well as the name for the detected people
    (faces).
    """
    def __init__(self, config):
        """Initializes the recognizer

        Assigns config attributes and initializes face analysis and
        classifier models along with embeddings representation. If
        initialization is unsuccessful, they can be reinitialized. This
        could happen, for example, if the model is not yet trained, thus
        the recognizer should reinitialize models.

        Args:
            config (dict): The configuration dictionary for "inference"
                parameters.
        """
        # Just assign config
        self.config = config

        # Init attributes
        self.app = None
        self.model = None
        self.embeddings = None
        self.identities = None

        # Init through reusable methods
        self.init_tunable()
        self.init_models(verbose=False)
        self.init_data(verbose=False)
        
    
    def init_tunable(self):
        """Initializes the tunable attributes for inference

        Simply goes through the key value pairs in `self.config`
        "tunable" sub-dictionary and assigns them as this class'
        attributes. There are `3` tunable values:
            * `sim_threshold` - threshold for similarity value. It is a
                minimum value the similarity function should yield when
                comparing the identified face with its counterparts in
                the face database.
            * `prob_threshold` - threshold for probability value. It is
                a minimum value the model should achieve when
                classifying which identity the captured face belongs to.
            * `num_to_compare` - the number of counterpart faces in the
                database the detected face to compare with to determine
                the mean similarity value.
        """
        for key, val in self.config["tunable"].items():
            # Set key-val attribute
            setattr(self, key, val)
    
    def init_models(self, verbose=False):
        """Initializes the face analysis app and the classifier

        This method tries to construct face analysis app and the
        classifier with the desired parameters specified in
        `self.config`. If anything fails, for instance, the model path
        is not found, the object is not initialized and, optionally,
        the error is printed that the initialization failed.

        Note: face analysis app should be the same which generated the
            embeddings file. This is to keep the embedding
            representations the same during classification when the
            extracted new embedding is compared against the ones in
            the database (embeddings file).

        Args:
            verbose (bool, optional): Whether to log the error in case
                the initialization fails. Defaults to False.
        """
        # Set the device if it is provided as one of model parameters
        self.device = torch.device(self.config["model"].pop("device", "cpu"))

        if self.app is None:
            try:
                # Try to initialize the face analysis/detection app
                self.app = get_app(self.config["face_analysis"])
            except FileNotFoundError:
                if verbose:
                    rospy.logerr("Could not initialize Face Analysis app")
                
                # Set app to None
                self.app = None
        
        if self.model is None:
            try:
                # Get the face analysis app and ID classifier
                self.app = get_app(self.config["face_analysis"])
                self.model = build_model(self.config["model"])

                # Get the path to saved model params and load model
                state_dict_path = join_by_kwd(self.config["data"], "model")
                self.model.load_state_dict(torch.load(state_dict_path))
                
                # Set correct device, eval
                self.model.to(self.device)
                self.model.eval()
            except RuntimeError as e:
                if verbose:
                    rospy.logerr(f"Cannot load the desired model: {e}")
                
                # Set model to None
                self.model = None
        
    def init_data(self, verbose=False):
        """Initializes the face embedding data

        It looks for the embeddings file specified in `self.config` and
        creates a tensor representation of it. It also creates a label
        encoder to represent labels numerically in the embeddings file.
        If the file is not fount, `self.embeddings` is set to None and,
        optionally, the log error is printed.

        Args:
            verbose (bool, optional): Whether to log the error in case
                the initialization fails. Defaults to False.
        """
        if self.embeddings is None:
            try:
                # Get the path to the embeddings file and load embeds
                data = load_dict(join_by_kwd(self.config["data"], "embed"))
                self.embeddings = torch.tensor(data["embeds"]).to(self.device)

                # Init and fit the label encoder
                self.label_encoder = LabelEncoder()
                self.labels = self.label_encoder.fit_transform(data["labels"])
                self.labels = torch.tensor(self.labels).to(self.device)
            except FileNotFoundError:
                if verbose:
                    rospy.logerr("Unable to find the embeddings file")
                
                self.embeddings = None

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
            float: A similarity score.
        """
        # Prepare the inputs as numpy arrays
        X = compare_embeddings.cpu().numpy()
        Y = [embedding.cpu().numpy()]

        return cosine_similarity(X, Y).mean()
    
    def compute_score(self, similarity, probability):
        """Computes a single confidence score

        It is a utility function to represent the similarity score and
        the probability score as a single value. It simply averages the
        normalized values between 0 and 1.

        Args:
            similarity (float): The similarity score between -1 and 1.
                How similar the captured face is to its counterparts in
                the database. 
            probability (float): The probability score between 0 and 1.
                How probable the face belongs to its identity.

        Returns:
            float: A single confidence score between 0 and 11.
        """
        return ((similarity + 1) / 2 + probability) / 2
    
    def process(self, frame):
        """Processes the image to recognize faces

        Takes an image, calls the face analysis app to extract the face
        embeddings from it (along with values for face boundary boxes,
        landmarks, presumable ages and genders) and calls the trained
        classifier to identify which identity the face belongs to. It
        then computes the similarity score between the corresponding
        identity faces in the embeddings file and if both the similarity
        and the probability scores are higher than the specified
        threshold, the name of the identity, instead of "Unknown", is
        appended to the list of recognized people.

        Args:
            frame (numpy.ndarray): The image represented as a numpy
                array from which to extract the face embeddings.

        Returns:
            dict: A dictionary with keys indicating features, such as
                "names", "genders", and values indicating identities,
                i.e, lists where each entry is a feature that the key
                specifies
        """
        # Get faces and disable grad
        torch.set_grad_enabled(False)
        faces = self.app.get(frame)

        # Create a list of keys that identities will have, init lists
        keys = ["boxes", "marks", "names", "scores", "genders", "ages"]
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
                identities["scores"].append(score)
            else:
                # Append the unknown identity as well
                identities["names"].append("Unknown")
                identities["scores"].append(1 - score)

            # Append properties to the identities collection
            identities["boxes"].append(face.bbox.astype(np.int))
            identities["marks"].append(face.kps.astype(np.int))
            identities["genders"].append(face.gender)
            identities["ages"].append(face.age)

        return identities
