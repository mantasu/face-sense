import os
import cv2
import rospy
import torch
import numpy as np

from tqdm import tqdm
from imutils import paths
from datetime import date
from torch.utils.data import Dataset
from insightface.app import FaceAnalysis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from face_sense.utils import load_dict, save_dict, verify_path

DEFAULT_PHOTO_DIR = "data/faces/photos"
DEFAULT_EMBED_DIR = "data/faces/embed"
DEFAULT_MODEL_DIR = "data"
DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_EMBED_PATH = "data/faces/embed/2022-08-07.pkl"

class FaceDataset(Dataset):
    def __init__(self, embed_path=DEFAULT_EMBED_PATH, is_relative=True):
        super().__init__()

        # Just verify that embed path is within package
        embed_path = verify_path(embed_path, is_relative)

        if not os.path.exists(embed_path):
            raise FileNotFoundError(f"Please ensure the path to the embeddings"
                                     " file {embed_path} is correct. Otherwise"
                                     ", please generate the face embeddings "
                                     "using `gen_embeds` static method.")

        x, y = self.parse_embeds(embed_path, is_relative)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    
    @staticmethod
    def gen_embeds(photo_dir=DEFAULT_PHOTO_DIR,
                   embed_dir=DEFAULT_EMBED_DIR,
                   model_dir=DEFAULT_MODEL_DIR,
                   model_name=DEFAULT_MODEL_NAME,
                   is_relative=True):
        """Generates embeddings from the face photo files.

        Loops through all the sub-directories of the face photos
        directory and generates the embeddings file in the `.pkl` format
        which contains a dictionary with 2 keys: `labels` and `embeds`.

        Args:
            photo_dir (str, optional): Path to a directory with face
                pictures. For each identity, a folder with identity's
                name must be present containing face pictures of them.
                Defaults to DEFAULT_PHOTO_DIR.
            embed_dir (str, optional): A directory to where the
                embeddings file will be saved. The name of the file will
                be today's date and the file extension will be `.pkl`.
                Defaults to DEFAULT_EMBED_DIR.
            model_dir (str, optional): The directory where the
                embeddings model is present or should be downloaded.
                Note that within this directory `models` directory
                should exist or will be created automatically where the
                actual model should be located. Defaults to
                DEFAULT_MODEL_DIR.
            model_name (str, optional): The name of the model to use for
                embeddings. If it is not present within `models`
                subdirectory, it will be downloaded automatically.
                Defaults to DEFAULT_MODEL_NAME.
            is_relative (bool, optional): Whether all the provided
                directories are relative to this package directory or
                take the form of absolute paths. Defaults to True.
        """
        # Init lists
        embeds = []
        labels = []

        # Verify and generate paths inside/outside pkg
        photo_dir = verify_path(photo_dir, is_relative)
        embed_dir = verify_path(embed_dir, is_relative)
        model_dir = verify_path(model_dir, is_relative)
        image_paths = list(paths.list_images(photo_dir))

        # Get the face analysis object and prepare for embeddings
        app = FaceAnalysis(name=model_name, root=model_dir)
        app.prepare(ctx_id=0, det_size=(160, 160))
        
        for image_path in tqdm(image_paths):
            # Read the image & make it RGB
            image = cv2.imread(image_path)
            image = image[:, :, ::-1]

            # Get every found face
            faces = app.get(image)

            if len(faces) != 0:
                # Get the name of the face and add to list
                label = image_path.split(os.path.sep)[-2]
                labels.append(label)

                # Get face embedding and append
                embedding = faces[0].embedding
                embeds.append(embedding)
        
        # Create the data dict of labels and embeds
        data = {"embeds": embeds, "labels": labels}
        rospy.loginfo(f"Detected {len(embeds)} faces")

        # Create a full path to the embeddings file and save the data
        embed_path = os.path.join(embed_dir, f"{date.today()}.pkl")
        save_dict(data, embed_path)
    
    @staticmethod
    def parse_embeds(embed_path, is_relative=True):
        """A static method for loading face embeddings

        Loads face embeddings and returns them in the form of training
        data. It loads the vector embeddings for every identity and
        encodes every identity to a one-hot vector.
        
        Note: the face embeddings file must be `.pkl` file storing a
            dictionary containing `labels` and `embeds` keys.

        Args:
            embed_path (str): The path to face embeddings file.
            is_relative (bool, optional): Whether `embeds_path` is
                relative to current package's path or is an absolute
                path. Defaults to True.
        
        Returns:
            x (np.array): Input face embedding data of shape (N, D)
            y (np.array): One hot vectors of shape (N, N)
        """
        # Load the embeddings dictionary file
        data = load_dict(verify_path(embed_path, is_relative))

        # Create numeric label values
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(data["labels"])

        # Create one hot label encodings
        one_hot_encoder = OneHotEncoder()
        y = one_hot_encoder.fit_transform(labels.reshape(-1, 1)).toarray()

        # Get input data (embeddings)
        x = np.array(data["embeds"])

        return x, y
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.y)