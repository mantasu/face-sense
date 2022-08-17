import os
import cv2
import rospy
import torch
import numpy as np

from tqdm import tqdm
from imutils import paths
from datetime import date
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from face_sense.utils import load_dict, save_dict, verify_path, get_app, join

DEFAULT_PHOTO_DIR = "data/identities/photos"
DEFAULT_EMBED_DIR = "data/identities/embeds"
DEFAULT_EMBED_NAME = "newest"

class FaceDataset(Dataset):
    """Face Dataset that can generate and parse face embeddings"""
    
    def __init__(self, embed_dir=DEFAULT_EMBED_DIR,
                       embed_name=DEFAULT_EMBED_NAME,
                       is_relative=True):
        super().__init__()

        # Join dir and name to a full path, verify within pkg
        embed_path = join(embed_dir, embed_name, is_relative)

        if not os.path.exists(embed_path):
            raise FileNotFoundError(f"Please ensure the path to the embeddings"
                                     " file {embed_path} is correct. Otherwise"
                                     ", please generate the face embeddings "
                                     "using `gen_embeds` static method.")

        # Parse the embeddings to inputs and true labels
        x, y = self.parse_embeds(embed_path, is_relative)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    
    @staticmethod
    def gen_embeds(app_config,
                   photo_dir=DEFAULT_PHOTO_DIR,
                   embed_dir=DEFAULT_EMBED_DIR,
                   is_relative=True):
        """Generates embeddings from the face photo files.

        Loops through all the sub-directories of the face photos
        directory and generates the embeddings file in the `.pkl` format
        which contains a dictionary with 2 keys: `labels` and `embeds`.

        Args:
            app_config (dict): The configuration dictionary for the face
                analysis app. It must have the following key-values:
                * model_dir (str): The directory where the embeddings
                    model is present or should be downloaded. Note that
                    within this directory `models` directory should
                    exist or will be created automatically where the
                    actual model should be located.
                * model_name (str): The name of the model to use for
                    embeddings. If it is not present within `models`
                    subdirectory, it will be downloaded automatically.
                * is_relative (bool): Whether the relative path is in
                    package.
                * ctx_id (int): The GPU to use. If GPU is not available
                    or the value is lower than 0, CPU is used.
                * det_size (tuple(int, int)|list[int, int]): The
                    detection size of the face.
            photo_dir (str, optional): Path to a directory with face
                pictures. For each identity, a folder with identity's
                name must be present containing face pictures of them.
                Defaults to DEFAULT_PHOTO_DIR.
            embed_dir (str, optional): A directory to where the
                embeddings file will be saved. The name of the file will
                be today's date and the file extension will be `.pkl`.
                Defaults to DEFAULT_EMBED_DIR.
            
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
        image_paths = list(paths.list_images(photo_dir))
        
        # Get the face analysis app
        app = get_app(app_config)
        
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

        return embed_path
    
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