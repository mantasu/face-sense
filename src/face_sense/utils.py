import os
import json
import pickle
import rospkg

from insightface.app import FaceAnalysis

def join(dir, filename, is_in_package=True):
    """Joins the directory and a filename.

    Takes the directory path and a filename and joins them into one
    path. If the filename is specified as "newest" or "oldest", then the
    directory contents are listed in a sorted way and, correspondingly,
    the entry with the newest or oldest name is chosen.

    Args:
        dir (str): The path to the directory with the file.
        filename (str): The name of the file within the provided
            directory. Can also be one of "newest", "oldest".
        is_in_package (bool, optional): Whether the relative path is in
            package. Defaults to True.

    Returns:
        str: A single path to the file within the given directory
    """
    
    if filename in ["newest", "oldest"]:
        # Get the top item from a sorted list returned by listing a dir
        filename = sorted(os.listdir(dir), reverse=filename=="newest")[0]
    
    # Join the directory and the file
    path = os.path.join(dir, filename)

    return verify_path(path, is_in_package)

def verify_path(path, is_in_package=True):
    """Verifies the path

    If the path is relative to the Face Sense package, it is appended to
    the absolute directory of the Face Sense package. It also checks
    whether the path itself exists and if it does not, it creates the
    corresponding directory(-ies) contained within the path.

    Args:
        path (str): The desired directory or path to file
        is_in_package (bool, optional): Whether the relative path is in
            package. Defaults to True.

    Returns:
        str: An existing verified path
    """
    if is_in_package:
        # Change the relative path to the package directory
        rel_path = rospkg.RosPack().get_path("face_sense")
        path = os.path.join(rel_path, path)
    
    # Get the non-existing full path to the directory and create it
    dirname = path if os.path.isdir(path) else os.path.dirname(path)

    if not os.path.exists(dirname):
        # Create if needed
        os.makedirs(dirname)
    
    return path

def save_dict(py_dict, path):
    """Saves python dict as json/pickle.

    Args:
        py_dict (dict): The python dictionary
        path (str): The path to json/pickle file
    """
    if path.endswith(".json"):
        with open(path, 'w') as fp:
            # Just save the json python dict
            json.dump(py_dict, fp, indent=4)
    elif path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "wb") as fp:
            # Just save the pickle py dict
            fp.write(pickle.dumps(py_dict))

def load_dict(path):
    """Loads json/pickle file as python dict.
    
    Returns:
        (dict): A python dictionary
    """
    if path.endswith(".json"):
        with open(path, 'r') as json_file:
            # Extract dict from JSON file
            py_dict = json.load(json_file)
    elif path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, 'rb') as f:
            # Load the pickle data
            py_dict = pickle.load(f)

    return py_dict

def get_face_analysis(config):
    """Prepares the face analysis object from insightface.

    Creates a face analysis object from insight face by loading the
    model and prepares it for the tasks by specifying the context and
    detection size.

    Args:
        config (dict): The configuration dictionary to prepare the face
            analysis app with the following parameters:
            * model_name (str): The name of the model to load. Note that
                the models provided by insightface cannot be used for
                commercial purposes.
            * model_dir (str): The directory which has the subdirectory
                with the model's name containing model's files.
            * is_relative (bool): Whether the relative path is in
                package. Defaults to True.
            * ctx_id (int): The GPU to use. If GPU is not available or
                the value is lower than 0, CPU is used.
            * det_size (tuple(int, int)|list[int, int]): The detection
                size of the face.
    
    Returns:
        insightface.app.FaceAnalysis: The prepared face analysis object
    """
    # Verify the model directory is a valid path in or out package
    root = verify_path(config["model_dir"], config["is_relative"])

    # Get the face analysis object and prepare for embeddings
    app = FaceAnalysis(name=config["model_name"], root=root)
    app.prepare(ctx_id=config["ctx_id"], det_size=tuple(config["det_size"]))

    return app