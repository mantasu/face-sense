import os
import json
import pickle
import rospkg

from pathlib import Path
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
        kwargs = {"key": os.path.getmtime, "reverse": filename=="newest"}
        filename = sorted(Path(dir).iterdir(), **kwargs)[0]
    
    # Join the directory and the file
    path = os.path.join(dir, filename)

    return verify_path(path, is_in_package)

def join_by_kwd(path_dict, keyword):
    """Joins the directory and the filename given their keyword.

    Given a dictionary that contains path to the desired directory and
    the desired filename, this method joins them into one path.

    Note: `path_dict` should also contain an entry named "is_relative"
        indicating whether the overall path is relative to the package
        directory or is an absolute path.

    Args:
        path_dict (dict): The dictionary with keyword paths. For
            instance, if keyword is "model", then `path_dict` should
            contain "model_dir" and "model_name".
        keyword (str): The keyword which the path is based on.

    Returns:
        str: A single path to the file within the specified directory
    """
    # Get dir, filename, is_rel params
    dir = path_dict[f"{keyword}_dir"]
    filename = path_dict[f"{keyword}_name"]
    is_in_pkg = path_dict.get("is_relative", True)

    return join(dir, filename, is_in_pkg)

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

def get_app(config):
    """Prepares the face analysis object from insightface.

    Creates a face analysis object from insight face by loading the
    model and prepares it for the tasks by specifying the context and
    detection size.

    Args:
        config (dict): The configuration dictionary to prepare the face
            analysis app with the following parameters:
            * model_dir (str): The directory where the embeddings model
                is present or should be downloaded. Note that within
                this directory `models` directory should exist or will
                be created automatically where the actual model should
                be located.
            * model_name (str): The name of the model to load. If it is
                not present within `models` subdirectory, it will be
                downloaded automatically. Note that the models provided
                by insightface cannot be used for commercial purposes.
            * is_relative (bool): Whether the relative path is in
                package.
            * ctx_id (int): The GPU to use. If GPU is not available or
                the value is lower than 0, CPU is used.
            * det_size (tuple(int, int)|list[int, int]): The detection
                size of the face.
    
    Returns:
        insightface.app.FaceAnalysis: The prepared face analysis object
    """
    # Verify the model directory is a valid path in or out package
    root = verify_path(config["model_dir"], config["is_relative"])

    # Set onnx runtime interface session providers in order
    prov = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # Get the face analysis object and prepare for embeddings
    app = FaceAnalysis(name=config["model_name"], root=root, providers=prov)
    app.prepare(ctx_id=config["ctx_id"], det_size=tuple(config["det_size"]))

    return app