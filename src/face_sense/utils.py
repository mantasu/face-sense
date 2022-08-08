import os
import json
import pickle
import rospkg

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
