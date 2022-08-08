from face_sense.learn.train_recognizer import train_recognizer
from face_sense.utils import verify_path, load_dict

if __name__ == "__main__":
    config = load_dict(verify_path("config.json"))
    train_recognizer(config["recognize"]["train"])