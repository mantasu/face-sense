from face_sense.learn.specific import FaceDataset
from face_sense.learn.general import Trainer
from face_sense.utils import verify_path, load_dict

def train_recognizer(config):
    embed_path = config["data"]["embed_path"]
    is_relative = config["data"]["is_relative"]
    dataset = FaceDataset(embed_path, is_relative)

    trainer = Trainer(config["specs"], dataset)
    trainer.run()

if __name__ == "__main__":
    config = load_dict(verify_path("config.json"))
    train_recognizer(config["recognizer"]["learn"])