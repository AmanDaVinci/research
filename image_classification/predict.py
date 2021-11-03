import torch
from PIL import Image
from pathlib import Path
from schema import Schema, And, Or
from base_predictor import BasePredictor, PredictorSchemaValidationException

from image_classification.models.cnn import CNN
from image_classification.preprocessors.basic import preprocessor


class ModelPredictor(BasePredictor):

    name = "fashion-image-classifier"
    major_version = 0
    minor_version = 1
    model_file = Path(__file__).parent/"model_archive"/"model-250240-0.292.pt"
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    input_schema = Schema({
        "id": int,
        "image": str,
        "height": And(int, lambda x: x==28),
        "width": And(int, lambda x: x==28),
        "pillow_mode": And(str, lambda s: s=="L"),
        "encoding": And(str, lambda s: s=="latin-1")
    })
    output_schema = Schema({
        "id": int,
        "label": Or(*classes)
    })

    def __init__(self):
        # TODO: load from experiment config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_ = CNN(in_channels=1, out_dim=10, device=device)
        checkpoint = torch.load(self.model_file, map_location=device)
        self._model_.load_state_dict(checkpoint["model_state_dict"])
        self._model_.to(device)
        self._model_.eval()
    
    def preprocess(self, data: dict) -> dict:
        try:
            byte_image = data["image"].encode("latin-1")
            image = Image.frombytes("L", (28,28), byte_image, 'raw')
        except Exception as e:
            raise PredictorSchemaValidationException(
                f"Invalid input data: {str(e)}"
            )
        image = preprocessor(image)
        # expand the batch dimension
        data["image"] = torch.unsqueeze("image",0)
        return data

    def predict(self, data: dict) -> dict:
        data["image"] = data["image"].to(self._model_.device)
        y_pred = self._model_(data["image"])
        class_idx = y_pred.argmax(axis=1)
        data["label"] = self.classes[class_idx]
        return data
    
    def postprocess(self, data: dict) -> dict:
        data.pop("image")
        return data