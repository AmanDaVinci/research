import gzip
import numpy as np
from PIL import Image
from typing import Any, List, Tuple, Optional, Callable

from image_classification.utils import download
from image_classification.datasets.base import ImageClassificationDataset


class FashionMNIST(ImageClassificationDataset):

    url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    def download_data(self):
        for file in self.files:
            download(f"{self.url}{file}", self.data_dir/file)
    
    def load_data(self) -> Tuple[List[Image.Image], List[int]]:
        label_file = f"{'train' if self.is_train else 't10k'}-labels-idx1-ubyte.gz"
        image_file = f"{'train' if self.is_train else 't10k'}-images-idx3-ubyte.gz"
        with gzip.open(self.data_dir/label_file, 'rb') as path:
            labels = np.frombuffer(path.read(), dtype=np.uint8, offset=8)\
                       .tolist()
        with gzip.open(self.data_dir/image_file, 'rb') as path:
            images = np.frombuffer(path.read(), dtype=np.uint8, offset=16)\
                       .reshape(len(labels), 28, 28)
        images = [Image.fromarray(image, mode="L") for image in images]
        return images, labels
