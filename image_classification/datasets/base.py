import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from abc import abstractmethod
from typing import Any, List, Tuple, Dict, Optional, Callable


class ImageClassificationDataset(Dataset):
    """ Abstract base class for Image Classification Datasets """

    @property
    @abstractmethod
    def url(self):
        """ The abstract property returns the URL str to download the data from. """
        raise NotImplementedError()

    @property
    @abstractmethod
    def files(self):
        """ The abstract property returns the list of data files. """
        raise NotImplementedError()

    @property
    @abstractmethod
    def classes(self):
        """ The abstract property returns the list of classes. """
        raise NotImplementedError()

    # @property
    # @abstractmethod
    # def data_dir(self):
    #     """ The abstract property points to the data directory. """
    #     raise NotImplementedError()

    @abstractmethod
    def download_data(self):
        """ Download the data files in the data_dir. """
        raise NotImplementedError

    @abstractmethod
    def load_data(self) -> Tuple[List[Image.Image], List[int]]:
        """ Load the data as a list of Pillow Images and Python Integers. """
        raise NotImplementedError

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {class_label: i for i, class_label in enumerate(self.classes)}

    def __init__(
        self, 
        is_train: bool, 
        transform: Callable,
        data_dir: Path,
        download_data: bool = True,
    ) -> None:
        self.is_train = is_train
        self.transform = transform
        self.data_dir = data_dir
        if download_data:
            self.download_data()
        if not self._check_exists_():
            raise RuntimeError("Data files not found. Please download them.")
        self.images, self.labels = self.load_data()
        assert isinstance(self.images, list) and isinstance(self.labels, list)
        assert isinstance(self.images[0], Image.Image)
        assert isinstance(self.labels[0], int)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
        image, label = self.transform(self.images[index]), self.labels[index]
        return image, label

    def _check_exists_(self) -> bool:
        return all((self.data_dir/file).exists() for file in self.files)
    