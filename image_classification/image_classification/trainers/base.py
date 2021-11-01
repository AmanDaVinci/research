import torch
import random
import numpy as np
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional, Callable

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from image_classification.utils import import_class, import_object

import wandb
import logging
log = logging.getLogger(__name__)


class Trainer:

    def __init__(self, config: dict):
        self.config = config
        self.set_seed(config.seed)
        self.data_dir = Path(config.data_dir)/config.dataset.class_name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config.output_dir)/wandb.run.id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(self.config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.load_data()
        self.load_model()
    
    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info(f"Setting seed: {seed}")

    def load_data(self):
        config = self.config.dataset
        train_preprocessor = import_object(
            "image_classification.preprocessors."+config.train_preprocessor,
            "preprocessor"
        )
        test_preprocessor = import_object(
            "image_classification.preprocessors."+config.test_preprocessor,
            "preprocessor"
        )
        dataset_class = import_class(config.module, config.class_name)
        train_dataset = dataset_class(
            is_train=True, transform=train_preprocessor, 
            data_dir=self.data_dir, download_data=config.download
        )
        eval_dataset = dataset_class(
            is_train=True, transform=test_preprocessor,
            data_dir=self.data_dir, download_data=config.download
        )
        test_dataset = dataset_class(
            is_train=False, transform=test_preprocessor,
            data_dir=self.data_dir, download_data=config.download
        )
        indices = range(len(train_dataset))
        train_indices = indices[:config.train_eval_split]
        eval_indices = indices[config.train_eval_split:]
        train_sampler = SubsetRandomSampler(train_indices)
        eval_sampler = SubsetRandomSampler(eval_indices)
        self.train_dl = DataLoader(train_dataset, config.batch_size, sampler=train_sampler)
        self.eval_dl = DataLoader(eval_dataset, config.batch_size*2, sampler=eval_sampler)
        self.test_dl = DataLoader(test_dataset, config.batch_size*2, shuffle=False)
        log.info(
            f"Loaded datasets: "
            f"train[{len(train_dataset)}] " 
            f"eval[{len(eval_dataset)}] " 
            f"test[{len(test_dataset)}] " 
        )

    def load_model(self): 
        config = self.config.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_class = import_class(config.module, config.class_name)
        self.model = model_class(**config, device=device)
        log.info(f"Loaded {config.class_name} on {device}")
    