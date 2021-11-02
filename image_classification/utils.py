import torch
import torch.nn as nn
from torch import optim

from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen
from importlib import import_module

from typing import Any, List, Tuple, Dict, Optional, Callable


def download(from_url: str, to_file: str) -> None:
    with urlopen(from_url) as response, open(to_file, 'wb') as out_file:
        copyfileobj(response, out_file)
def import_class(module: str, class_name: str) -> Any:
    return getattr(import_module(module), class_name)

def import_object(module: str, object_name: str) -> Any:
    return getattr(import_module(module), object_name)

def save(model: nn.Module, optimizer: optim, 
         loss: float, examples_seen: int, path: Path):
    state = {
        'loss': loss,
        'examples_seen': examples_seen,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, path)