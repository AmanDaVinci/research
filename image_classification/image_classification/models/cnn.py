import torch
import torch.nn as nn
from typing import Any, List, Tuple, Dict, Optional, Callable


class CNN(nn.Module):
    
    def __init__(self, in_channels: int, out_dim: int, 
                 device: torch.device, **kwargs):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=out_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        x, y = batch
        x = x.to(self.device)
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        accuracy = (y_pred.argmax(axis=1)==y).float().mean().item()
        return loss, accuracy