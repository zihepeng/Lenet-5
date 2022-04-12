import os
from os import walk
import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn


class Net(nn.Module):
    """Define the model architecture.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)

        )
        self.full_connection = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(36864, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 15),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        size = x.size(0)
        x = self.feature_extraction(x)
        x = x.view(size, -1)
        x = self.full_connection(x)
        return x



