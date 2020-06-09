import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


from collections import OrderedDict

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        
        self.encoder = torchvision.models.resnet50()
        self.encoder.fc = nn.Identity()
        self.concat_dim = 180 * 6
        
        self.compress = nn.Sequential(OrderedDict([
            ('linear0', nn.Linear(2048, 180)),
            ('drop', nn.Dropout(p = 0.5)),
            ('relu', nn.ReLU()),
        ]))
        
        self.classification = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(self.concat_dim, 256)),
        ]))
        
        self.x_offset = nn.Sequential(OrderedDict([
            ('xoff1', nn.Linear(self.concat_dim, 256)),
            ('tanh1', nn.Tanh())
        ]))
        
        self.y_offset = nn.Sequential(OrderedDict([
            ('yoff1', nn.Linear(self.concat_dim, 256)),
            ('tanh2', nn.Tanh())
        ]))
        
        self.counts = nn.Sequential(OrderedDict([
            ('count1', nn.Linear(self.concat_dim, 90))
        ]))
        
        self.segmentation = nn.Sequential(OrderedDict([
            ('linear1_segmentation', nn.Linear(self.concat_dim, 25600)),
            ('sigmoid', nn.Sigmoid())
        ]))
        
    def forward(self, x):
        batch_size = x.shape[0]
        num_images = x.shape[1]
        channels = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]
        # Reshape here
        x = x.view(-1, channels, height, width)
        x = self.encoder(x)
        x = self.compress(x)
        x = x.view(-1, self.concat_dim)
        
        return self.segmentation(x)


class Yandhi(nn.Module):
    def __init__(self, count_dim=28):
        super(Yandhi, self).__init__()
        
        self.encoder = torchvision.models.resnet18()
        self.encoder.fc = nn.Identity()
        self.concat_dim = 100
        self.dropout = nn.Dropout(p = 0.2)
        self.bn = nn.BatchNorm1d(self.concat_dim)
        
        self.compress = nn.Sequential(OrderedDict([
            ('linear0', nn.Linear(512, 100)),
            ('relu', nn.ReLU()),
        ]))
        
        self.vehicle_map = nn.Sequential(OrderedDict([
            ('linear1_vehicle', nn.Linear(self.concat_dim, 6400)),
        ]))
        
        self.counts = nn.Sequential(OrderedDict([
            ('count1', nn.Linear(self.concat_dim, count_dim))
        ]))
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        # Reshape here
        x = x.view(-1, channels, height, width)
        x = self.encoder(x)
        x = self.compress(x)
        x = x.view(-1, self.concat_dim)
        x = self.bn(x)
        x = self.dropout(x)
    
        return self.vehicle_map(x), self.counts(x)