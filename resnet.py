import torchvision
import torch
from torch import nn
from torch.nn import functional as F


class ResnetEncoder(nn.Module):
    def __init__(self, download=False):
        nn.Module.__init__(self)
        resnet50 = torchvision.models.resnet50(download)
        # only retain useful modules
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # at this point, 1/4 resolution
    
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # at this point 1/32 resolution, 2048
    
        return x

        

