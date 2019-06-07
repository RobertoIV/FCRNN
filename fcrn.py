import torch
from torch import nn
from torch.nn import functional as F
from resnet import ResnetEncoder


class FCRN(nn.Module):
    def __init__(self, init=False):
        nn.Module.__init__(self)
        self.encoder = ResnetEncoder(init)
        self.decoder = UpProjDecoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    
class UpProjDecoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(2048, 1024, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.upproj1 = UpProject(1024, 512)
        self.upproj2 = UpProject(512, 256)
        self.upproj3 = UpProject(256, 128)
        self.upproj4 = UpProject(128, 64)
        self.dropout = nn.Dropout2d()
        self.pred = nn.Conv2d(64, 1, 3, 1, 1)
        self.upsample = nn.Upsample(size=(228, 304), mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(self.upproj1(x))
        x = F.relu(self.upproj2(x))
        x = F.relu(self.upproj3(x))
        x = F.relu(self.upproj4(x))
        
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pred(x)
        x = F.relu(x)
        x = self.upsample(x)
        
        return x
    
    
class UpConv(nn.Module):
    """
    Up convolution layer with fast computation.
    Stride defaults to 1, kernels size defaults to 5
    """
    
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)
        # Names are as those in the paper.
        # Note the author's naming in their implementation is not consistent
        # with the paper. So "B" here is actually "C" in their implementation
        self.convA = nn.Conv2d(in_channels, out_channels, (3, 3), bias=False)
        self.convB = nn.Conv2d(in_channels, out_channels, (3, 2), bias=False)
        self.convC = nn.Conv2d(in_channels, out_channels, (2, 3), bias=False)
        self.convD = nn.Conv2d(in_channels, out_channels, (2, 2), bias=False)
        
        
    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return: (B, C, 2H, 2W)
        """
        # Note the original implementation's padding is incorrect.
        # That is also mentioned in their issues
        # The correct version should be as follows
        B, C, H, W = x.size()
        # top left
        mapA = self.convA(F.pad(x, [1, 1, 1, 1]))
        # top right (3, 2)
        mapB = self.convB(F.pad(x, [0, 1, 1, 1]))
        # bottom left (2, 3)
        mapC = self.convC(F.pad(x, [1, 1, 0, 1]))
        # bottom right
        mapD = self.convD(F.pad(x, [0, 1, 0, 1]))
        
        # top part (B, C, H, W, 2)
        top = torch.stack((mapA, mapB), dim=-1)
        # bottom part (B, C, H, W, 2)
        bottom = torch.stack((mapC, mapD), dim=-1)
        # final (B, C, H, 2, W, 2)
        final = torch.stack((top, bottom), dim=3)
        # reshape (B, C, 2H, 2W)
        final = final.view(B, -1, 2*H, 2*W)
        
        return final
    
class UpProject(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)
        # branch 1
        self.upconv1 = UpConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # branch 2
        self.upconv2 = UpConv(in_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # branch 1
        x1 = F.relu(self.bn1(self.upconv1(x)))
        x1 = self.bn2(self.conv1(x1))
        # branch 2
        x2 = self.bn3(self.upconv2(x))
        # add output
        return F.relu(x1 + x2)
    
    
if __name__ == '__main__':
    res = FCRN()
    
    B = 32
    H = 288
    W = 304
    C = 3
    x = torch.rand(B, C, H, W)
    a = res(x)
    print(a.size())
