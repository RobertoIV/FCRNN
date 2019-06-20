import torch
from torch import nn
from torch.nn import functional as F
from resnet import ResnetEncoder
from utils import vis_logger
import math


class FCRN(nn.Module):
    def __init__(self, init=False):
        nn.Module.__init__(self)
        self.encoder = ResnetEncoder(init)
        self.decoder = UpProjDecoder()
        
        # encoder modules
        self.conv1 = self.encoder.conv1
        self.bn1 = self.encoder.bn1
        self.relu = self.encoder.relu
        self.maxpool = self.encoder.maxpool
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4
        
        # decoder modules
        self.conv = self.decoder.conv
        self.bn = self.decoder.bn
        self.upproj1 = self.decoder.upproj1
        self.upproj2 = self.decoder.upproj2
        self.upproj3 = self.decoder.upproj3
        self.upproj4 = self.decoder.upproj4
        
        # self.dropout = nn.Dropout2d()
        self.upsample = self.decoder.upsample
        self.pred = self.decoder.pred
        self.normal_pred = self.decoder.normal_pred


        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # at this point, 1/4 resolution
    
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        # at this point 1/32 resolution, 2048

        x = self.bn(self.conv(c4))
        
        
        x = torch.cat((c4, x), dim=1)
        x = self.upproj1(x)
        x = torch.cat((c3, x), dim=1)
        x = self.upproj2(x)
        x = torch.cat((c2, x), dim=1)
        x = self.upproj3(x)
        u3 = x
        x = torch.cat((c1, x), dim=1)
        x = self.upproj4(x)

        # x = self.dropout(x)
        # depth prediction
        x1 = x
        x1 = self.pred(x1)
        x1 = F.relu(x1)
        x1 = self.upsample(x1)

        # surface normal prediction
        # take the last three channels
        x2 = u3[:, -3:]
        # x2 = self.normal_pred(x2)
        x2 = self.upsample(x2)
        x2 = F.normalize(x2, dim=1)

        return x1, x2
    
class UpProjDecoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        # resnet 50
        # self.conv1 = nn.Conv2d(2048, 1024, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(1024)
        # self.upproj1 = UpProject(1024, 512)
        # self.upproj2 = UpProject(512, 256)
        # self.upproj3 = UpProject(256, 128)
        # self.upproj4 = UpProject(128, 64)
        # self.dropout = nn.Dropout2d()
        # self.pred = nn.Conv2d(64, 1, 3, 1, 1)
        
        # resnet 18
        self.conv = nn.Conv2d(512, 512, 1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        # self.upproj1 = UpProject(512, 256)
        # self.upproj2 = UpProject(256, 128)
        # self.upproj3 = UpProject(128, 64)
        # self.upproj4 = UpProject(64, 32)
        self.upproj1 = UpProject(1024, 256)
        self.upproj2 = UpProject(512, 128)
        self.upproj3 = UpProject(256, 64)
        self.upproj4 = UpProject(128, 32)
        # self.dropout = nn.Dropout2d()
        self.pred = nn.Conv2d(32, 1, 3, 1, 1)
        self.normal_pred = nn.Conv2d(32, 3, 3, 1, 1)
        
        
        self.upsample = nn.Upsample(size=(224, 320), mode='bilinear', align_corners=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.upproj1(x)
        x = self.upproj2(x)
        x = self.upproj3(x)
        x = self.upproj4(x)
        
        # x = self.dropout(x)
        # depth prediction
        x1 = x
        x1 = self.pred(x1)
        x1 = F.relu(x1)
        x1 = self.upsample(x1)

        # surface normal prediction
        x2 = x
        x2 = self.normal_pred(x2)
        x2 = F.normalize(x2, dim=1)
        x2 = self.upsample(x2)
        
        return x1, x2
    
    
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


