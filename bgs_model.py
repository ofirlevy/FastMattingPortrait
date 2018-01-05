import torch
import torch.nn as nn
import torch.nn.init as init


class CatConvLayer(nn.Module):
    def __init__(self, nChannels, nPadding, nDilation):
        super(CatConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=nChannels, out_channels=16, kernel_size=(3, 3), stride=1, padding=nPadding, dilation=nDilation, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input1, input2):
        out = torch.cat((input1, input2), 1)
        out = self.ReLU(self.conv1(out))
        return out




class MattNet(nn.Module):

    # TODO - how do I set the LR mult

    def __init__(self):
        super(MattNet, self).__init__()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2),
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=13, kernel_size=(3, 3), stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=2, dilation=2, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=4, dilation=4, groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=(3, 3), stride=1, padding=6, dilation=6, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=1, padding=8, dilation=8, groups=1, bias=True)
        self.convLast = nn.Conv2d(in_channels=64, out_channels=2,  kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.interp = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # feather
        self.convF1 = nn.Conv2d(in_channels=14, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm2d(nChannels=8)
        self.convF2 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        self._initialize_weights()


    def forward(self, x):
        x1 = self.conv1(x)    ## conv 2
        x2 = self.maxpool1(x)
        catOut = torch.cat((x1, x2), 1)   ## cat 16

        convOut1 = self.ReLU(self.conv2(catOut))  ## conv 2
        catOut = torch.cat((catOut, convOut1), 1)   ## cat 32

        convOut = self.ReLU(self.conv3(catOut))   ## conv 3
        catConvOut = torch.cat((convOut1, convOut), 1)
        catOut = torch.cat((catOut, convOut), 1)  ## cat 48

        convOut = self.ReLU(self.conv4(catOut))   ## conv 4
        catConvOut = torch.cat((catConvOut, convOut), 1)
        catOut = torch.cat((catOut, convOut), 1)  ## cat 64

        convOut = self.ReLU(self.conv5(catOut))   ## conv 5
        catConvOut = torch.cat((catConvOut, convOut), 1)

        convOut = self.ReLU(self.conv6(catConvOut))  ## conv 6
        convOut = self.interp(convOut)
                
        # fethering inputs:
        # {I, bkg, fwg, I^2, Ixfwg}
        xx = torch.split(convOut, 1, dim=0)
        background, foreground = xx        
        imgSqr = x * x
        foregroundCat = torch.cat((foreground, foreground, foreground, foreground), 0)
        imgMasked = x * foregroundCat  # TODO - need to duplicate foreground to 4 channels        
        featherInput = torch.cat((x, background, foreground, imgSqr, imgMasked), 0)                
        convOut =  self.ReLU(self.bn1(self.convF1(convOut)))
        convOut = self.convF2(convOut)
        
        xx = torch.split(convOut, 1, dim=0)
        a, b, c = xx
        a = a * background
        b = b * foreground
        c = a + b + c        
        c = self.sigmoid(c)

        return convOut
    

    



    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight)
