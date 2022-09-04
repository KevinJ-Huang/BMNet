import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



class ConditionNet(nn.Module):
    def __init__(self, channels = 8):
        super(ConditionNet,self).__init__()
        self.convpre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv1 = DenseBlock(channels, channels)
        self.down1 = nn.Conv2d(channels, 2*channels, stride=2, kernel_size=2)
        self.conv2 = DenseBlock(2*channels, 2*channels)
        self.down2 = nn.Conv2d(2*channels, 4*channels, stride=2, kernel_size=2)
        self.conv3 = DenseBlock(4*channels, 4*channels)

        self.Global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0))
        self.context_g = DenseBlock(8 * channels, 4 * channels)

        self.context2 = DenseBlock(2 * channels, 2 * channels)
        self.context1 = DenseBlock(channels, channels)

        self.merge2 = nn.Sequential(nn.Conv2d(6*channels,2*channels,1,1,0),CALayer(2*channels,4),nn.Conv2d(2*channels,2*channels,3,1,1))
        self.merge1 = nn.Sequential(nn.Conv2d(3*channels,channels,1,1,0),CALayer(channels,4),nn.Conv2d(channels,channels,3,1,1))

        self.conv_last = nn.Conv2d(channels,3,3,1,1)


    def forward(self, x, mask):
        xpre = x/(torch.mean(x,1).unsqueeze(1)+1e-8)
        mask = torch.cat([mask,mask],1)
        x1 = self.conv1(self.convpre(torch.cat([xpre,x,mask],1)))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))

        x_global = self.Global(x3)
        _,_,h,w = x3.size()
        x_global = x_global.repeat(1,1,h,w)
        x3 = self.context_g(torch.cat([x_global,x3],1))

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x2 = self.context2(self.merge2(torch.cat([x2, x3], 1)))

        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x1 = self.context1(self.merge1(torch.cat([x1, x2], 1)))

        xout = self.conv_last(x1)

        return xout



############################################################################################################################


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)

        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out



def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)



######################################################################################################

class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x

