import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



class ConditionNet(nn.Module):
    def __init__(self, channels):
        super(ConditionNet,self).__init__()
        self.convpre = nn.Conv2d(3, channels, 3, 1, 1)
        self.conv1 = DenseBlock(channels, channels)
        self.down1 = nn.Conv2d(channels, 2*channels, stride=2, kernel_size=2)
        self.conv2 = DenseBlock(2*channels, 2*channels)
        self.down2 = nn.Conv2d(2*channels, 4*channels, stride=2, kernel_size=2)
        self.conv3 = DenseBlock(4*channels, 4*channels)
        self.down3 = nn.Conv2d(4*channels, 8*channels, stride=2, kernel_size=2)
        self.conv4 = DenseBlock(8*channels, 8*channels)

        self.context4 = DenseBlock(8*channels,8*channels)
        self.context3 = DenseBlock(4*channels,4*channels)
        self.context2 = DenseBlock(2*channels,2*channels)
        self.context1 = DenseBlock(channels,channels)

        self.process4 = nn.Conv2d(8*channels,channels,1,1,0)
        self.process3 = nn.Conv2d(4*channels,channels,1,1,0)
        self.process2 = nn.Conv2d(2*channels,channels,1,1,0)

        self.merge3 = nn.Sequential(nn.Conv2d(12*channels,4*channels,1,1,0),CALayer(4*channels,4),nn.Conv2d(4*channels,4*channels,3,1,1))
        self.merge2 = nn.Sequential(nn.Conv2d(6*channels,2*channels,1,1,0),CALayer(2*channels,4),nn.Conv2d(2*channels,2*channels,3,1,1))
        self.merge1 = nn.Sequential(nn.Conv2d(3*channels,channels,1,1,0),CALayer(channels,4),nn.Conv2d(channels,channels,3,1,1))

        self.conv_last = nn.Sequential(nn.Conv2d(4*channels,channels,1,1,0),CALayer(channels,4),nn.Conv2d(channels,channels//2,3,1,1))


    def forward(self, x):
        x1 = self.conv1(self.convpre(x))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))

        x4 = self.context4(x4)
        x4_out = self.process4(F.interpolate(x4,scale_factor=8,mode='bilinear'))
        x4 = F.interpolate(x4,scale_factor=2,mode='bilinear')

        x3 = self.context3(self.merge3(torch.cat([x3,x4],1)))
        x3_out = self.process3(F.interpolate(x3,scale_factor=4,mode='bilinear'))
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')

        x2 = self.context2(self.merge2(torch.cat([x2, x3], 1)))
        x2_out = self.process2(F.interpolate(x2,scale_factor=2,mode='bilinear'))
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')

        x1 = self.context1(self.merge1(torch.cat([x1, x2], 1)))
        x1_out = x1

        xout = self.conv_last(torch.cat([x1_out,x2_out,x3_out,x4_out],1))

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

############################################################################################################


class GlobalContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio = 1.0,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out