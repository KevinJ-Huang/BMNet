from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from models.archs.arch_util import ConditionNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoupleLayer(nn.Module):
    def __init__(self, channels, substructor, condition_length,  clamp=5.):
        super().__init__()

        channels = channels
        # self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = True
        # condition_length = sub_len
        self.shadowpre = nn.Sequential(
            nn.Conv2d(4, channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2))
        self.shadowpro = ShadowProcess(channels // 2)


        self.s1 = substructor(self.split_len1 + condition_length, self.split_len2*2)
        self.s2 = substructor(self.split_len2 + condition_length, self.split_len1*2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        c_star = self.shadowpre(c)
        c_star = self.shadowpro(c_star)

        if not rev:
            # r2 = self.s2(torch.cat([x2, c_star], 1) if self.conditional else x2)
            r2 = self.s2(x2, c_star)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            # r1 = self.s1(torch.cat([y1, c_star], 1) if self.conditional else y1)
            r1 = self.s1(y1, c_star)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            # self.last_jac = self.log_e(s1) + self.log_e(s2)

        else: # names of x and y are swapped!
            # r1 = self.s1(torch.cat([x1, c_star], 1) if self.conditional else x1)
            r1 = self.s1(x1, c_star)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            # r2 = self.s2(torch.cat([y2, c_star], 1) if self.conditional else y2)
            r2 = self.s2(y2, c_star)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            # self.last_jac = - self.log_e(s1) - self.log_e(s2)

        return torch.cat((y1, y2), 1)

    # def jacobian(self, x, c=[], rev=False):
    #     return torch.sum(self.last_jac, dim=tuple(range(1, self.ndims+1)))
    def output_dims(self, input_dims):
        return input_dims




def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


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


class ShadowProcess(nn.Module):
    def __init__(self, channels):
        super(ShadowProcess, self).__init__()
        self.process = UNetConvBlock(channels, channels)
        self.Attention = nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.process(x)
        xatt = self.Attention(x)

        return xatt

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=False):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 1, 1, 0, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


# class MultiscaleDense(nn.Module):
#     def __init__(self,channel_in, channel_out, init):
#         super(MultiscaleDense, self).__init__()
#         self.conv_mul = nn.Conv2d(channel_out//2,channel_out//2,3,1,1)
#         self.conv_add = nn.Conv2d(channel_out//2, channel_out//2, 3, 1, 1)
#         self.op = DenseBlock(channel_in, channel_out, init)
#         self.fuse = nn.Conv2d(3 * channel_out, channel_out, 1, 1, 0)
#
#     def forward(self, x, s):
#
#         s_mul = self.conv_mul(s)
#         s_add = self.conv_add(s)
#         x_trans = s_mul*x+s_add
#
#         x = torch.cat([x,x_trans],1)
#
#         x1 = x
#         x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
#         x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')
#         x1 = self.op(x1)
#         x2 = self.op(x2)
#         x3 = self.op(x3)
#         x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
#         x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
#         x = self.fuse(torch.cat([x1, x2, x3], 1))
#
#         return x



class MultiscaleDense(nn.Module):
    def __init__(self,channel_in, channel_out, init):
        super(MultiscaleDense, self).__init__()
        self.conv_mul = nn.Conv2d(channel_out//2,channel_out//2,3,1,1)
        self.conv_add = nn.Conv2d(channel_out//2, channel_out//2, 3, 1, 1)
        self.down1 = nn.Conv2d(channel_out//2,channel_out//2,stride=2,kernel_size=2,padding=0)
        self.down2 = nn.Conv2d(channel_out//2, channel_out//2, stride=2, kernel_size=2, padding=0)
        self.op1 = DenseBlock(channel_in, channel_out, init)
        self.op2 = DenseBlock(channel_in, channel_out, init)
        self.op3 = DenseBlock(channel_in, channel_out, init)
        self.fuse = nn.Conv2d(3 * channel_out, channel_out, 1, 1, 0)

    def forward(self, x, s):
        s_mul = self.conv_mul(s)
        s_add = self.conv_add(s)
        # x_trans = s_mul*x+s_add
        # x = torch.cat([x,x_trans],1)

        x1 = x
        x2,s_mul2,s_add2 = self.down1(x),\
                           F.interpolate(s_mul, scale_factor=0.5, mode='bilinear'),F.interpolate(s_add, scale_factor=0.5, mode='bilinear')
        x3, s_mul3, s_add3 = self.down2(x2), \
                             F.interpolate(s_mul, scale_factor=0.25, mode='bilinear'), F.interpolate(s_add,scale_factor=0.25,mode='bilinear')
        x1 = self.op1(torch.cat([x1,s_mul*x1+s_add],1))
        x2 = self.op2(torch.cat([x2,s_mul2*x2+s_add2],1))
        x3 = self.op3(torch.cat([x3,s_mul3*x3+s_add3],1))
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x = self.fuse(torch.cat([x1, x2, x3], 1))

        return x




def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return MultiscaleDense(channel_in, channel_out, init)
            else:
                return MultiscaleDense(channel_in, channel_out, init)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor




class InvISPNet(nn.Module):
    def __init__(self, channel_in=3, subnet_constructor=subnet('DBNet'), block_num=4):
        super(InvISPNet, self).__init__()
        operations = []
        # level = 3
        self.condition = ConditionNet()
        self.condition.load_state_dict(torch.load('/home/jieh/Projects/Shadow/MainNet/pretrain/condition.pth'))
        for p in self.parameters():
            p.requires_grad = False

        channel_num = 16  # total channels at input stage
        self.CG0 = nn.Conv2d(channel_in, channel_num, 1, 1, 0)
        self.CG1 = nn.Conv2d(channel_num, channel_in, 1, 1, 0)
        self.CG2 = nn.Conv2d(channel_in, channel_num, 1, 1, 0)
        self.CG3 = nn.Conv2d(channel_num, channel_in, 1, 1, 0)

        for j in range(block_num):
            b = CoupleLayer(channel_num, substructor = subnet_constructor, condition_length=channel_num//2)  # one block is one flow step.
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.initialize()

        # self.transcon = nn.Conv2d(channel_num//2,1,1,1,0)

        # self.pyin = Lap_Pyramid_Conv(num_high=level)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    def forward(self, input, mask, gt, rev=False):
        b, c, m, n = input.shape
        maskcolor = self.condition(input,mask)
        maskfea = torch.cat([maskcolor,mask],1)

        if not rev:
            x = input
            out = self.CG0(x)
            out_list = []
            for op in self.operations:
                out_list.append(out)
                out = op.forward(out, maskfea, rev)
            out = self.CG1(out)
        else:
            out = self.CG2(gt)
            out_list = []
            for op in reversed(self.operations):
                out = op.forward(out, maskfea, rev)
                out_list.append(out)
            out_list.reverse()
            out = self.CG3(out)
        # return out, out[:, :4, :, :]
        return out, maskcolor



if __name__ == '__main__':
    level =3
    pyin = Lap_Pyramid_Conv(num_high=level)
    net = InvISPNet(channel_in=3,block_num=8)
    print('#generator parameters:',sum(param.numel() for param in net.parameters()))
    x = torch.randn(2, 3, 128, 128)
    out = pyin.pyramid_decom(x)
    for i in range(len(out)):
        x =torch.cat([x,out[i]],dim=1)
    print(x.size())
    out = net(x)
    print(out.shape)
