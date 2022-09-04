import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss



import numpy as np
def histcal(x, bins=256, min=0.0, max=1.0):
    n,c,h,w = x.size()
    n_batch = n
    row_m = h
    row_n = w
    channels = c

    delta = (max - min) / bins
    BIN_Table = np.arange(0, bins, 1)
    BIN_Table = BIN_Table * delta

    zero = torch.tensor([[[0.0]]],requires_grad=False).cuda()
    zero = zero.repeat(n,c,1)
    temp = torch.ones(size=x.size()).cuda()
    temp1 = torch.zeros(size=x.size()).cuda()
    for dim in range(1, bins - 1, 1):
        h_r = BIN_Table[dim]  # h_r
        h_r_sub_1 = BIN_Table[dim - 1]  # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1]  # h_(r+1)

        h_r = torch.tensor(h_r).float().cuda()
        h_r_sub_1 = torch.tensor(h_r_sub_1).float().cuda()
        h_r_plus_1 = torch.tensor(h_r_plus_1).float().cuda()

        h_r_temp = h_r * temp
        h_r_sub_1_temp = h_r_sub_1 * temp
        h_r_plus_1_temp = h_r_plus_1 * temp

        mask_sub = torch.where(torch.greater(h_r_temp, x) & torch.greater(x, h_r_sub_1_temp), temp, temp1)
        mask_plus = torch.where(torch.greater(x, h_r_temp) & torch.greater(h_r_plus_1_temp, x), temp, temp1)

        temp_mean1 = torch.mean((((x - h_r_sub_1) * mask_sub).view(n_batch, channels, -1)), dim=-1)
        temp_mean2 = torch.mean((((h_r_plus_1 - x) * mask_plus).view(n_batch, channels, -1)), dim=-1)

        if dim == 1:
            temp_mean = torch.add(temp_mean1, temp_mean2)
            temp_mean = torch.unsqueeze(temp_mean, -1)  # [1,1,1]
        else:
            if dim != bins - 2:
                temp_mean_temp = torch.add(temp_mean1, temp_mean2)
                temp_mean_temp = torch.unsqueeze(temp_mean_temp, -1)
                temp_mean = torch.cat([temp_mean, temp_mean_temp], dim=-1)
            else:
                zero = torch.cat([zero, temp_mean], dim=-1)
                temp_mean_temp = torch.add(temp_mean1, temp_mean2)
                temp_mean_temp = torch.unsqueeze(temp_mean_temp, -1)
                temp_mean = torch.cat([temp_mean, temp_mean_temp], dim=-1)

    # diff = torch.abs(temp_mean - zero)
    return temp_mean




###################################################################### Wavelet transform

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_HL, x_LH, x_HH), 1)


# def iwt_init(x):
#     r = 2
#     in_batch, in_channel, in_height, in_width = x.size()
#     # print([in_batch, in_channel, in_height, in_width])
#     out_batch, out_channel, out_height, out_width = in_batch, int(
#         in_channel / (r ** 2)), r * in_height, r * in_width
#     x1 = x[:, 0:out_channel, :, :] / 2
#     x2 = x[:, out_channel:out_channel * 2, :, :] / 2
#     x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
#     x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
#
#     h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
#
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h



class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


# class IWT(nn.Module):
#     def __init__(self):
#         super(IWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         return iwt_init(x)