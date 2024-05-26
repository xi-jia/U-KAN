import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from kaconv.convkan import ConvKAN
from kaconv.kaconv import FastKANConvLayer
from torch.nn import Conv2d, BatchNorm2d

# def get_activation(name):
    # activations = {
        # 'Tanh': nn.Tanh(),
        # 'ReLU': nn.ReLU(),
        # 'PReLU': nn.PReLU(),
        # 'Softsign': nn.Softsign()
    # }
    # return activations.get(name, None)

class UNet(nn.Module):
    def __init__(self, in_channel, flow_dimension, start_channel, net_activation):
        self.in_channel = in_channel
        self.flow_dimension = flow_dimension
        self.start_channel = start_channel
        self.net_activation = net_activation

        bias_opt = False

        super(UNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt, net_activation=self.net_activation)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt, net_activation=self.net_activation)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt, net_activation=self.net_activation)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt, net_activation=self.net_activation)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt, net_activation=self.net_activation)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt, net_activation=self.net_activation)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt, net_activation=self.net_activation)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt, net_activation=self.net_activation)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt, net_activation=self.net_activation)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt, net_activation=self.net_activation)

        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3, stride=1, bias=bias_opt, net_activation=self.net_activation)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt, net_activation=self.net_activation)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt, net_activation=self.net_activation)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt, net_activation=self.net_activation)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt, net_activation=self.net_activation)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt, net_activation=self.net_activation)
        self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt, net_activation=self.net_activation)
        self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt, net_activation=self.net_activation)
        self.dc9 = self.outputs(self.start_channel * 2, self.flow_dimension, kernel_size=3, stride=1, padding=1, bias=False, net_activation=self.net_activation)
        
        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8, net_activation=self.net_activation)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4, net_activation=self.net_activation)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2, net_activation=self.net_activation)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2, net_activation=self.net_activation)
        
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, net_activation=None):
        if net_activation is not None:
            layer = FastKANConvLayer(in_channels = in_channels, out_channels = out_channels, padding=padding, kernel_size=kernel_size, stride=stride, use_base_update = False, kan_type=net_activation)
        else:
            raise Exception("This is a wrongly used activation.")
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0, bias=True, net_activation=None):
        if net_activation is not None:
            layer = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                FastKANConvLayer(in_channels = in_channels, out_channels = out_channels, padding=padding, kernel_size=kernel_size, stride=stride, use_base_update = False, kan_type=net_activation))
        else:
            raise Exception("This is a wrongly used activation.")
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, net_activation=None):
        if net_activation is not None:
            layer = FastKANConvLayer(in_channels = in_channels, out_channels = out_channels, padding=padding, kernel_size=kernel_size, stride=stride, use_base_update = False, kan_type=net_activation)
        else:
            raise Exception("This is a wrongly used activation.")
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        en = self.eninput(x_in)        
        
        e1 = self.ec1(en)
        e2 = self.ec2(e1)
        
        e3 = self.ec3(e2)
        e4 = self.ec4(e3)
        
        e5 = self.ec5(e4)
        e6 = self.ec6(e5)
        
        e7 = self.ec7(e6)
        e8 = self.ec8(e7)
        
        e9 = self.ec9(e8)

        d0 = torch.cat((self.up1(e9), e7), 1)

        d1 = self.dc1(d0)
        d2 = self.dc2(d1)

        d2 = torch.cat((self.up2(d2), e5), 1)

        d3 = self.dc3(d2)
        d4 = self.dc4(d3)

        d4 = torch.cat((self.up3(d4), e3), 1)

        d5 = self.dc5(d4)
        d6 = self.dc6(d5)

        d6 = torch.cat((self.up4(d6), e1), 1)
        
        d7 = self.dc7(d6)
        d8 = self.dc8(d7)

        flo = self.dc9(d8)
        return flo




class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        # d2, h2, w2 = mov_image.shape[-3:]
        h2, w2 = mov_image.shape[-2:]
        # grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        # grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        # grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_h = flow[:,:,:,0]
        flow_w = flow[:,:,:,1]
        # flow_w = flow[:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        
        # Remove Channel Dimension
        # disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        
        return sample_grid, warped

class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
    
        # print(flow.shape)
        h2, w2 = flow.shape[-2:]
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        
        
        for i in range(self.time_step):
            flow_h = flow[:,0,:,:]
            flow_w = flow[:,1,:,:]
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)
            
            deformation = torch.stack((disp_w,disp_h), dim=3)
    
            # print(deformation.shape)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear',padding_mode="border", align_corners = True)
            #Softsign
            #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
            #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
            #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
            
            # Remove Channel Dimension
            #disp_h = (grid_h + (flow_h)).squeeze(1)
            #disp_w = (grid_w + (flow_w)).squeeze(1)

            #sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, H, W, 2)
            #flow = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod)
        
        return flow



def smoothloss(y_pred):
    #print('smoothloss y_pred.shape    ',y_pred.shape)
    #[N,3,D,H,W]
    h2, w2 = y_pred.shape[-2:]
    # dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dz*dz))/2.0

"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv2d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)

class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 4))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice



class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
class SAD:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))
