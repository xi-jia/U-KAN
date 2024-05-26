import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from Models import *
from Functions import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
from natsort import natsorted
import csv
import random

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = ArgumentParser()


parser.add_argument("--global_seed", type=int,
                    dest="global_seed", default=0, help="global_seed")

parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=804001,
                    help="number of total iterations")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=2010,
                    help="frequency of saving models")

parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.01)

parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='../ACDC/',
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")

parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--net_activation", type=str,
                    dest="net_activation",
                    default='ReLU',
                    help="net_activation")
opt = parser.parse_args()

global_seed = opt.global_seed
lr = opt.lr
bs = opt.bs
iteration = opt.iteration
n_checkpoint = opt.checkpoint

using_l2 = opt.using_l2
smooth = opt.smth_labda


datapath = opt.datapath
trainingset = opt.trainingset

start_channel = opt.start_channel
net_activation = opt.net_activation

####################################################################################
##################################Reproducibility###################################
####################################################################################

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(global_seed)
setup_seed(seed=global_seed, cuda_deterministic=True)

####################################################################################
##################################Reproducibility###################################
####################################################################################


def dice(pred1, truth1):
    dice_k=[]
    # mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    # print(len(mask4_value2))
    # mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    for k in mask4_value2[1:]:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        dice_k.append(intersection / (np.sum(pred) + np.sum(truth)))
    return np.mean(dice_k)

def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    # disp = disp.transpose(1, 2, 3, 0)
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    
    import pystrum.pynd.ndutils as nd
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def test(modelpath):
    
    use_cuda = False
    
    
    device = torch.device("cuda" if use_cuda else "cpu")
    model = UNet(2, 2, start_channel, net_activation).to(device)
    
    
    transform = SpatialTransform().to(device)
    diff_transform = DiffeomorphicTransform(time_step=7).to(device)
    model.load_state_dict(torch.load(modelpath))
    
    model.eval()
    transform.eval()
    diff_transform.eval()
    
    Dices_RVLVMyo=[]
    NegJ_RVLVMyo=[]
    GradJ_RVLVMyo=[]
    SDLogJ_RVLVMyo=[]
    
    test_set = TestDataset(os.path.join(datapath, 'test'))
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)
    
    for mov_img, fix_img, mov_lab, fix_lab in test_generator:
        with torch.no_grad():
            V_xy = model(mov_img.float().to(device), fix_img.float().to(device))
            # V_xy = diff_transform(vf_xy)
            
            __, warped_mov_lab = transform(mov_lab.float().to(device), V_xy.permute(0, 2, 3, 1), mod = 'nearest')
            hh, ww = V_xy.shape[-2:]
            V_xy = V_xy.detach().cpu().numpy()
            V_xy[:,0,:,:] = V_xy[:,0,:,:] * (hh-1) / 2
            V_xy[:,1,:,:] = V_xy[:,1,:,:] * (ww-1) / 2
            #print('V_xy.shape . . . ', V_xy.shape)  #([1, 3, 160, 192, 224])
            #print('warped_mov_lab.shape . . . ', warped_mov_lab.shape) #([1, 1, 160, 192, 224])
            
            for bs_index in range(bs):
                dice_bs = dice(warped_mov_lab[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy())
                Dices_RVLVMyo.append(dice_bs)
                jac_det = jacobian_determinant_vxm(V_xy[0, :, :, :])
                mag_j_det = np.abs(np.gradient(jac_det)).mean()
                negJ = np.sum(jac_det <= 0) / 160 / 192 * 100
                NegJ_RVLVMyo.append(negJ)
                # if np.min(jac_det)<0:
                    # print(np.min(jac_det))
                    # sdJ = (jac_det - np.min(jac_det) + 1).clip(0.000000001, 1000000000)
                # else:
                    # sdJ = (jac_det + 1).clip(0.000000001, 1000000000)
                log_jac_det = jac_det.std()#np.log(sdJ).std()
                SDLogJ_RVLVMyo.append(log_jac_det)
                GradJ_RVLVMyo.append(mag_j_det)

    print('Dice: {:.3f} ± {:.3f}'.format(np.mean(Dices_RVLVMyo), np.std(Dices_RVLVMyo)))
    print('NegJ 100% {:.3f} ± {:.3f} {}'.format(np.mean(NegJ_RVLVMyo), np.std(NegJ_RVLVMyo), np.mean(NegJ_RVLVMyo)))
    print('SDJ: {:.3f} ± {:.3f}'.format(np.mean(SDLogJ_RVLVMyo), np.std(SDLogJ_RVLVMyo)))
    print('GradJ: {:.3f} ± {:.3f}'.format(np.mean(GradJ_RVLVMyo), np.std(GradJ_RVLVMyo)))

    return np.mean(Dices_RVLVMyo)


if __name__ == '__main__':

    model_path='./L2ss_{}_Seed_{}_Chan_{}_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Pth2/'.format(using_l2, global_seed, start_channel, net_activation, smooth, trainingset, lr, bs)
    model_idx = -1
    from natsort import natsorted
    print('Best model: {}'.format(natsorted(os.listdir(model_path))[model_idx]))
    
    temp= test(model_path + natsorted(os.listdir(model_path))[model_idx])
    
    