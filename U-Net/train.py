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

def train():
    use_cuda = False
    
    
    device = torch.device("cuda" if use_cuda else "cpu")
    model = UNet(2, 2, start_channel, net_activation).to(device)
    
    
    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC(win=9)
    
    loss_smooth = smoothloss

    transform = SpatialTransform().to(device)
    diff_transform = DiffeomorphicTransform(time_step=7).to(device)
    
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    lossall = np.zeros((3, iteration))
    train_set = TrainDataset(os.path.join(datapath, 'train'))
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    test_set = TestDataset(os.path.join(datapath, 'val'))
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)
    model_dir = './L2ss_{}_Seed_{}_Chan_{}_{}_Smth_{}_Set_{}_LR_{}_BZ_{}/'.format(using_l2, global_seed, start_channel, net_activation, smooth, trainingset, lr, bs)
    model_dir_pth1 = './L2ss_{}_Seed_{}_Chan_{}_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Pth1/'.format(using_l2, global_seed, start_channel, net_activation, smooth, trainingset, lr, bs)
    model_dir_pth2 = './L2ss_{}_Seed_{}_Chan_{}_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Pth2/'.format(using_l2, global_seed, start_channel, net_activation, smooth, trainingset, lr, bs)
    csv_name = 'L2ss_{}_Seed_{}_Chan_{}_{}_Smth_{}_Set_{}_LR_{}_BZ_{}.csv'.format(using_l2, global_seed, start_channel, net_activation, smooth, trainingset, lr, bs)
    if os.path.exists(csv_name):
        assert 0==1
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(model_dir_pth1):
        os.mkdir(model_dir_pth1)
    if not os.path.isdir(model_dir_pth2):
        os.mkdir(model_dir_pth2)
    
    
    step = 1

    while step <= iteration:
        for mov_img, fix_img in training_generator:

            fix_img = fix_img.to(device).float()
            mov_img = mov_img.to(device).float()
            
            f_xy = model(mov_img, fix_img)
            # Df_xy = diff_transform(f_xy)
            Df_xy = f_xy
            
            __, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
           
            loss1 = loss_similarity(fix_img, warped_mov) # GT shall be 1st Param
            loss2 = loss_smooth(f_xy)
            
            loss = loss1 + smooth * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item(),loss1.item(),loss2.item()])
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.5f}" - sim "{2:.5f}" -smo "{3:.5f}" '.format(step, loss.item(),loss1.item(),loss2.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0) or (step==1):
                with torch.no_grad():
                    Dices_Validation = []
                    for vmov_img, vfix_img, vmov_lab, vfix_lab in test_generator:
                        model.eval()
                        V_xy = model(vmov_img.float().to(device), vfix_img.float().to(device))
                        # DV_xy = diff_transform(V_xy)
                        DV_xy = V_xy
                        
                        grid, warped_vmov_lab = transform(vmov_lab.float().to(device), DV_xy.permute(0, 2, 3, 1), mod = 'nearest')
                        dice_bs = dice(warped_vmov_lab[0,...].data.cpu().numpy().copy(),vfix_lab[0,...].data.cpu().numpy().copy())
                        Dices_Validation.append(dice_bs)
                    
                    modelname = 'DiceVal_{:.5f}_Step_{:09d}.pth'.format(np.mean(Dices_Validation), step)
                    csv_dice = np.mean(Dices_Validation)
                    if step <= iteration / 2.0: 
                        save_checkpoint(model.state_dict(), model_dir_pth1, modelname)
                    else:
                        save_checkpoint(model.state_dict(), model_dir_pth2, modelname)
                    np.save(model_dir + 'Loss.npy', lossall)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice])
            if (step % (n_checkpoint * 10) == 0):
                sample_path = os.path.join(model_dir, '{:08d}-images.jpg'.format(step))
                save_flow(mov_img, fix_img, warped_mov, grid.permute(0, 3, 1, 2), sample_path)
            step += 1

            if step > iteration:
                break
        print("one epoch pass")
    np.save(model_dir + '/Loss.npy', lossall)

def save_flow(X, Y, X_Y, f_xy, sample_path):
    x = X.data.cpu().numpy()
    y = Y.data.cpu().numpy()
    
    x_pred = X_Y.data.cpu().numpy()
    
    x_pred = x_pred[0,...]
    x = x[0,...]
    y = y[0,...]
    
    flow = f_xy.data.cpu().numpy()
    op_flow =flow[0,:,:,:]
    
    
    plt.subplots(figsize=(7, 4))
    # plt.subplots()
    plt.subplot(231)
    plt.imshow(x[0, :, :], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(y[0, :, :], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(x_pred[0, :, :], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(234)
    
    interval = 7
    for i in range(0,op_flow.shape[1]-1,interval):
        plt.plot(op_flow[0,i,:], op_flow[1,i,:],c='g',lw=1)

    for i in range(0,op_flow.shape[2]-1,interval):
        plt.plot(op_flow[0,:,i], op_flow[1,:,i],c='g',lw=1)
        
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(abs(x[0, :, :]-y[0, :, :]), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(abs(x_pred[0, :, :]-y[0, :, :]), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(sample_path,bbox_inches='tight')
    plt.close()
    
train()
