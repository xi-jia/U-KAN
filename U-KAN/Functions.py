"""
Helper functions from https://github.com/zhangjun001/ICNet.

Some functions has been modified.
"""

import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir
from os.path import join
import itertools

import pickle
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_acdc_data_2d(data_path, filename):
    
    
    frame1 = 'ES'
    frame2 = 'ED'
    
    image_frame1 = pkload(os.path.join(data_path, filename, 'crop_128','sa_'+frame1+'_crop_mid.pkl'))
    image_frame2 = pkload(os.path.join(data_path, filename, 'crop_128','sa_'+frame2+'_crop_mid.pkl'))
    
    image_frame1 = np.reshape(image_frame1, (1,) + image_frame1.shape)
    image_frame2 = np.reshape(image_frame2, (1,) + image_frame2.shape)
    
    return image_frame1, image_frame2

class TrainDataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root=''):
        self.filename = [f for f in sorted(listdir(root))]
        self.data_path = root
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

    def __getitem__(self, index):
        'Generates one sample of data'
        
        mov_img, fix_img = load_acdc_data_2d(self.data_path, self.filename[index])
        
        return  mov_img, fix_img

def load_acdc_test_flow_data(data_path, filename, frame):

    image = pkload(os.path.join(data_path, filename, 'crop_128','sa_'+frame+'_crop_mid.pkl'))
    nim = nib.load(os.path.join(data_path, filename, 'crop_128', 'label_sa_'+frame+'_crop.nii.gz'))
    seg = nim.get_fdata()[:, :, :,0]
    mid_ven_idx = int(round((seg.shape[2] - 1) * 0.5))  # 50% from basal
    rand_z = mid_ven_idx
    seg = seg[...,rand_z]
    
    image = np.reshape(image, (1,) + image.shape)
    seg = np.reshape(seg, (1,) + seg.shape)
    
    return image, seg
class TestDataset(Data.Dataset):
    def __init__(self, root=''):
        super(TestDataset, self).__init__()
        self.data_path = root
        self.filename = [f for f in sorted(listdir(self.data_path))]
        
    def __getitem__(self, index):

        input_es, seg_es = load_acdc_test_flow_data(self.data_path, self.filename[index], 'ES')
        input_ed, seg_ed = load_acdc_test_flow_data(self.data_path, self.filename[index], 'ED')
        
        return input_es, input_ed, seg_es, seg_ed
        
    def __len__(self):
        return len(self.filename)
