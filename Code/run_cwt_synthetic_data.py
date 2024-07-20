from utilities import cwt
from utilities import DAS 
import os
import numpy as np
import torch
from tqdm import tqdm

DAS_Directory = "./Data/synthetic-DAS/train-syntheticDAS/samples-DAS"

file_list =  os.listdir(DAS_Directory)
file_list.sort()

n_samples = 3000

sample = torch.load(DAS_Directory +'/' +file_list[0])[:,:n_samples]
print(sample.shape)

#parameters for cwt
dt =0.02
dj =0.5
w0 =8
minSpaceFrq = 0.002
maxSpaceFrq = 0.12
n_features = 30


n_channels = sample.shape[0]
samples_per_second = 50
samples_per_sub_sample = 1
space_log = np.logspace(np.log10(minSpaceFrq), np.log10(maxSpaceFrq), n_features)
time_scales= cwt.get_scales(dt, dj, w0, n_samples)

transform_dir = "./Data/synthetic-DAS/train-syntheticDAS/CWT-edDAS"
print(len(file_list))
for index, file in tqdm(enumerate(file_list)):
    sample = np.array(torch.load(DAS_Directory +'/' +file))
    sample2 = sample[:,n_samples:]
    sample1= sample[:,:n_samples]
    transform1 = cwt.transform_window(sample1, n_channels, samples_per_second, samples_per_sub_sample, space_log, time_scales, start_window=0, end_window=n_channels, window_length=n_samples, subsampling = False, derivative = False)
    transform2 = cwt.transform_window(sample2, n_channels, samples_per_second, samples_per_sub_sample, space_log, time_scales, start_window=0, end_window=n_channels, window_length=n_samples, subsampling = False, derivative = False)
    torch.save(transform1, transform_dir +'/'+ f'transform{index:04d}_window1.pt')
    torch.save(transform2, transform_dir +'/'+ f'transform{index:04d}_window2.pt')


