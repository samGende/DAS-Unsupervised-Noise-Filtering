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

sample = torch.load(DAS_Directory +'/' +file_list[0])
print(file_list[0])
print(sample.shape)

#parameters for cwt
dt =0.5
dj =0.5
w0 =8
minSpaceFrq = 0.002
maxSpaceFrq = 0.12
n_features = 30
n_samples = sample.shape[1]

n_channels = sample.shape[0]
samples_per_second = 50
samples_per_sub_sample = 25
space_log = np.logspace(np.log10(minSpaceFrq), np.log10(maxSpaceFrq), n_features)
time_scales= cwt.get_scales(dt, dj, w0, n_samples)


transform_dir = "./Data/synthetic-DAS/train-syntheticDAS/CWT-edDAS/subsampled"
print(len(file_list))
for index, file in tqdm(enumerate(file_list)):
    sample = np.array(torch.load(DAS_Directory +'/' +file))
    transform = cwt.transform_window(sample, n_channels, samples_per_second, samples_per_sub_sample, space_log, time_scales, start_window=0, end_window=5950, window_length=238, subsampling = True, derivative = True)
    torch.save(transform, transform_dir +'/'+ f'transform{index:04d}_window2.pt')
    if(index == 34):
        break
