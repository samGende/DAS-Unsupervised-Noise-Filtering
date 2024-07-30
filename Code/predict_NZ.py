import torch 
import numpy as np
from kmeans_gpu import KMeans
import matplotlib.colors as mcolors
from utilities import cwt, paper_cwt, DAS, models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(2)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)




dir = './Data/CWT_4min/CWTNZ_Dt_SS'
out_dir ='./Data/clusterResults/inference'

sample = torch.tensor(np.load(f'{dir}/cwt_2023p152354.npy'))
print(f'samples device is {sample.device}')
files = os.listdir(dir)
files.sort()
files = files[25:]
print(files)
print(f'{len(files)} files in directory')


#info for loading files
#samples_per_subsample = 25
n_channels = sample.shape[0]
n_samples = sample.shape[1]
n_features = sample.shape[2]
nfiles = len(files)

#load files 
trainingData = np.empty((nfiles, n_channels, n_samples, n_features), dtype=np.float64)
print(trainingData.shape)
print(nfiles)
for index, file in enumerate(files):
  file = dir + '/' + file
  trainingData[index,:,:,:] = np.load(file)
print("data shape before reshape", trainingData.shape)


#parameters for cwt
dt =0.5
dj =0.5
w0 =8
minSpaceFrq = 0.002
maxSpaceFrq = 0.12
samples_per_second = 50
samples_per_sub_sample = 25
space_features = 30
space_log = np.logspace(np.log10(minSpaceFrq), np.log10(maxSpaceFrq), space_features)
time_scales= cwt.get_scales(dt, dj, w0, n_samples)

#init kmeans 
n_clusters = 3
kmeans = KMeans(n_clusters, distance='euclidean')
#Load clustering from the training Data
centers = torch.load('./Data/clusterResults/gpuKmeansNZ_Dt_SS_centers', map_location='cpu').float().to(device)
labels = torch.load('./Data/clusterResults/gpuKmeansNZ_Dt_SS_labels', map_location='cpu').to(device)
'Clustering is of the first 25 files'


#Decide what scales to Mute
n_scales_muted = 6

differences = torch.abs(centers[0,:time_scales.shape[0]] - centers[1,:time_scales.shape[0]])
rank = np.argsort(differences.detach().cpu().numpy())

print(f'differences in time scales of centers {differences[rank][-n_scales_muted:]}')

scales_to_mute = np.zeros(time_scales.shape)
scales_to_mute[rank[-n_scales_muted:]] = 1

#perform mute and inverse on all the data 
for index, transform in enumerate(trainingData):
    #get the prediction from the k-means centers
    flatt_features = transform.reshape(transform.shape[0] * transform.shape[1], -1)
    flatt_labels = kmeans.predict(torch.tensor(flatt_features).float().to(device), centers, distance='euclidean').detach().cpu().numpy()
    labels= np.reshape(flatt_labels, (transform.shape[0], transform.shape[1]))
    car_clusters = [0,2]
    flatt_mask = np.isin(flatt_labels, car_clusters)
    mask  = np.reshape(flatt_mask, (transform.shape[0], transform.shape[1]))


    muted_inverse = cwt.mute((transform[:,:,:time_scales.shape[0]]), time_scales, mask, scales_to_mute, dj, dt, w0)
    inverse = cwt.inverse_DAS(transform, time_scales, dj, dt, w0)

    #Plot Muted DAS and Cluster Labels 
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12,6))
    ax1.imshow((muted_inverse / muted_inverse.std()), origin='lower', interpolation='nearest', cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
    cmap = mcolors.ListedColormap(['yellow', 'green', 'red', 'blue'])
    ax2.imshow((inverse / inverse.std()), origin='lower', interpolation='nearest', cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title(" DAS Inverse with mute")
    ax2.set_title("Normal Inverse")
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{files[index]}_plot', format='png')
    np.save(f'{out_dir}/{files[index]}_invers.npy', inverse)
    np.save(f'{out_dir}/{files[index]}_muted.npy', muted_inverse)
