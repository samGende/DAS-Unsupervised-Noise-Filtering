import torch
import numpy as np 
import os
from kmeans_gpu import KMeans
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(2)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

type = 'nosubsampled'
dir = f'./Data/synthetic-DAS/train-syntheticDAS/CWT-edDAS/{type}'
out_dir ='./Data/clusterResults/'

files = os.listdir(dir)
files.sort()
files = files[:25]
print(files)
print(f'{len(files)} files in directory')

sample = torch.load(f'{dir}/{files[0]}')
sample = sample.to(device)
print(f'samples device is {sample.device}')
print(sample.shape)

#info for loading files
#samples_per_subsample = 25
nChannels = sample.shape[0]
nSamples = sample.shape[1]
n_features = sample.shape[2]
nfiles = len(files)

#load files 
trainingData = torch.empty((nChannels, nSamples * nfiles, n_features), dtype=torch.float64)
print(trainingData.shape)
print(nfiles)
for index, file in enumerate(files):
  file = dir + '/' + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = torch.load(file)

# Reshape data and push to device  
trainingData = torch.reshape(trainingData, (nChannels * nSamples * nfiles, -1)).float().to(device)

print("training data shape after reshape", trainingData.shape)

#subsample for when data is to large
if(trainingData.shape[0] >= 70000000):
    trainingData = trainingData[::2,:]

print(f'training data is now loaded on {trainingData.device}')
print(trainingData.dtype)
#Setup kmeans 

n_clusters = 3

kmeans = KMeans(
    n_clusters=n_clusters,
    max_iter=1000,
    tolerance=1e-4,
    distance='euclidean',
    sub_sampling=None,
    max_neighbors=15,
)

labels, centers = kmeans.fit_predict(trainingData)

print(labels.shape)
print(centers.shape)

torch.save(labels, f'{out_dir}/gpuKmeansSynth_Dt_{type}_labels')
torch.save(centers, f'{out_dir}/gpuKmeansSynth_Dt_{type}_centers')
