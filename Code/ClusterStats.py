import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from kmeans_gpu import KMeans
from utilities import clusters

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
trainingData = torch.empty((nChannels, nSamples * nfiles, n_features), dtype=torch.float32)
print(trainingData.shape)
print(nfiles)
for index, file in enumerate(files):
  file = dir + '/' + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = torch.load(file)
print("training data shape before reshape", trainingData.shape)

# Reshape data and push to device  
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))
trainingData= torch.tensor(trainingData).float().to(device)

print(f'training data is now loaded on {trainingData.device}')
print(trainingData.dtype)

#Setup 

dir = 'Data/clusterResults'
stats_dict = {}
name = "NZ_NO_SUB"
for i in range(2,10):
    # K-means
    kmeans = KMeans(i, 1000, distance='euclidean')
    labels, centers = kmeans.fit_predict(trainingData)


    torch.save(labels, f'{out_dir}/gpuKmeansSynth_Dt_{type}_k={i}_labels')
    torch.save(centers, f'{out_dir}/gpuKmeansSynth_Dt_{type}_k={i}_centers')
    stats = clusters.evaluate_cluster(trainingData, data_labels)
    data_labels = np.reshape(data_labels, (nChannels, nSamples, -1))
    stats_dict[f'cluster{i}'] = stats
    print(f'stats for {i} are {stats}')
