from utilities.models import Autoencoder_v1
import torch
import torch.nn as nn 
import numpy as np 
import os
from kmeans_gpu import KMeans


if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(2)} is available')
else:
    print("No GPU available. Training will run on CPU.")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)




dir = './Data/CWT_4min/CWTNZ_Dt_SS'
out_dir = './Data/CWT_4min/encoded_CWT_Dt_SS'

sample = torch.tensor(np.load(f'{dir}/cwt_2023p152354.npy'))
sample = sample.to(device)
print(f'samples device is {sample.device}')
files = os.listdir(dir)
files.sort()
print(files)
print(f'{len(files)} files in directory')
files = files[:25]


#info for loading files
samples_per_subsample = 25
nChannels = sample.shape[0]
nSamples = sample.shape[1]
n_features = sample.shape[2]
nfiles = len(files)

#load files 
trainingData = np.empty((nChannels, nSamples * nfiles, n_features), dtype=np.float64)
print(trainingData.shape)
print(nfiles)
for index, file in enumerate(files):
  file = dir + '/' + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = np.load(file)
print("training data shape before reshape", trainingData.shape)

# Reshape data and push to device  
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))
trainingData= torch.tensor(trainingData).float().to(device)

print(f'training data is now loaded on {trainingData.device}')
print(trainingData.shape)

#load trained model 
trained_AE = torch.load('./NZ_Dt_SS_AEv1.nn').to(device)

#encoded training data
encoded_training_data = trained_AE.encode(trainingData)



print(f'training data is now loaded on {encoded_training_data.device}')
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

labels, centers = kmeans.fit_predict(encoded_training_data)

print(labels.shape)
print(centers.shape)

torch.save(labels, f'{out_dir}/gpuKmeans_encoded_NZ_Dt_SS_labels')
torch.save(centers, f'{out_dir}/gpuKmeans_encoded_NZ_Dt_SS_centers')
torch.save(encoded_training_data.to('cpu'), f'{out_dir}/encoded_data.pt')
