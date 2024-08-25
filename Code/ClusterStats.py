import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from kmeans_gpu import KMeans
from utilities import clusters
import pickle 

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

type = 'paper_cwt_noSS-NZ'
dir = f'./Data/CWT_4min/{type}'
#dir = f'./Data/synthetic-DAS/strongEq/CWT/paper_cwt_noSS/train'
out_dir ='./Data/clusterResults/'

us_numpy = True
    
files = os.listdir(dir)
files.sort()
files = files[:25]
print(files)
print(f'{len(files)} files in directory')
if us_numpy:
    sample = torch.tensor(np.load(f'{dir}/{files[0]}'))
else:
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
  if (us_numpy):
        sample = torch.tensor(np.load(file))
  else:
        sample = torch.load(file)
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = sample
print("training data shape before reshape", trainingData.shape)

# Reshape data and push to device  
if (trainingData.shape[1] * trainingData.shape[0] > 7000000):
    trainingData = trainingData[:,::15, :]
    print(f'reduced data shape is {trainingData.shape}')
    trainingData = np.reshape(trainingData, (trainingData.shape[0] * trainingData.shape[1], -1))
else:  
    trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))

trainingData= torch.tensor(trainingData).float().to(device)

#scale and center data
means = trainingData.mean(dim=0)
print('Means are calculated')
trainingData = trainingData - means
stds = torch.std(trainingData, dim=0, correction=0 )
print('stds are calculated')
trainingData = trainingData / stds
torch.save(means, f'{out_dir}/{type}_means.pt')
torch.save(stds, f'{out_dir}/{type}_stds.pt')


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


    torch.save(labels, f'{out_dir}/gpuKmeans_{type}_sc_k={i}_labels')
    torch.save(centers, f'{out_dir}/gpuKmeans_{type}_sc_k={i}_centers')
    stats = clusters.evaluate_cluster(trainingData.detach().cpu().numpy(), labels.detach().cpu().numpy())
    stats_dict[f'cluster{i}'] = stats
    print(f'stats for {i} are {stats}')
with open(f'{type}stats_dict.pkl', 'wb') as pickle_file:
    pickle.dump(stats_dict, pickle_file)
