import torch
import numpy as np 
import os
from kmeans_gpu import KMeans
from tqdm import tqdm
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(2)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)


dir = './Data/CWT_4min/supports_inverse'
out_dir ='./Data/clusterResults/supports_inverse'

sample = torch.tensor(np.load(f'{dir}/cwt_20160905_01:51:54.npy'))
sample = sample.to(device)
print(f'samples device is {sample.device}')
files = os.listdir(dir)
files.sort()
print(f'{len(files)} files in directory')
#last_file_index = 2160
#first_file_index = 1440
#print(files[first_file_index])

#files = files[first_file_index:last_file_index:2]
files =files[::2]
files = files[:-1]
#info for loading files

samples_per_subsample =25
nChannels = sample.shape[0]
nSamples = (sample.shape[1]//samples_per_subsample )
n_features = sample.shape[2]
nfiles = len(files)
#load files 
trainingData = np.empty((nChannels, nSamples * nfiles, n_features), dtype=np.float64)
print(trainingData.shape)
print(nfiles)
print(nSamples)
for index, file in tqdm(enumerate(files)):
  file = dir + '/' + file
  sampleChunk =  np.reshape(np.load(file), (nChannels, -1, 25, n_features))
  sampleChunk = np.mean(sampleChunk, axis =2)
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] =sampleChunk
print("training data shape before reshape", trainingData.shape)

# Reshape data and push to device  
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))
trainingData= torch.tensor(trainingData).float().to(device)

print(f'training data is now loaded on {trainingData.device}')
print(trainingData.dtype)
#Setup kmeans 

n_clusters = 4

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

torch.save(labels, f'{out_dir}/gpuKmeansNoDt_SS_labels')
torch.save(centers, f'{out_dir}/gpuKmeansNoDt_SS_centers')