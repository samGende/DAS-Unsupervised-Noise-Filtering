import torch
import numpy as np 
import os
from kmeans_gpu import KMeans
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


dir = './Data/CWT_4min/paper_cwt_noSS-NZ'
out_dir ='./Data/clusterResults/'
run_name = 'gpukmeans_nz_dt_noss_sc'

sample = torch.tensor(np.load(f'{dir}/cwt_2023p152354.npy'))
sample = sample.to(device)
print(f'samples device is {sample.device}')
files = os.listdir(dir)
files.sort()
files = files[:20]
print(files)
print(f'{len(files)} files in directory')


#info for loading files
#samples_per_subsample = 25
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

#subsample for when data is to large
if(trainingData.shape[0] >= 70000000):
    print('subsampling data')
    trainingData = trainingData[::20,:]

trainingData= torch.tensor(trainingData).float().to(device)
print(f'training data is now loaded on {trainingData.device}')


#scale and center data
means = trainingData.mean(dim=0)
print('Means are calculated')
trainingData = trainingData - means
stds = torch.std(trainingData, dim=0, correction=0 )
print('stds are calculated')
trainingData = trainingData / stds
torch.save(means, f'{out_dir}/{run_name}_means.pt')
torch.save(stds, f'{out_dir}/{run_name}_stds.pt')

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

torch.save(labels, f'{out_dir}/{run_name}_labels')
torch.save(centers, f'{out_dir}/{run_name}_centers')