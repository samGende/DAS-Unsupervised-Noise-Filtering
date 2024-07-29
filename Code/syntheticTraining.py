import os
os.environ['MKL_NUM_THREADS'] = "4"
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
import numpy as np
import datetime as dt
import struct
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AggloCluster
from utilities import clusters
import joblib
import torch 


from params_training import *




outfileList = []

ndays = 1
outfileListFile = []
files = []

transform_dir = ("Data/synthetic-DAS/train-syntheticDAS/CWT-edDAS")


files = os.listdir(transform_dir)
files.sort()
files=files[:100]


nfiles = len(files)
print(files[-1])
sample = torch.load(transform_dir + '/' + files[-1])
print(sample.shape)

n_features = sample.shape[2]
sps = 50
samplingRate = 50
secondsPerWindowOffset = sample.shape[1]
nChannels = sample.shape[0]
nSamples = secondsPerWindowOffset * int(sps / samplingRate)
nSamples = nSamples //2
print(nSamples)

trainingData = np.empty((nChannels, nSamples * nfiles, n_features), dtype=np.float64)
print(nfiles)

dir = 'Data/synthetic-DAS/train-syntheticDAS/ClusteringResults'

for index, file in enumerate(files):
  file = transform_dir + '/' + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = np.array(torch.load(file))[:,::2, :]
print("training data shape before reshape", trainingData.shape)
# Clustering
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))

scaler = StandardScaler(copy=False).fit(trainingData)
trainingData = scaler.transform(trainingData)

print(trainingData.shape)

kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
kmeans.fit(trainingData)
    
data_labels = kmeans.labels_
  
np.save(f'{dir}/Trainingcluster_labels_k=3', data_labels)
np.save(f'{dir}/Trainingcluster_centers_k=3', kmeans.cluster_centers_)