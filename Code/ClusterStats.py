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
import random 


from params_training import *




outfileList = []

ndays = 1
outfileListFile = []
files = []

transform_dir = ("./Data/CWT_NZ_NOSUB")


files = os.listdir(transform_dir)
random.seed(10)
random.shuffle(files)
files=files[:10]
print(files)


nfiles = len(files)
print(nfiles)
print(files[-1])
sample = np.load(transform_dir + '/' + files[-1])
print(sample.shape)

n_features = sample.shape[2]
sps = 50
samplingRate = 1
secondsPerWindowOffset = 240
nChannels = sample.shape[0]
nSamples = sample.shape[1]
print(nSamples)

trainingData = np.empty((nChannels, nSamples * nfiles, n_features), dtype=np.float64)
print(trainingData.shape)
print(nfiles)
for index, file in enumerate(files):
  file = transform_dir + '/' + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = np.load(file)
print("training data shape before reshape", trainingData.shape)
# Clustering
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))
print(trainingData.shape)

dir = 'Data/clusterResults'
stats_dict = {}
name = "NZ_NO_SUB"
for i in range(2,10):
    # K-means
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(trainingData)
    
    data_labels = kmeans.labels_
    print(trainingData.shape)
    print(data_labels.shape)

    stats = clusters.evaluate_cluster(trainingData, data_labels)
    data_labels = np.reshape(data_labels, (nChannels, nSamples, -1))
    np.save(f'{dir}/{name}cluster_labels_k={i}', data_labels)
    np.save(f'{dir}/{name}cluster_centers_k={i}', kmeans.cluster_centers_)
    stats_dict[f'cluster{i}'] = stats
    print(f'stats for {i} are {stats}')
joblib.dump(stats_dict, f'{dir}/{name}cluster_stats.pkl')