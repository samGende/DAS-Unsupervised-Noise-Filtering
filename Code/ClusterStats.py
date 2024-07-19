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


from params_training import *




outfileList = []
nChannels = 3704
ndays = 1
outfileListFile = []
files = []
n_features = 54
sps = 50
samplingRate = 25
secondsPerWindowOffset = 239
nSamples = secondsPerWindowOffset * int(sps / samplingRate)
transform_dir = ("CWT_NewZeland")


files = os.listdir(transform_dir)
files.sort()
print(files)


nfiles = len(files)
print(files[-1])


trainingData = np.empty((nChannels, nSamples * nfiles, n_features), dtype=np.float64)

sample_file = np.load(transform_dir + '/cwt_2023p152354.npy')
print(sample_file.shape)
print(nfiles)

for index, file in enumerate(files):
  file = transform_dir + '/' + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = np.load(file)
print("training data shape before reshape", trainingData.shape)
# Clustering
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))
print(trainingData.shape)

dir = 'NZClustering'
stats_dict = {}
for i in range(2,10):
    data_labels = np.load(f'./{dir}/kmeansClusterLabels{i}.npz')
    data_labels = np.squeeze(np.reshape(data_labels["arr_0"], (nChannels * nSamples * nfiles, -1)))
    print(trainingData.shape)
    print(data_labels.shape)

    stats = clusters.evaluate_cluster(trainingData, data_labels)

    stats_dict[f'cluster{i}'] = stats
    print(f'stats for {i} are {stats}')
joblib.dump(stats_dict, 'cluster_stats.pkl')