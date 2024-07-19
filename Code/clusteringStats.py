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



from params_training import *




outfileList = []
nChannels = 286
ndays = 1
outfileListFile = []
files = []
n_features = 60
sps = 50
samplingRate = 25
secondsPerWindowOffset = 150
nSamples = secondsPerWindowOffset * int(sps / samplingRate)
transform_dir = ("CWT")


files = os.listdir(transform_dir)
files.sort()
files = files[:721] 
print(files)


nfiles = len(files)
print(files[-1])


trainingData = np.empty((nChannels, nSamples * nfiles, n_features), dtype=np.float64)
print(trainingData.shape)


for index, file in enumerate(files):
  file = transform_dir + '/' + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = np.load(file)
print("training data shape before reshape", trainingData.shape)
# Clustering
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))

dir = 'singleDayClusters'

data_labels = np.load(f'./{dir}/kmeansClusterLabels.npz')

stats = clusters.evaluate_cluster(trainingData, data_labels)

print(stats)