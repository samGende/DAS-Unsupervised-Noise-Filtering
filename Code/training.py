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


from params_training import *




srcChannelsStrings = ["35", "75"]
nChannels = endCh - startCh + 1
outfileList = []

ndays = 1
outfileListFile = outfileListFile[:-5]
files = []
n_features = 59
sps = 50
samplingRate = 25
secondsPerWindowOffset = 239
nSamples = secondsPerWindowOffset * int(sps / samplingRate)
transform_dir = ("CWT_4min/05SubSampled")


files = os.listdir(transform_dir)
files.sort()
print(files)
files = files[:2]

nfiles = len(files)

trainingData = np.empty((nChannels, nSamples * nfiles, n_features), dtype=np.float64)

for index, file in enumerate(files):
  file = transform_dir + '/' + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = np.load(file)
print("training data shape before reshape", trainingData.shape)
# Clustering
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))

print("training data shape", trainingData.shape)
assert(0)
scaler = StandardScaler(copy=False).fit(trainingData)
trainingData = scaler.transform(trainingData)
nClusters = 4
# dump the scaler
import joblib
outfileName = transform_dir + '/'+ 'scaler.pkl'
joblib.dump(scaler, outfileName)
outfileList.append(outfileName)


# K-means
kmeans = KMeans(init='k-means++', n_clusters=nClusters, n_init=10)
kmeans.fit(trainingData)
trainingLabels = np.reshape(kmeans.labels_, (nChannels, -1))
outfileName = transform_dir+ '/' + 'kmeansClusterLabels' 
np.savez(outfileName, trainingLabels)
outfileList.append(outfileName)
outfileName = transform_dir +'/' + 'kmeansClusterCenters' 
np.savez(outfileName, kmeans.cluster_centers_)
outfileList.append(outfileName)

# dump the estimator
outfileName = transform_dir + '/' + 'kmeans.pkl'
joblib.dump(kmeans, outfileName)
outfileList.append(outfileName)
"""
# Hierarchical clustering 
aggloCluster = AggloCluster(n_clusters=nClusters, affinity='euclidean', compute_full_tree='false', linkage='average')
aggloCluster.fit(trainingData)
trainingLabels = np.reshape(aggloCluster.labels_, (nChannels, -1))
outfileName = outfilePath + 'aggloClusterLabels' 
np.savez(outfileName, trainingLabels)
outfileList.append(outfileName)

# dump the estimator
outfileName = outfilePath + 'aggloCluster.pkl'
joblib.dump(aggloCluster, outfileName)
outfileList.append(outfileName)
"""
# write the list of output file names
outfileListFile = outfileListFile + '_clustering'
outFile = open(outfileListFile,'w')
for filename in outfileList:
  outFile.write(filename + '\n')
outFile.close()
