from utilities.models import Autoencoder_v1
import torch
import torch.nn as nn 
import numpy as np 
import os


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(2)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)


dir = './Data/CWT_4min/CWTNZ_Dt_SS'

sample = torch.tensor(np.load(f'{dir}/cwt_2023p152354.npy'))
sample = sample.to(device)
print(f'samples device is {sample.device}')
files = os.listdir(dir)
files.sort()
print(files)
print(f'{len(files)} files in directory')
files = files[:20]


#info for loading files
samples_per_subsample = 25
nChannels = sample.shape[0]
nSamples = sample.shape[1]//samples_per_subsample
n_features = sample.shape[2]
nfiles = len(files)

#load files 
trainingData = np.empty((nChannels, nSamples * nfiles, n_features), dtype=np.float64)
print(trainingData.shape)
print(nfiles)
assert(0)
for index, file in enumerate(files):
  file = dir + '/' + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = np.load(file)
print("training data shape before reshape", trainingData.shape)

# Reshape data and push to device  
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))
trainingData= torch.tensor(trainingData).float().to(device)

print(f'training data is now loaded on {trainingData.device}')
print(trainingData.dtype)

AE = Autoencoder_v1(10)

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(AE.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 50
losses = np.zeros(n_epochs)

batch_size = 1000

batched_data = torch.reshape(trainingData, (-1, batch_size, trainingData[1]))

#move model and data to device 
AE.to(device)
batched_data.to(device)

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data in batched_data:
        features = torch.tensor(data, dtype=torch.float32)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = AE(features)
        # calculate the loss
        loss = criterion(outputs, features)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*features.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(batched_data)
    losses[epoch-1] = train_loss
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    
AE.to('cpu')
torch.save(AE.nn, 'NZ_Dt_SS_AEv1.nn')
np.save(losses, 'NZ_Dt_SS_AEv1_losses.npy')