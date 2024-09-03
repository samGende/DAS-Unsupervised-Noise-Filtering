from utilities.models import Autoencoder_v1, Autoencoder_v2, Autoencoder_v3
import torch
import torch.nn as nn 
import numpy as np 
import os
import argparse 


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(1)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description="Process model parameters.")

# Add arguments
parser.add_argument("model_name", type=str, help="Name of the model")
parser.add_argument("batch_size", type=int, help="Batch size for training")
parser.add_argument("epochs", type=int, help="Number of epochs for training")
parser.add_argument("learning_rate", type=float, help="Learning rate for training")
parser.add_argument("version", type=int, help="The AE version")

out_dir = './Data/Autoencoders'

# Parse the arguments
args = parser.parse_args()

# Access the arguments
model_name = args.model_name
batch_size = args.batch_size
n_epochs = args.epochs
lr = args.learning_rate
ae_version = args.version

# Print the values to verify
print(f"Model Name: {model_name}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {n_epochs}")


dir = './Data/CWT_4min/paper_cwt_noSScomplex-NZ'

sample = torch.tensor(np.load(f'{dir}/cwt_2023p152354.npy'))
print(sample.shape)
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
nSamples = sample.shape[1]
n_features = sample.shape[2]
nfiles = len(files)

#load files 
trainingData = np.empty((nChannels, nSamples * nfiles, n_features), dtype=np.complex64)
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
    trainingData = trainingData[::25,:]
    
trainingData= torch.tensor(trainingData).to(device)
print(f'training data shape after reshape {trainingData.shape}')
print(f'training data is now loaded on {trainingData.device}')

#scale and center data
means = trainingData.mean(dim=0)
print('Means are calculated')
trainingData = trainingData - means
stds = torch.std(trainingData, dim=0, correction=0 )
print('stds are calculated')
trainingData = trainingData / stds
torch.save(means, f'{out_dir}/{model_name}_means.pt')
torch.save(stds, f'{out_dir}/{model_name}_stds.pt')

if(ae_version == 1):
    AE = Autoencoder_v1(10, sample.shape[2])
if(ae_version == 2):
    AE = Autoencoder_v2(10, sample.shape[2])
    state_dict = torch.load(f'{out_dir}/AEv2_NZ_Normalized.nn')
    AE.load_state_dict(state_dict)
if(ae_version == 3):
    print('using complex AE')
    AE = Autoencoder_v3(10, sample.shape[2])
    print(AE)



# specify loss function

loss_real = torch.nn.MSELoss()
loss_imag = torch.nn.MSELoss()
def complex_loss(x,y):
    real = loss_real(x.real, y.real)
    imag = loss_imag(x.imag, y.imag)
    return (real + imag) / 2

# specify loss function
optimizer = torch.optim.Adam(AE.parameters(), lr=lr)

# number of epochs to train the model
losses = np.zeros(n_epochs)


batches = trainingData.shape[0] // batch_size

batched_data = torch.reshape(trainingData[:batches*batch_size], (batches, batch_size, sample.shape[2]))
print(f'batched data shape {batched_data.shape}')



#move model and data to device 
AE.to(device)
batched_data.to(device)

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    first_batch = True
    for data in batched_data:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = AE(data)
        # calculate the loss
        loss = complex_loss(outputs, data)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()
        if(first_batch):
            print('batch finished')
            first_batch = False
            
    # print avg training statistics 
    train_loss = train_loss/len(batched_data)
    losses[epoch-1] = train_loss
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    
    AE.to('cpu')
    torch.save(AE.state_dict(), f'{out_dir}/{model_name}.nn')
    AE.to(device)
    np.save(f'{out_dir}/{model_name}_losses.npy', losses)
