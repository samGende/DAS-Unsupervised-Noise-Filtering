import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import torch
import complexPyTorch.complexFunctions as complexFunctions
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU, NaiveComplexBatchNorm1d, ComplexDropout
from complexPyTorch.complexFunctions import complex_relu


def complex_sigmoid(inp):
    return F.sigmoid(inp.real).type(torch.complex64) + 1j * F.sigmoid(inp.imag).type(
        torch.complex64
    )


def complex_tanh(inp):
    return F.tanh(inp.real).type(torch.complex64) + 1j * F.tanh(inp.imag).type(
        torch.complex64
    )

class ComplexSigmoid(Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid(inp)
    
class ComplexTanh(Module):
    @staticmethod
    def forward(inp):
        return complex_tanh(inp)


class Autoencoder_v1(nn.Module):
    def __init__(self, encoding_dim, input_dim):
        super(Autoencoder_v1, self).__init__()
        ## encoder ##
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, encoding_dim)
        )
        ## decoder ##
        self.decoder =   self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32,input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x, dtype=torch.float32)
        # define feedforward behavior 
        # and scale the *output* layer with a sigmoid activation function
        
        # pass x into encoder
        out = F.relu(self.encoder(x))
        # pass out into decoder
        out = self.decoder(out)
        
        return out
    
    def encode(self, x ):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x, dtype=torch.float32)
        return self.encoder(x)
    
    def decode(self, x):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x, dtype=torch.float32)
        return self.decoder(x)

    
    
    
class Autoencoder_v2(nn.Module):
    def __init__(self, encoding_dim, input_dim):
        super(Autoencoder_v2, self).__init__()
        ## encoder ##
        self.encoder = nn.Sequential(
            Complex(input_dim, 32),
            nn.ReLU(),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, encoding_dim)
        )
        ## decoder ##
        self.decoder =   self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32,input_dim),
        )

    def forward(self, x):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x, dtype=torch.float32)
        # define feedforward behavior 
        # and scale the *output* layer with a sigmoid activation function
        
        # pass x into encoder
        out = F.relu(self.encoder(x))
        # pass out into decoder
        out = self.decoder(out)
        
        return out
    
    def encode(self, x ):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x, dtype=torch.float32)
        return self.encoder(x)
    
    def decode(self, x):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x, dtype=torch.float32)
        return self.decoder(x)

    
class Autoencoder_v3(nn.Module):
    def __init__(self, encoding_dim, input_dim):
        super(Autoencoder_v3, self).__init__()
        ## encoder ##
        self.encoder = nn.Sequential(
            ComplexLinear(input_dim, 32),
            NaiveComplexBatchNorm1d(32),
            ComplexReLU(),
            ComplexDropout(0.2),
            ComplexTanh(),
            ComplexLinear(32, 16),
            NaiveComplexBatchNorm1d(16),
            ComplexTanh(),
            ComplexDropout(0.2),
            ComplexLinear(16, encoding_dim)
        )
        ## decoder ##
        self.decoder =   self.decoder = nn.Sequential(
            ComplexLinear(encoding_dim, 16),
            NaiveComplexBatchNorm1d(16),
            ComplexTanh(),
            ComplexLinear(16, 32),
            NaiveComplexBatchNorm1d(32),
            ComplexTanh(),
            ComplexLinear(32,input_dim),
            ComplexSigmoid()
        )

    def forward(self, x):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x, dtype=torch.complex64)
        # define feedforward behavior 
        # and scale the *output* layer with a sigmoid activation function

        # pass x into encoder
        out = self.encoder(x)
        # pass out into decoder
        out = self.decoder(out)

        return out

    def encode(self, x ):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x, dtype=torch.complex64)
        return self.encoder(x)

    def decode(self, x):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x, dtype=torch.complex64)
        return self.decoder(x)