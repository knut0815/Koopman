#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:16:43 2021

@author: mac
"""

import numpy as np

def entropy_vector(matrix):
    """Calculate entropy vector from correlation matrix""" 
    
    ### avoid numerical instabilities: 
    eps = np.finfo(float).eps
    
    R = np.corrcoef(matrix)

    rho = R/np.shape(R)[0]
    
    eigen = np.linalg.eig(rho)
    
    P = eigen[0]
    
    bits = np.multiply(P,-1.0*np.log(P+eps))
    
    return bits.real

def informative_dimensions(matrix,alpha):
    
    N = len(matrix)
    
    bits = entropy_vector(matrix)
    
    sorted_indices = np.argsort(-bits)[:N]
    
    iter_, sum_, max_ = 0, 0, np.sum(bits)
    
    for i in sorted_indices: 
        
        if sum_/max_ < alpha:
            
            sum_ += bits[i]
            
            iter_ += 1
            
        else: 
                        
            break 
        
    return sorted_indices[:iter_-1], np.sum(bits[sorted_indices[:iter_-1]])/np.sum(bits)

## load model: 
import torch
import torch.nn as nn

## set random seed: 
np.random.seed(137)
torch.manual_seed(137)

# hyperparameters
input_size = 3
eigen_dim = 15 ## dimension of the eigenspace
hidden_size = 100

class Koopman(nn.Module):
    
    def __init__(self):
        super(Koopman, self).__init__()
        self.l1 = nn.Linear(eigen_dim, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, eigen_dim)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x
    
koopman = Koopman()

koopman.load_state_dict(torch.load('/Users/mac/Desktop/Koopman/von_karman/models/random_koopman.h5'))
koopman.eval()

class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, eigen_dim)
        self.l4 = nn.Linear(eigen_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, input_size)
        
    def encode(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x
    
    def decode(self,x):
        x = self.l4(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l5(x)
        return x
    
encoder = Encoder()
                
encoder.load_state_dict(torch.load('/Users/mac/Desktop/Koopman/von_karman/models/encoder.h5'))

encoder.eval()

def K2(x,horizon):
    ## iterated composition of the Koopman operator: 
    
    temp_ = x
    
    for i in range(horizon):
    
        next_ = koopman(temp_)
        
        temp_ = next_
    
    return temp_

def future_state(y,horizon): 
    
    z = torch.from_numpy(y).float()
    
    forward = K2(encoder.encode(z),horizon)
    
    z_ = encoder.decode(forward)
    
    return z_.cpu().detach().numpy()
    

### Calculate information dimension: 
N = 4000
    
Z = np.zeros((15,N))

X = np.zeros((3,N))

X[:,0] = np.random.rand(3)
    
for i in range(1,N-5):
    
    X[:,i] = future_state(X[:,i-1],5)
    
    z = torch.from_numpy(X[:,i]).float()
    
    forward = K2(encoder.encode(z),5)
    
    Z[:,i] = forward.cpu().detach().numpy()

## figure out which dimensions contain ~95% of the information: 
indices, ratio = informative_dimensions(Z,.95)    

### The answer is approximately four dimensions. 