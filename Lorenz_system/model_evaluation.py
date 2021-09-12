#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:46:59 2021

@author: mac
"""

## try random initial conditions: 
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

koopman.load_state_dict(torch.load('/Users/mac/Desktop/Koopman/lorenz/models/random_koopman.h5'))
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
                
encoder.load_state_dict(torch.load('/Users/mac/Desktop/Koopman/lorenz/models/encoder.h5'))

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

## Begin interpolation or missing values test: 

## parameters of Lorenz system: 
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(t, state):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

state0 = np.random.rand(3)

## time 
t = np.arange(0.0, 40.0, 0.01)

## numerical integration of exact model: 
X = solve_ivp(f, [0.0,40.0], state0,t_eval=t,method='LSODA')

x_train = X.y

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(x_train[0], x_train[1], x_train[2])
plt.draw()
plt.show()

x_train_ = np.array(x_train)

## Interpolating missing values using the Koopman operator: 
for i in range(len(x_train_[0])-5):
    
    if np.random.rand() > 0.5:
        
        x_train_[:,i+2] = future_state(x_train[:,i],2)
        
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(x_train_[0], x_train_[1], x_train_[2])
plt.draw()
plt.show()

mean_squared_error = np.square(np.subtract(x_train, x_train_)).mean()