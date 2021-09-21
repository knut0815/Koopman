#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:02:07 2021

@author: mac
"""

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

## Begin interpolation or missing values test: 

## parameters of Von Karman system: 
mu = 0.1
omega = 1.0
lambda_ = 10.0
A = -0.1

x0, y0, z0 = 0.2*(np.random.rand()-0.5),0.2*(np.random.rand()-0.5), 0.3

state0 = np.array([x0, y0, z0])

def f(t,state):
    x, y, z = state  # Unpack the state vector
    return mu*x-omega*y+A*x*z, omega*x+mu*y+A*y*z, -lambda_*(z-x*x-y*y)  # Derivatives

## time 
t = np.arange(0.0, 40.0, 0.01)

train_data = []
    
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
        
        x_train_[:,i+5] = future_state(x_train[:,i],5)
        
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(x_train_[0], x_train_[1], x_train_[2])
plt.draw()
plt.show()

mean_squared_error = np.square(np.subtract(x_train, x_train_)).mean()

