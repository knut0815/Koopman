#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:52:10 2021

@author: mac
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

## set random seed: 
np.random.seed(137)
torch.manual_seed(137)
    
# hyperparameters
input_size = 3
eigen_dim = 100 ## dimension of the eigenspace
hidden_size = 100

class Encoder:
    ## We are using the theory of random projections here, specifically 
    ## the Johnson-Lindenstrauss lemma. 

    def __init__(self):
        self.encode_ = np.random.normal(size=(eigen_dim,3))
        self.decode_ = np.linalg.solve(self.encode_.T.dot(self.encode_), self.encode_.T)

    def encode(self,x):
        return torch.matmul(torch.from_numpy(self.encode_).float(),x)
    
    def decode(self,x):
        return torch.matmul(torch.from_numpy(self.decode_).float(),x)
            
decoder = Encoder()
            
## data path: 
data_path = '/Users/mac/Desktop/Koopman/lorenz/data/'

### create dataframes: 
train_data = pd.read_csv(data_path + 'train.csv', sep=",",index_col=0)
test_data = pd.read_csv(data_path + 'test.csv', sep=",",index_col=0)

### instantiate tensors for training and test data: 
x_train = torch.FloatTensor(train_data.values)
x_test = torch.FloatTensor(test_data.values)

## use a deep network to approximate the eigenfunctions of a Koopman operator:
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

def K2(x,horizon):
    ## iterated composition of the Koopman operator: 
    
    temp_ = x
    
    for i in range(horizon):
    
        next_ = koopman(temp_)
        
        temp_ = next_
    
    return temp_


optimizer = torch.optim.Adam(koopman.parameters(), lr=1e-3, weight_decay=1e-5)

loss_fn = nn.MSELoss()

loss_log = []

## training parameters: 
period = 4000 ## the trajectory length
epochs = 5
batch_size = 128
learning_rate = 0.001
horizon = 5
iters= 0

## perform gradient clipping: 
clipping_value = 1 

def loss(x,iters,horizon):
    
    ## initial loss:
    x_mini, x_mini_ = x[iters:iters+batch_size], x[iters+1:iters+batch_size+1]
                
    x_in = Variable(x_mini)
    x_out = Variable(x_mini_)
                
    prediction = K2(decoder.encode(x_in.T).T,1)
    
    net_out = decoder.decode(prediction.T)
    
    loss = 0.1*loss_fn(net_out, x_out.T) + loss_fn(prediction,decoder.encode(x_out.T).T)
    
    for h in range(2,horizon):
        
        x_mini, x_mini_ = x[iters:iters+batch_size], x[iters+h:iters+batch_size+h]
                
        x_in = Variable(x_mini)
        x_out = Variable(x_mini_)
                
        prediction = K2(decoder.encode(x_in.T).T,h)
        
        net_out = decoder.decode(prediction.T)
        
        ## the prediction loss and the linearity loss: 
        loss += 0.1*loss_fn(net_out, x_out.T) + loss_fn(prediction,decoder.encode(x_out.T).T)
        
    return loss/horizon

## Train Koopman network: 
for e in range(epochs):
    
    while iters < x_train.shape[0]-batch_size-horizon:      
        
        ## break dataset into distinct trajectories: 
        if iters % (period*(iters+1)-1) == 0:
            
            iters += 1
        
        loss_ = loss(x_train,iters,horizon)
        
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(koopman.parameters(), clipping_value)
        
        loss_.backward()
        optimizer.step()
        
        if iters % 100 == 0:
            loss_log.append(loss_.data)
            
            print(iters)
            
        iters +=1
        
    ## reset iterator:
    iters = 0
            
    print('Epoch: {} - Loss: {:.6f}'.format(e, loss_.data))

## save the model: 
PATH = '/Users/mac/Desktop/Koopman/lorenz/models/random_koopman.h5'
    
torch.save(koopman.state_dict(), PATH)

## Evaluate the model after training: 
loss_log = []

## Evaluate Koopman network: 
while iters < x_test.shape[0]-batch_size-horizon:      
    
    ## break dataset into distinct trajectories: 
    if iters % (period*(iters+1)-1) == 0:
        
        iters += 1
    
    loss_ = loss(x_test,iters,horizon)
    
    if iters % 100 == 0:
        loss_log.append(loss_.data)
        
        print(loss_.data)
        
    iters +=1
    
## Try using the Koopman Operator to interpolate missing data: 
def future_state(y,horizon): 
    
    z = torch.from_numpy(y).float()
    
    forward = K2(decoder_.encode(z.T).T,horizon)
    
    z_ = decoder_.decode(forward)
    
    return z_.cpu().detach().numpy()


## use a random initial state: 
state_0 = np.random.rand(3)

## numerical integration of exact model: 
traj = odeint(f, state_0, t)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(traj[:,0], traj[:,1], traj[:,1])
plt.draw()
plt.show()

traj_ = np.array(traj)

## Interpolating missing values using the Koopman operator: 
for i in range(len(traj_)-5):
    
    if np.random.rand() > 0.5:
        
        traj_[i+2] = future_state(traj_[i],2)
        
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(traj_[:,0], traj_[:,1], traj_[:,1])
plt.draw()
plt.show()

mean_squared_error = np.square(np.subtract(traj, traj_)).mean()
