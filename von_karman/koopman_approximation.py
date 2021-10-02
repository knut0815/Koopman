#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:48:58 2021

@author: mac
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## set random seed: 
np.random.seed(137)
torch.manual_seed(137)
    
# hyperparameters
input_size = 3
eigen_dim = 15 ## dimension of the eigenspace
hidden_size = 100

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

            
## data path: 
data_path = '/Users/mac/Desktop/Koopman/von_karman/data/'

### create dataframes: 
train_data = pd.read_csv(data_path + 'train.csv', sep=",",index_col=0)

### instantiate tensors for training and test data: 
x_train = torch.FloatTensor(train_data.values)

## use a deep network to approximate the eigenfunctions of a Koopman operator:

class Koopman(nn.Module):
    ## this particular network includes a residual layer: 
    
    def __init__(self):
        super(Koopman, self).__init__()
        self.l1 = nn.Linear(eigen_dim, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, eigen_dim)
                
    def forward(self, x):        
        x1 = self.l1(x)
        x2 = self.relu(x1)
        
        x3 = self.l2(x2)
        x4 = self.relu(x3)
                
        x5 = self.l3(x4)
        
        return x5 + x
    
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
batch_size = 100
horizon = 10
iters= 0

## perform gradient clipping: 
clipping_value = 1 

def inf_norm(x,x_hat):
    
    abs_ = torch.abs(x-x_hat)
    
    max_ = torch.max(abs_,1)[0]
    
    return torch.mean(max_)

def loss(x,iters,horizon):
    
    ## initial loss:
    x_mini, x_mini_ = x[iters:iters+batch_size], x[iters+1:iters+batch_size+1]
                
    x_in = Variable(x_mini)
    x_out = Variable(x_mini_)
                
    prediction = K2(encoder.encode(x_in),1)
    
    net_out = encoder.decode(prediction)
    
    loss = 0.1*loss_fn(net_out, x_out) + 0.001*inf_norm(prediction,encoder.encode(x_out)) + loss_fn(prediction,encoder.encode(x_out))
    
    for h in range(2,horizon):
        
        x_mini, x_mini_ = x[iters:iters+batch_size], x[iters+h:iters+batch_size+h]
                
        x_in = Variable(x_mini)
        x_out = Variable(x_mini_)
                
        prediction = K2(encoder.encode(x_in),h)
        
        net_out = encoder.decode(prediction)
        
        ## the prediction loss, the linearity loss and the infinity norm loss(torch.max): 
        loss += 0.1*loss_fn(net_out, x_out) + 0.001*inf_norm(prediction,encoder.encode(x_out)) + loss_fn(prediction,encoder.encode(x_out)) 
        
    return loss/horizon

def future_state(y,horizon): 
    
    z = torch.from_numpy(y).float()
    
    forward = K2(encoder.encode(z),horizon)
    
    z_ = encoder.decode(forward)
    
    return z_.cpu().detach().numpy()

def eyeball_test():
    
    y_hat = np.zeros((3,1000))
    
    x0, y0, z0 = 0.2*(np.random.rand()-0.5),0.2*(np.random.rand()-0.5), 0.3 
    
    state0 = np.array([x0, y0, z0])
    
    y_hat[:,0] = state0
    
    for i in range(1,999):
                
        y_hat[:,i+1] = future_state(y_hat[:,i],5)
    
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(y_hat[0], y_hat[1], y_hat[2])
    plt.draw()
    plt.show()

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
            
            print(loss_.data)
            
            eyeball_test()
            
        iters +=1
        
    ## reset iterator:
    iters = 0
            
    print('Epoch: {} - Loss: {:.6f}'.format(e, loss_.data))

## save the model: 
PATH = '/Users/mac/Desktop/Koopman/von_karman/models/random_koopman.h5'
    
torch.save(koopman.state_dict(), PATH)

## Evaluate the model after training: 
loss_log = []

### create dataframes: 
test_data = pd.read_csv(data_path + 'train.csv', sep=",",index_col=0)

### instantiate tensors for training and test data: 
x_test = torch.FloatTensor(test_data.values)

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