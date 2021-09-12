#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 13:00:21 2021

@author: mac
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd

# hyperparameters
input_size = 3
output_size = 15
hidden_size = 100

epochs = 500
batch_size = 100
learning_rate = 0.001

## define the encoder and the decoder: 

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.l4 = nn.Linear(output_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x
    
    def backward(self,x):
        x = self.l4(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l5(x)
        return x
    
net = Network()

## load data: 

## set random seed: 
torch.manual_seed(137)

## data path: 
path = '/Users/mac/Desktop/Koopman/lorenz/data/'

### check prime_encodings.py for the method used for data generation:
train_data = pd.read_csv(path + 'train.csv', sep=",",index_col=0)


x = torch.FloatTensor(train_data.values)

## define the loss function and training algorithm:     
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

## the weight decay was essential
loss_fn = nn.MSELoss()

loss_log = []

## perform gradient clipping: 
clipping_value = 1 

def encoder(x):
    """Specify the deterministic encoder network."""
    
    encode = net.forward(x)
    
    decode = net.backward(encode)
            
    return decode

## train encoder network: 
for e in range(epochs):
    for i in range(0, x.shape[0], batch_size):
        x_mini = x[i:i + batch_size]
        
        x_var = Variable(x_mini)
        
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm(net.parameters(), clipping_value)
        
        net_out = encoder(x_var)
        
        loss = loss_fn(net_out, x_var)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            loss_log.append(loss.data)
        
    print('Epoch: {} - Loss: {:.6f}'.format(e, loss.data))
    

## save the model: 

PATH = '/Users/mac/Desktop/Koopman/lorenz/models/encoder.h5'
    
torch.save(net.state_dict(), PATH)
    

