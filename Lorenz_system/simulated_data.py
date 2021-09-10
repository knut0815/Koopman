#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:30:00 2021

@author: mac
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint

## Build training set: 

## parameters of Lorenz system: 
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

## time 
t = np.arange(0.0, 40.0, 0.01)

train_data = []

for i in range(5):
    
    state0 = np.random.rand(3)
    
    x_train = odeint(f, state0, t)
    
    train_data.append(x_train)
    
x_train = np.vstack(train_data)

pd.DataFrame(x_train).to_csv("/Users/mac/Desktop/Koopman/lorenz/data/train.csv")

## do the same for test data: 
test_data = []

for i in range(5):
    
    state0 = np.random.rand(3)
    
    x_test = odeint(f, state0, t)
    
    test_data.append(x_test)
    
x_test = np.vstack(test_data)

pd.DataFrame(x_test).to_csv("/Users/mac/Desktop/Koopman/lorenz/data/test.csv")