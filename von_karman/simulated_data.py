#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:44:54 2021

@author: mac
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

## parameters of Von Karman system: 
mu = 0.1
omega = 1.0
lambda_ = 10.0
A = -0.1

def f(t,state):
    x, y, z = state  # Unpack the state vector
    return mu*x-omega*y+A*x*z, omega*x+mu*y+A*y*z, -lambda_*(z-x*x-y*y)  # Derivatives

## time 
t = np.arange(0.0, 40.0, 0.01)

train_data = []

for i in range(5):        
    
    x0, y0, z0 = 0.2*(np.random.rand()-0.5),0.2*(np.random.rand()-0.5), 0.3 

    state0 = np.array([x0, y0, z0])
    
    X = solve_ivp(f, [0.0,100.0], state0,t_eval=t,method='LSODA')
    
    train_data.append(X.y.T)
    
x_train = np.vstack(train_data)

pd.DataFrame(x_train).to_csv("/Users/mac/Desktop/Koopman/von_karman/data/train.csv")


## analyse the evolution of the eigenspace: 
import numpy as np
from sklearn.decomposition import PCA

rand_matrix = np.random.rand(3,100)

out = np.matmul(x_train,rand_matrix)

pca = PCA(n_components=3, svd_solver='full')

eigenspace = pca.fit(out)

eigenvectors = np.matmul(eigenspace.components_,out.T)

fig = plt.figure()
ax = fig.gca(projection="3d",fc='orange')
ax.plot(eigenvectors[0], eigenvectors[0], eigenvectors[2])
plt.draw()
plt.show()

plt.plot(eigenvectors[0], eigenvectors[2],'r')

fig = plt.figure()
ax = fig.gca(projection="3d",fc='orange')
ax.plot(X.y[0], X.y[0], X.y[2])
plt.draw()
plt.show()


