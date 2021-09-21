#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:08:41 2021

@author: mac
"""

### Aidan Rocke. analytic_min-max_operators(2020).GitHub repository, 
### https://github.com/AidanRocke/analytic_min-max_operators

import torch

def infinity_norm(X,q=100):
    """An analytic approximation of the infinity norm whose partial derivative 
    is the softmax and where q is the degree of the approximation.""" 
        
    mu, sigma = torch.mean(X), torch.std(X)
    
    ## rescale vector so it has zero mean and unit variance:
    Z_score = (X-mu)/sigma
    
    exp_sum = torch.sum(torch.exp(Z_score*q))
    
    log_ = torch.log(exp_sum)/q

    return (log_*sigma)+mu