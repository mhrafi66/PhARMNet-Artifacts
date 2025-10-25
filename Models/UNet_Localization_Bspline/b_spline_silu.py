import torch
import torch.nn as nn
import numpy as np

import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
import pandas as pd
from sympy.printing import latex
from sympy import *
import sympy
import yaml
from b_spline_utils import *

class NonLinearBSplineSiLU(nn.Module):
    def __init__(self,
                 seed = 0,
                 num=5, #number of control points
                 k=3, #order of the spline
                 grid_range=[-1, 1] #get the grid or input within this range
    ):
        super(NonLinearBSplineSiLU, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.seed = seed
        self.num=num #number of control points
        self.k=k #order of the spline
        self.grid_range=grid_range

        self.in_dim=1 #nodes in the input layer
        self.out_dim=1 #nodes in the output layer
        self.width = [1, 1]
        self.depth = len(self.width) - 1


        self.noise_scale=0.5 #dunno
        self.scale_base_mu=0.0
        self.scale_base_sigma=1.0
        self.scale_sp=1.0
        self.base_fun=torch.nn.SiLU()
        self.grid_eps=0.02
        self.sp_trainable=True #spline coefficient
        self.sb_trainable=True #base function coefficient
        self.save_plot_data = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sparse_init=False

        grid = torch.linspace(self.grid_range[0], self.grid_range[1], steps=self.num + 1)[None,:].expand(self.in_dim, self.num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * self.noise_scale / self.num

        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))

        self.mask = torch.nn.Parameter(torch.ones(self.in_dim, self.out_dim)).requires_grad_(False)

        self.scale_base = torch.nn.Parameter(self.scale_base_mu * 1 / np.sqrt(self.in_dim) + \
                         self.scale_base_sigma * (torch.rand(self.in_dim, self.out_dim)*2-1) * 1/np.sqrt(self.in_dim)).requires_grad_(self.sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(self.in_dim, self.out_dim) * self.scale_sp * self.mask).requires_grad_(self.sp_trainable)  # make scale trainable
        self.input_id = torch.arange(self.in_dim)
        self.to(self.device)




    def to(self, device):
        super(NonLinearBSplineSiLU, self).to(device)
        self.device = device
        return self

    def forward(self, x, y_th=10):
        '''
        forward pass
        Args:
        -----
            x : 2D torch.tensor
                inputs
            singularity_avoiding : bool
                whether to avoid singularity for the symbolic branch
            y_th : float
                the threshold for singularity

        Returns:
        --------
            None
        '''
        # print(" In spline function, shape of x = ", x.shape)
        # print(" In spline function, input id = ", self.input_id)
        x = x[:,self.input_id.long()]
        assert x.shape[1] == self.width[0]

        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)


        base = self.base_fun(x) # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)

        postspline = y.clone().permute(0,2,1)

        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y


        postacts = y.clone().permute(0,2,1)

        y = torch.sum(y, dim=1)

        x_numerical = y
        preacts = preacts
        postacts_numerical = postacts
        postspline = postspline


        return x_numerical