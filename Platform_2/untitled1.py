#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:47:39 2020

@author: zayern
"""
import numpy as np
from mkl_fft import fftn, ifftn
from mkl import set_num_threads, get_max_threads
from numba import jit, prange
Nnod=1000


def Km(Nnod):
    kz = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    ky = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    kx = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    for i3 in range(Nnod):
        if i3 <= (Nnod-1)/2:
            kz[:,:,i3] = complex(0,i3)
        else:
            kz[:,:,i3] = complex(0,i3-Nnod) 
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
            ky[:,i2,:] = complex(0,i2)
        else:
            ky[:,i2,:] = complex(0,i2-Nnod)      
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
            kx[i1,:,:] = complex(0,i1)
        else:
            kx[i1,:,:] = complex(0,i1-Nnod)
    K = kx**2+ky**2+kz**2 
    K[0,0,0]=0.0001
    return K


K = Km(Nnod)