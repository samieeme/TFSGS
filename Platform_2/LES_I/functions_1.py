# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:24:28 2019

@author: samieeme
"""
import numpy as np
from mkl_fft import fftn, ifftn
from mkl import set_num_threads, get_max_threads

set_num_threads(4)

def remain (a,b,v):
    c = a%b
    if c == 0: c = b
    return c

def deriv_x(Nnod,V):
    Vhat = fftn(V)
    divhat = np.zeros((Nnod,Nnod,Nnod),dtype = complex)
    kx = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:,:] = complex(0,i1)
        else:
           kx[i1,:,:] = complex(0,i1-Nnod)   
    
    divhat = kx * Vhat
    diverx_V = np.real(ifftn(divhat))
    return diverx_V

def deriv_y(Nnod,V):
    Vhat = fftn(V)
    divhat = np.zeros((Nnod,Nnod,Nnod),dtype = complex)
    ky = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
           ky[:,i2,:] = complex(0,i2)
        else:
           ky[:,i2,:] = complex(0,i2-Nnod)  
    
    divhat = ky * Vhat
    divery_V = np.real(ifftn(divhat))
    return divery_V

def deriv_z(Nnod,V):
    Vhat = fftn(V)
    divhat = np.zeros((Nnod,Nnod,Nnod),dtype = complex)
    kz = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    for i3 in range(Nnod):
        if i3 <= (Nnod-1)/2:
           kz[:,:,i3] = complex(0,i3)
        else:
           kz[:,:,i3] = complex(0,i3-Nnod)   
    
    divhat = kz * Vhat
    diverz_V = np.real(ifftn(divhat))
    return diverz_V

def div(v1,v2,v3,Nnod):
    div1 = deriv_x(Nnod,v1)
    div2 = deriv_y(Nnod,v2)
    div3 = deriv_z(Nnod,v3)
    diver_V = div1[:] + div2[:] + div3[:]
    return diver_V


def Reduce_period(V,N):
    Vbar = V.reshape(N,N,N)
#    Vbar2 = Vbar1[range(0,N-1),:,:]
#    Vbar3 = Vbar2[:,range(0,N-1),:]
#    Vbar = np.transpose(Vbar3[:,:,range(0,N-1)])
    return Vbar 


def Reduce_period_JHU(V,N):
    Vbar1 = V.reshape(N,N,N)
    Vbar2 = Vbar1[range(0,N-1),:,:]
    Vbar3 = Vbar2[:,range(0,N-1),:]
    Vbar = np.transpose(Vbar3[:,:,range(0,N-1)])
    return Vbar 

def get_TwoPointCorr(a,b,res,ax):
    Lnn = np.array([])
    if ax == 0:
        for r in range(0,res):
            aa  = np.concatenate((a[r:res,:,:],a[0:r,:,:]),axis=0)
            Lnn = -np.append(Lnn,np.mean(b*aa))
    elif ax == 1:
        for r in range(0,res):
            aa = np.concatenate((a[:,r:res,:],a[:,0:r,:]),axis=1)
            Lnn = -np.append(Lnn,np.mean(aa*b))
    else:
        for r in range(0,res):
            aa = np.concatenate((a[:,:,r:res],a[:,:,0:r]),axis=2)
            Lnn = -np.append(Lnn,np.mean(aa*b))
    
#    for i in range(0,res):
#        if Lnn[i+1] < 0:
#            break
#     #   im = i
#    
#    Lnn =  Lnn[0:i]
    return Lnn
