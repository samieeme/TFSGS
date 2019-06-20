# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:24:28 2019

@author: samieeme
"""
import numpy as np

def remain (a,b):
    c = a%b
    if c == 0: c = b
    return c

def deriv_x(Nnod,V):
    Vhat = np.fft.fftn(V)
    divhat = np.zeros((Nnod,Nnod,Nnod),dtype = complex)
    kx = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:,:] = complex(0,i1)
        else:
           kx[i1,:,:] = complex(0,i1-Nnod)   
    
    divhat = kx * Vhat
    diverx_V = np.real(np.fft.ifftn(divhat))
    return diverx_V

def deriv_y(Nnod,V):
    Vhat = np.fft.fftn(V)
    divhat = np.zeros((Nnod,Nnod,Nnod),dtype = complex)
    ky = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
           ky[:,i2,:] = complex(0,i2)
        else:
           ky[:,i2,:] = complex(0,i2-Nnod)  
    
    divhat = ky * Vhat
    divery_V = np.real(np.fft.ifftn(divhat))
    return divery_V

def deriv_z(Nnod,V):
    Vhat = np.fft.fftn(V)
    divhat = np.zeros((Nnod,Nnod,Nnod),dtype = complex)
    kz = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    for i3 in range(Nnod):
        if i3 <= (Nnod-1)/2:
           kz[:,:,i3] = complex(0,i3)
        else:
           kz[:,:,i3] = complex(0,i3-Nnod)   
    
    divhat = kz * Vhat
    diverz_V = np.real(np.fft.ifftn(divhat))
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


def Find_Neighboar(R,N):
    redsz =  int((N)/(2*R))
    x_DNS = np.zeros((N,1))
    x_LES = np.zeros((redsz,1))
    dx = 2.0*np.pi/(N)
    for i in range(N):
        x_DNS[i,0] = i*dx
        
    for i in range(redsz):
        x_LES[i,0] = i*2.0*np.pi/(redsz)
    Ip_Neigh = {}
    i = 0
    for i in range(redsz):
        Ip_Neigh[i] = []
        rcenter = int(x_LES[i,0]/dx)
        for j in range(2*R+3):
            if rcenter==24:
                print(rcenter-R-1+j)
                print(abs(x_LES[i,0]-(rcenter-R-1+j)*dx)-(R*dx))
    #        print(rcenter-R-1+j)
            if abs(x_LES[i,0]-(rcenter-R-1+j)*dx)-(R*dx)<= 10**(-14):
                Ip_Neigh[i].append((rcenter-R-1+j)%N)    
    return Ip_Neigh,redsz

def Eval_num_Neighboar(Ip_Neigh, redsz):
    Ip_num = np.zeros((redsz,redsz,redsz))
    for i in range(redsz):
        for j in range(redsz):
            for k in range(redsz):
                Ip_num[i,j,k]= len()
    

















    
    
    
    