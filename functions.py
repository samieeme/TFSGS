# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:24:28 2019

@author: samieeme
"""
import numpy as np

def remain (a,b,v):
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

def Fractional_Laplacian_2(Vhat,Nnod,alpha):
#    Vhat = np.fft.fftn(v)
    divhat = np.zeros((Nnod,Nnod,Nnod))
    kz = np.zeros((Nnod,Nnod,Nnod))
    ky = np.zeros((Nnod,Nnod,Nnod))
    kx = np.zeros((Nnod,Nnod,Nnod)) 
    for i3 in range(Nnod):
        if i3 <= (Nnod-1)/2:
           kz[:,:,i3] = i3
        else:
           kz[:,:,i3] = i3-Nnod
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
           ky[:,i2,:] = i2
        else:
           ky[:,i2,:] = i2-Nnod
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:,:] = i1
        else:
           kx[i1,:,:] = i1-Nnod
    frac_L = -(kx[:]**2+ky[:]**2+kz[:]**2)**(alpha)       
    divhat = frac_L * Vhat
    diverz_V = np.real(np.fft.ifftn(divhat))
    return diverz_V

def Tempered_Fractional_Laplacian_2(Vhat,Nnod,alpha,Lambda):
#    Vhat = np.fft.fftn(v)
    divhat = np.zeros((Nnod,Nnod,Nnod))
    kz = np.zeros((Nnod,Nnod,Nnod))
    ky = np.zeros((Nnod,Nnod,Nnod))
    kx = np.zeros((Nnod,Nnod,Nnod)) 
    for i3 in range(Nnod):
        if i3 <= (Nnod-1)/2:
           kz[:,:,i3] = i3
        else:
           kz[:,:,i3] = i3-Nnod
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
           ky[:,i2,:] = i2
        else:
           ky[:,i2,:] = i2-Nnod
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:,:] = i1
        else:
           kx[i1,:,:] = i1-Nnod
    abs_omega2 = (kx[:]**2+ky[:]**2+kz[:]**2)
    abs_omega = np.sqrt(kx[:]**2+ky[:]**2+kz[:]**2)
    frac_L = -(Lambda**(2*alpha)-(Lambda**2+abs_omega2)**(alpha) * np.cos(2.0*alpha*np.arctan(abs_omega/Lambda)) )       
    divhat = frac_L * Vhat
    diverz_V = np.real(np.fft.ifftn(divhat))
    return diverz_V

def Reduce_period(V,N):
    Vbar1 = V.reshape(N,N,N)
    Vbar2 = Vbar1[range(0,N-1),:,:]
    Vbar3 = Vbar2[:,range(0,N-1),:]
    Vbar = np.transpose(Vbar3[:,:,range(0,N-1)])
    return Vbar