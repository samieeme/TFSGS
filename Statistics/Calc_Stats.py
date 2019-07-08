#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:51:16 2019

@author: akhavans
"""

import numpy as np
from numpy import linalg as LA
import os
import sys

time=sys.argv[1]
filepath=os.path.join(os.getcwd(),'uvw')
filename='Up'+time+'.csv'
data=np.genfromtxt(filepath+'/'+filename, delimiter=',', skip_header=1)

sz=data.shape[0]
res=int(np.ceil(sz ** (1./3.)))

u1=data[:,0].reshape(res,res,res)
u2=data[:,1].reshape(res,res,res)
u3=data[:,2].reshape(res,res,res)

del data

res=res-1
u1=u1[0:res,0:res,0:res]
u2=u2[0:res,0:res,0:res]
u3=u3[0:res,0:res,0:res]

PI=np.pi

Delta=2.0*PI/res

#viscosity
nu=0.001

#Total kinetic energy
E_tot=0.5*(np.mean(u1**2)+np.mean(u2**2)+np.mean(u3**2))

#rms velocity
U_rms=np.sqrt(2.0*E_tot/3.0)

du1dx=np.zeros([res,res,res])
du1dy=np.zeros([res,res,res])
du1dz=np.zeros([res,res,res])

du2dx=np.zeros([res,res,res])
du2dy=np.zeros([res,res,res])
du2dz=np.zeros([res,res,res])

du3dx=np.zeros([res,res,res])
du3dy=np.zeros([res,res,res])
du3dz=np.zeros([res,res,res])

#Computing strain-rate tensor
for ix in range(0,res-1):
    du1dx[ix,:,:]=0.5*(u1[ix+1,:,:]-u1[ix-1,:,:])/Delta
    du2dx[ix,:,:]=0.5*(u2[ix+1,:,:]-u2[ix-1,:,:])/Delta
    du3dx[ix,:,:]=0.5*(u3[ix+1,:,:]-u3[ix-1,:,:])/Delta

    du1dy[:,ix,:]=0.5*(u1[:,ix+1,:]-u1[:,ix-1,:])/Delta
    du2dy[:,ix,:]=0.5*(u2[:,ix+1,:]-u2[:,ix-1,:])/Delta
    du3dy[:,ix,:]=0.5*(u3[:,ix+1,:]-u3[:,ix-1,:])/Delta

    du1dz[:,:,ix]=0.5*(u1[:,:,ix+1]-u1[:,:,ix-1])/Delta
    du2dz[:,:,ix]=0.5*(u2[:,:,ix+1]-u2[:,:,ix-1])/Delta
    du3dz[:,:,ix]=0.5*(u3[:,:,ix+1]-u3[:,:,ix-1])/Delta


ix=res-1
du1dx[ix,:,:]=0.5*(u1[0,:,:]-u1[ix-1,:,:])/Delta
du2dx[ix,:,:]=0.5*(u2[0,:,:]-u2[ix-1,:,:])/Delta
du3dx[ix,:,:]=0.5*(u3[0,:,:]-u3[ix-1,:,:])/Delta

du1dy[:,ix,:]=0.5*(u1[:,0,:]-u1[:,ix-1,:])/Delta
du2dy[:,ix,:]=0.5*(u2[:,0,:]-u2[:,ix-1,:])/Delta
du3dy[:,ix,:]=0.5*(u3[:,0,:]-u3[:,ix-1,:])/Delta

du1dz[:,:,ix]=0.5*(u1[:,:,0]-u1[:,:,ix-1])/Delta
du2dz[:,:,ix]=0.5*(u2[:,:,0]-u2[:,:,ix-1])/Delta
du3dz[:,:,ix]=0.5*(u3[:,:,0]-u3[:,:,ix-1])/Delta

#Computing <S_ij S_ij>
S11=du1dx**2
S22=du2dy**2
S33=du3dz**2
S12=0.25*(du1dy+du2dx)**2
S23=0.25*(du2dz+du3dy)**2
S13=0.25*(du1dz+du3dx)**2

S=np.mean(S11+S22+S33+2.0*(S12+S13+S23))

#Dissipation
epsilon=2.0*nu*S

#Taylor microscale
lamb=np.sqrt(15.0*nu*U_rms**2/epsilon)

#Taylor-scale Reynols
Re_T=U_rms*lamb/nu

#Kolmogorov time scale
tau_eta=np.sqrt(nu/epsilon)

#Kolmogorov length scale
eta=(nu**3/epsilon)**0.25

#clear the memory
del S11, S12, S13, S22, S23, S33
del du1dx, du2dx, du3dx, du1dy, du2dy, du3dy, du1dz, du2dz, du3dz

out=np.array([E_tot, U_rms, epsilon, lamb, Re_T, tau_eta, eta])
np.savetxt('Stats_'+time+'.dat', out, delimiter=',')

###############################################################
###############################################################

#Taking FFT from the velocity components
u1hat = np.fft.fftn(u1)
u2hat = np.fft.fftn(u2)
u3hat = np.fft.fftn(u3)

Esp=np.zeros((sz,2))
wave_n=np.array([0.0,0.0,0.0])
max_wave=int(res/2)

tmp_U=np.zeros((1,3), dtype=complex)
ndx=0

for k in range(0,res):
    for j in range(0,res):
        for i in range(0,res):

            tmp_U[0,0]=u1hat[i,j,k]
            tmp_U[0,1]=u2hat[i,j,k]
            tmp_U[0,2]=u3hat[i,j,k]

            U_mag=LA.norm(tmp_U, ord=2)
            Esp[ndx,0]=0.5*U_mag**2
            
            wave_n[0]=i
            if i > max_wave:
                wave_n[0]=i-res
            wave_n[1]=j
            if j > max_wave:
                wave_n[1]=j-res
            wave_n[2]=k
            if k > max_wave:
                wave_n[2]=k-res

            Esp[ndx,1]=np.round(LA.norm(wave_n, ord=2))
            ndx +=1

Esp_sort = Esp[Esp[:,1].argsort()]

del Esp

Esp_sort[:,1][Esp_sort[:,1]==0.0]=1.0
Energy_Spct = np.zeros(int(Esp_sort[-1,1]))
Intg_scale = 0.0

for j in range(0,Energy_Spct.size):
    i=j+1
    tmp1=np.nonzero(Esp_sort[:,1]==i)
    tmp2=tmp1[0]
    Energy_Spct[j]=np.mean(Esp_sort[tmp2[0]:tmp2[-1]+1,0])
    
    if i == 1:
        vol=(1.5**3)*PI*4.0/3.0
    else:
        vol=((i+0.5)**3-(i-0.5)**3)*PI*4.0/3.0
        
    Energy_Spct[j] *= vol
    
    Intg_scale +=Energy_Spct[j]/i
    

#Integral scale    
Intg_scale /= np.sum(Energy_Spct)

#Large eddy turnover time
T_L = Intg_scale/U_rms
    
    
    
