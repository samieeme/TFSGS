# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:28:48 2019

@author: samieeme
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, gamma
from astropy.convolution import convolve
from functions_2 import remain, deriv_x, deriv_y, deriv_z, div, \
    Fractional_Laplacian_2, Reduce_period
from Models_2 import Model


class Solver_filtered_field(object):
    
    def __init__(self,filename,Rs):
        name_Vx = "/U_Filtered_vx_"+Rs+".dat"
        name_Vy = "/U_Filtered_vy_"+Rs+".dat"
        name_Vz = "/U_Filtered_vz_"+Rs+".dat"
        name_sxx = "/U_Filtered_Sxx_"+Rs+".dat"
        name_sxy = "/U_Filtered_Sxy_"+Rs+".dat"
        name_sxz = "/U_Filtered_Sxz_"+Rs+".dat"
        name_syz = "/U_Filtered_Syz_"+Rs+".dat"
        name_syy = "/U_Filtered_Syy_"+Rs+".dat"
        name_szz = "/U_Filtered_Szz_"+Rs+".dat"
        
        vx = np.loadtxt(filename+name_Vx)
        vy = np.loadtxt(filename+name_Vy)
        vz = np.loadtxt(filename+name_Vz)
        sxxm = np.loadtxt(filename+name_sxx)
        sxym = np.loadtxt(filename+name_sxy)
        sxzm = np.loadtxt(filename+name_sxz)
        syym = np.loadtxt(filename+name_syy)
        syzm = np.loadtxt(filename+name_syz)
        szzm = np.loadtxt(filename+name_szz)
        
        prsz = vx.shape[0]
        matsz = int(np.ceil(prsz ** (1./3.)))
        self.redsz = matsz-1
        
        self.vxbar = Reduce_period(vx,matsz)
        self.vybar = Reduce_period(vy,matsz)
        self.vzbar = Reduce_period(vz,matsz)
        
        self.sxx = Reduce_period(sxxm,matsz) - self.vxbar * self.vxbar
        self.sxy = Reduce_period(sxym,matsz) - self.vxbar * self.vybar
        self.sxz = Reduce_period(sxzm,matsz) - self.vxbar * self.vzbar
        self.syy = Reduce_period(syym,matsz) - self.vybar * self.vzbar
        self.syz = Reduce_period(syzm,matsz) - self.vybar * self.vzbar
        self.szz = Reduce_period(szzm,matsz) - self.vzbar * self.vzbar
        
        
        
    def Output(self):
        return self.vxbar, self.vybar, self.vzbar, self.sxx, self.sxy, self.sxz, self.syy, self.syz, self.szz, self.redsz

    def plot(self):
#        x = np.linspace(0, 2*pi, self.mainsz)
#        y = np.linspace(0, 2*pi, self.mainsz)
#        
#        X, Y = np.meshgrid(x, y)
#        sx = self.vxm[1]
#         
#        plt.contourf(X, Y, sx, 50, cmap='RdGy')
#        plt.colorbar();
#        plt.show()
#        
        x = np.linspace(0, 2*pi, self.redsz)
        y = np.linspace(0, 2*pi, self.redsz)
        
        X, Y = np.meshgrid(x, y)
        sx = self.vxbar[1]
#         
        plt.contourf(X, Y, sx, 50, cmap='RdGy')
        plt.colorbar()
        plt.show()
#%%
Rs = str(128)

filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/DNS-data/t-1000/DNS_"+ Rs


solver = Solver_filtered_field(filename,Rs)
vxbar , vybar, vzbar, sxx, sxy, sxz, syy, syz, szz, redsz = solver.Output()
sx_div = div(sxx,sxy,sxz, redsz)
sy_div = div(sxy,syy,syz, redsz)
sz_div = div(sxz,syz,szz, redsz)

solver.plot()


mdata = Model(vxbar , vybar, vzbar)

SMG_xx, SMG_yy, SMG_zz, SMG_xy, SMG_xz, SMG_yz, SMG_div_x_div, SMG_div_y_div, SMG_div_z_div = mdata.SMG()

#%%
alpha = 0.755000
s_fL_x,s_fL_y,s_fL_z = mdata.Fractional_Laplacian(alpha)

Lambda = 0.1
s_tfL_x,s_tfL_y,s_tfL_z = mdata.Tempered_Fractional_Laplacian(alpha,Lambda)
#
sxxy = sx_div.reshape(redsz**3)
SMG_xxy = SMG_div_x_div.reshape(redsz**3)
FL_xxy = s_fL_x.reshape(redsz**3)
TFL_xxy = s_tfL_x.reshape(redsz**3)
print(np.corrcoef(sxxy, SMG_xxy))
print(np.corrcoef(sxxy, FL_xxy))
print(np.corrcoef(sxxy, TFL_xxy))

corr_SMG = 0
corr_FL = 0 
corr_TFL = 0 
for i in range(redsz):
    for j in range(redsz):
        corr_SMG += np.corrcoef(sy_div[i,j,:], SMG_div_y_div[i,j,:])
        corr_FL += np.corrcoef(sy_div[i,j,:], s_fL_y[i,j,:])
        corr_TFL += np.corrcoef(sx_div[i,j,:], s_tfL_x[i,j,:])

corr_SMG = corr_SMG/(redsz)**2
corr_FL = corr_FL/(redsz)**2
corr_TFL = corr_TFL/(redsz)**2
