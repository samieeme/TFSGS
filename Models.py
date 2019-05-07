# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:39:09 2019

@author: samieeme
"""
import numpy as np
from math import pi, gamma
from astropy.convolution import convolve
from functions import remain, deriv_x, deriv_y, deriv_z, div, \
    Fractional_Laplacian_2, Reduce_period, Tempered_Fractional_Laplacian_2
    
class Model (object):
    
    def __init__(self,vxbar , vybar, vzbar):
        self.Nnod = len(vxbar)
        self.Nmode = (self.Nnod-1)/2 
        self.vxhat = np.fft.fftn(vxbar)
        self.vyhat = np.fft.fftn(vybar)
        self.vzhat = np.fft.fftn(vzbar)
        self.kx = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype=complex)
        self.ky = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype=complex)
        self.kz = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype=complex)
        for i3 in range(self.Nnod):
            if i3 <= (self.Nnod-1)/2:
               self.kz[:,:,i3] = complex(0,i3)
            else:
               self.kz[:,:,i3] = complex(0,i3-self.Nnod) 
        for i2 in range(self.Nnod):
            if i2 <= (self.Nnod-1)/2:
               self.ky[:,i2,:] = complex(0,i2)
            else:
               self.ky[:,i2,:] = complex(0,i2-self.Nnod) 
        for i1 in range(self.Nnod):
            if i1 <= (self.Nnod-1)/2:
               self.kx[i1,:,:] = complex(0,i1)
            else:
               self.kx[i1,:,:] = complex(0,i1-self.Nnod) 
               
    def div(self,v1,v2,v3):
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype = complex)
        divhat = self.kx * v1 + self.ky * v2 + self.kz * v3
        diver_V = np.real(np.fft.ifftn(divhat))
        return diver_V

    def deriv_x(self,Vhat):
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype = complex)
        divhat = self.kx * Vhat
        diverx_V = np.real(np.fft.ifftn(divhat))
        return diverx_V
    
    def deriv_y(self,Vhat):
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype = complex)
        divhat = self.ky * Vhat
        diverx_V = np.real(np.fft.ifftn(divhat))
        return diverx_V
    
    def deriv_z(self,Vhat):
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype = complex)
        divhat = self.kz * Vhat
        diverx_V = np.real(np.fft.ifftn(divhat))
        return diverx_V
    
#    def strain(self):
#        Vxhat = self.vxhat.copy()
#        sxx = self.deriv_x(self,Vxhat)
#        Vyhat = self.vyhat.copy()
#        syy = self.deriv_y(self,Vyhat)
#        Vzhat = self.vzhat.copy()
#        szz = self.deriv_z(self,Vzhat)
#        sxy = 1/2*(self.deriv_x(self,Vyhat)+self.deriv_y(self,Vxhat))
#        sxz = 1/2*(self.deriv_z(self,Vxhat)+self.deriv_x(self,Vzhat))
#        syz = 1/2*(self.deriv_z(self,Vyhat)+self.deriv_y(self,Vzhat))
#        return sxx, syy, szz, sxy, sxz, syz


        
    def strain(self):
        Vxhat = self.vxhat.copy()
        sxx = self.deriv_x(Vxhat)
        Vyhat = self.vyhat.copy()
        syy = self.deriv_y(Vyhat)
        Vzhat = self.vzhat.copy()
        szz = self.deriv_z(Vzhat)
        sxy = 1/2*(self.deriv_x(Vyhat)+self.deriv_y(Vxhat))
        sxz = 1/2*(self.deriv_z(Vxhat)+self.deriv_x(Vzhat))
        syz = 1/2*(self.deriv_z(Vyhat)+self.deriv_y(Vzhat))
        return sxx, syy, szz, sxy, sxz, syz
    
    def SMG(self):
        sxx, syy, szz, sxy, sxz, syz = self.strain()
        S = np.sqrt(sxx**2+syy**2+szz**2+sxy**2+sxz**2+syz**2)
        SMG_xx = -S * sxx * 0.02
        SMG_yy = -S * syy * 0.02
        SMG_zz = -S * szz * 0.02
        SMG_xy = -S * sxy * 0.02
        SMG_xz = -S * sxz * 0.02
        SMG_yz = -S * syz * 0.02
        SMG_div_x_div = div(SMG_xx,SMG_xy,SMG_xz,self.Nnod)
        SMG_div_y_div = div(SMG_xy,SMG_yy,SMG_yz,self.Nnod)
        SMG_div_z_div = div(SMG_xz,SMG_yz,SMG_zz,self.Nnod)
        return SMG_xx, SMG_yy, SMG_zz, SMG_xy, SMG_xz, SMG_yz, SMG_div_x_div, SMG_div_y_div, SMG_div_z_div
     
    def Fractional_Laplacian(self,alpha):
        nu = 0.000185
        tau = (6*nu+1)/2
        mu = (0.1 - 10)/0.9 *(alpha - 2.0) * \
        (2**(alpha)*gamma((alpha+3.)/2.))/np.pi**(3.0/2.0) \
        *np.abs(gamma(-alpha/2.))*gamma(alpha+1)*tau**(alpha-1)
        s_fL_x = -mu/8. * Fractional_Laplacian_2(self.vxhat,self.Nnod,alpha)
        s_fL_y = -mu/8. * Fractional_Laplacian_2(self.vyhat,self.Nnod,alpha)
        s_fL_z = -mu/8. * Fractional_Laplacian_2(self.vzhat,self.Nnod,alpha)
        return s_fL_x,s_fL_y,s_fL_z
    
    def Tempered_Fractional_Laplacian(self,alpha,Lambda):
        nu = 0.000185
        tau = (6*nu+1)/2
        mu = (0.1 - 10)/0.9 *(alpha - 2.0) * \
        (2**(alpha)*gamma((alpha+3.)/2.))/np.pi**(3.0/2.0) \
        *np.abs(gamma(-alpha/2.))*gamma(alpha+1)*tau**(alpha-1)
        s_fL_x = -mu/8. * Tempered_Fractional_Laplacian_2(self.vxhat,self.Nnod,alpha,Lambda)
        s_fL_y = -mu/8. * Tempered_Fractional_Laplacian_2(self.vyhat,self.Nnod,alpha,Lambda)
        s_fL_z = -mu/8. * Tempered_Fractional_Laplacian_2(self.vzhat,self.Nnod,alpha,Lambda)
        return s_fL_x,s_fL_y,s_fL_z