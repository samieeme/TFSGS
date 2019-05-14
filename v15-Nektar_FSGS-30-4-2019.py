# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:28:48 2019

@author: samieeme
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, gamma
from astropy.convolution import convolve
#def div(self):
#    div_vel = np.zeros(self.Nnod^3)
#    div_vel.reshape(self.Nnod,self.Nnod,self.Nnod)
#    fdtest = np.fft.fftn(div_vel)
#    fdx = np.real(np.fft.ifftn(kz*fdtest))

def dist_phi(alpha):
    return np.exp(-1*(alpha-0.7)**2)
      
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

def box_filter_sim(VV,N,R):
 #   redsz = int((self.Nnod)/(2*R))
#   Ix = np.zeros(redsz)
    vxbar = np.zeros((N,N,N))
    kernel = np.ones((2*R+1,2*R+1,2*R+1))
    vxbar = convolve(VV , kernel, boundary='wrap') #signal.convolve(vx3 , kernel, mode='same', boundary='wrap')/(2*R+1)**3
    vx1 = vxbar[range(0,N,2*R),:,:]
    vx2 = vx1[:,range(0,N,2*R),:]
    vx3 = vx2[:,:,range(0,N,2*R)]
  #  vx3 = vxbar[:]
    return vx3
#%%
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

    def box_filter_sim(self,VV,R):
     #   redsz = int((self.Nnod)/(2*R))
    #   Ix = np.zeros(redsz)
        vxbar = np.zeros((self.Nnod,self.Nnod,self.Nnod))
        kernel = np.ones((2*R+1,2*R+1,2*R+1))
        
        vxbar = convolve(VV , kernel, boundary='wrap') #signal.convolve(vx3 , kernel, mode='same', boundary='wrap')/(2*R+1)**3
        vx3 = vxbar[:]
        return vx3
        
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
    
    def SIM (self,vx,vy,vz,R):
        self.vxbar2 = self.box_filter_sim(vx,R)
        self.vybar2 = self.box_filter_sim(vy,R)
        self.vzbar2 = self.box_filter_sim(vz,R)
        
        sxxbar2 = self.box_filter_sim(vx*vx,R)- self.vxbar2*self.vxbar2
        sxybar2 = self.box_filter_sim(vx*vy,R)- self.vxbar2*self.vybar2
        sxzbar2 = self.box_filter_sim(vx*vz,R)- self.vxbar2*self.vzbar2
        syzbar2 = self.box_filter_sim(vy*vz,R)- self.vybar2*self.vzbar2
        syybar2 = self.box_filter_sim(vy*vy,R)- self.vybar2*self.vybar2
        szzbar2 = self.box_filter_sim(vz*vz,R)- self.vzbar2*self.vzbar2
        
        SIM_div_x_div = div(sxxbar2,sxybar2,sxzbar2,self.Nnod)
        SIM_div_y_div = div(sxybar2,syybar2,syzbar2,self.Nnod)
        SIM_div_z_div = div(sxzbar2,syzbar2,szzbar2,self.Nnod)
        return sxxbar2,syybar2,szzbar2,sxybar2,sxzbar2,syzbar2, SIM_div_x_div, SIM_div_y_div, SIM_div_z_div

    def Fractional_Laplacian_bar(self,vx2 , vy2, vz2,alpha):
        vx2hat = np.fft.fftn(vx2)
        vy2hat = np.fft.fftn(vy2)
        vz2hat = np.fft.fftn(vz2)
        nu = 0.000185
        tau = (6*nu+1)/2
        mu = (0.1 - 10)/0.9 *(alpha - 2.0) * \
        (2**(alpha)*gamma((alpha+3.)/2.))/np.pi**(3.0/2.0) \
        *np.abs(gamma(-alpha/2.))*gamma(alpha+1)*tau**(alpha-1)
        s_fL_x = -mu/8. * Fractional_Laplacian_2(vx2hat,self.Nnod,alpha)
        s_fL_y = -mu/8. * Fractional_Laplacian_2(vy2hat,self.Nnod,alpha)
        s_fL_z = -mu/8. * Fractional_Laplacian_2(vz2hat,self.Nnod,alpha)
        return s_fL_x,s_fL_y,s_fL_z



#%%   
class Solver(object):
    
    def __init__(self,filename,R):
        data_u = np.loadtxt(filename+'Res_12_u.dat')
        data_v = np.loadtxt(filename+'Res_12_v.dat')
        data_w = np.loadtxt(filename+'Res_12_w.dat')
        prsz = data_u.shape[0]
        matsz = int(np.ceil(prsz ** (1./3.)))
        self.mainsz = matsz-1
        
        vx = np.ones(prsz)
        vx = data_u[:]
        vy = data_v[:]
        vz = data_w[:]
        
        vxa = vx.reshape(matsz,matsz,matsz)
        vya = vy.reshape(matsz,matsz,matsz)
        vza = vz.reshape(matsz,matsz,matsz)        

        vx1m = vxa[range(0,matsz-1),:,:]
        vx2m = vx1m[:,range(0,matsz-1),:]
        self.vxm = vx2m[:,:,range(0,matsz-1)]
        
        vx1m = vya[range(0,matsz-1),:,:]
        vx2m = vx1m[:,range(0,matsz-1),:]
        self.vym = vx2m[:,:,range(0,matsz-1)]
        
        vx1m = vza[range(0,matsz-1),:,:]
        vx2m = vx1m[:,range(0,matsz-1),:]
        self.vzm = vx2m[:,:,range(0,matsz-1)]
        
        self.vxx = self.vxm * self.vxm
        self.vxy = self.vxm * self.vym
        self.vyy = self.vym * self.vym
        self.vzz = self.vzm * self.vzm
        self.vxz = self.vxm * self.vzm
        self.vyz = self.vym * self.vzm
        self.rad = R
        self.rrr = div(self.vxm,self.vym,self.vzm,self.mainsz)
        
    def box_filter(self,VV,R):
        redsz = int((self.mainsz)/(2*R))
    #   Ix = np.zeros(redsz)
        vxbar = np.zeros((self.mainsz,self.mainsz,self.mainsz))
        kernel = np.ones((2*R+1,2*R+1,2*R+1))
        
        vxbar = convolve(VV , kernel, boundary='wrap') #signal.convolve(vx3 , kernel, mode='same', boundary='wrap')/(2*R+1)**3
        vx1 = vxbar[range(0,self.mainsz,2*R),:,:]
        vx2 = vx1[:,range(0,self.mainsz,2*R),:]
        vx3 = np.transpose(vx2[:,:,range(0,self.mainsz,2*R)])
        return vx3, redsz

    def Gaussian_filter(self,VV,R):
        redsz = int((self.mainsz)/(2*R))
    #   Ix = np.zeros(redsz)
        vxbar = np.zeros((self.mainsz,self.mainsz,self.mainsz))
        dx = np.pi/(1.0*self.mainsz)
        kernel = np.ones((2*R+1,2*R+1,2*R+1))
        filt = 2.0*R*dx
        for i in range(2*R+1):
            for j in range(2*R+1):
                for k in range(2*R+1):
                    kernel[i,j,k] = 1.0/np.sqrt(2*np.pi*filt**2)*np.exp(-1./2.*(( (i-R)**2+(j-R)**2+(k-R)**2 )*dx**2/filt**2))
        
        vxbar = convolve(VV , kernel, boundary='wrap') #signal.convolve(vx3 , kernel, mode='same', boundary='wrap')/(2*R+1)**3
        vx1 = vxbar[range(0,self.mainsz,2*R),:,:]
        vx2 = vx1[:,range(0,self.mainsz,2*R),:]
        vx3 = np.transpose(vx2[:,:,range(0,self.mainsz,2*R)])
        return vx3, redsz

    def get_bar_box(self):
        self.vxbar , self.redsz = self.box_filter(self.vxm , self.rad)
        self.vybar , self.redsz = self.box_filter(self.vym , self.rad)
        self.vzbar , self.redsz = self.box_filter(self.vzm , self.rad)
        self.sxxbar1 , self.redsz = self.box_filter(self.vxx , self.rad)
        self.sxybar1 , self.redsz = self.box_filter(self.vxy , self.rad)
        self.sxzbar1 , self.redsz = self.box_filter(self.vxz , self.rad)
        self.syybar1 , self.redsz = self.box_filter(self.vyy , self.rad)
        self.syzbar1 , self.redsz = self.box_filter(self.vyz , self.rad)
        self.szzbar1 , self.redsz = self.box_filter(self.vzz , self.rad)
        sxxbar = self.sxxbar1  - self.vxbar*self.vxbar
        sxybar = self.sxybar1  - self.vxbar*self.vybar
        sxzbar = self.sxzbar1  - self.vxbar*self.vzbar
        syybar = self.syybar1  - self.vybar*self.vybar
        syzbar = self.syzbar1  - self.vybar*self.vzbar
        szzbar = self.szzbar1  - self.vzbar*self.vzbar
        return  self.vxbar, self.vybar, self.vzbar, sxxbar, sxybar, sxzbar, syybar, syzbar, szzbar, self.redsz, self.rrr

    def Strsses_bar (self,R):
        sxxbar2 = box_filter_sim(self.sxxbar1,self.redsz,R)
        sxybar2 = box_filter_sim(self.sxybar1,self.redsz,R)
        sxzbar2 = box_filter_sim(self.sxzbar1,self.redsz,R)
        syybar2 = box_filter_sim(self.syybar1,self.redsz,R)
        syzbar2 = box_filter_sim(self.syzbar1,self.redsz,R)
        szzbar2 = box_filter_sim(self.szzbar1,self.redsz,R)
        vxbar2 = box_filter_sim(self.vxbar,self.redsz,R)  
        vybar2 = box_filter_sim(self.vybar,self.redsz,R)
        vzbar2 = box_filter_sim(self.vybar,self.redsz,R)
        sxxbar2 = sxxbar2 - vxbar2 * vxbar2
        sxybar2 = sxybar2 - vxbar2 * vybar2
        sxzbar2 = sxzbar2 - vxbar2 * vzbar2
        syybar2 = syybar2 - vybar2 * vybar2
        syzbar2 = syzbar2 - vybar2 * vzbar2
        szzbar2 = szzbar2 - vzbar2 * vzbar2
        return sxxbar2, sxybar2, sxzbar2, syybar2, syzbar2, szzbar2,vxbar2,vybar2,vzbar2
        
    def get_bar_gaussian(self):
        self.vxbar , self.redsz = self.Gaussian_filter(self.vxm , self.rad)
        self.vybar , self.redsz = self.Gaussian_filter(self.vym , self.rad)
        self.vzbar , self.redsz = self.Gaussian_filter(self.vzm , self.rad)
        sxxbar , self.redsz = self.Gaussian_filter(self.vxx , self.rad)
        sxybar , self.redsz = self.Gaussian_filter(self.vxy , self.rad)
        sxzbar , self.redsz = self.Gaussian_filter(self.vxz , self.rad)
        syybar , self.redsz = self.Gaussian_filter(self.vyy , self.rad)
        syzbar , self.redsz = self.Gaussian_filter(self.vyz , self.rad)
        szzbar , self.redsz = self.Gaussian_filter(self.vzz , self.rad)
        sxxbar = sxxbar  - self.vxbar*self.vxbar
        sxybar = sxybar  - self.vxbar*self.vybar
        sxzbar = sxzbar  - self.vxbar*self.vzbar
        syybar = syybar  - self.vybar*self.vybar
        syzbar = syzbar  - self.vybar*self.vzbar
        szzbar = szzbar  - self.vzbar*self.vzbar
        return  self.vxbar, self.vybar, self.vzbar, sxxbar, sxybar, sxzbar, syybar, syzbar, szzbar, self.redsz, self.rrr            

    def plot(self,R):
        x = np.linspace(0, 2*pi, self.mainsz)
        y = np.linspace(0, 2*pi, self.mainsz)
        
        X, Y = np.meshgrid(x, y)
        sx = self.vxm[5]
         
        plt.contourf(X, Y, sx, 50, cmap='RdGy')
        plt.colorbar();
        plt.show()
        
        x = np.linspace(0, 2*pi, self.redsz)
        y = np.linspace(0, 2*pi, self.redsz)
        
        X, Y = np.meshgrid(x, y)
        sx = self.vxbar[1]
         
        plt.contourf(X, Y, sx, 50, cmap='RdGy')
        plt.colorbar()
        plt.show()
#%%
#filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/April-15-2019/data_DNS/DrJaberi/DNS3Duvw.dat"
filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/DNS-data/Nektar/"
R = 4

solver = Solver(filename,R)
vxbar , vybar, vzbar, sxx, sxy, sxz, syy, syz, szz, redsz ,rrr = solver.get_bar_box()
sx_div = div(sxx,sxy,sxz, redsz)
sy_div = div(sxy,syy,syz, redsz)
sz_div = div(sxz,syz,szz, redsz)
#sxx_bar2, sxy_bar2, sxz_bar2, syy_bar2, syz_bar2, szz_bar2, vxbar2 , vybar2, vzbar2 = solver.Strsses_bar (1)
#sx_div_bar2 = div(sxx_bar2,sxy_bar2,sxz_bar2, redsz)
#sy_div_bar2 = div(sxy_bar2,syy_bar2,syz_bar2, redsz)
#sz_div_bar2 = div(sxz_bar2,syz_bar2,szz_bar2, redsz)


solver.plot(R)


mdata = Model(vxbar , vybar, vzbar)

SMG_xx, SMG_yy, SMG_zz, SMG_xy, SMG_xz, SMG_yz, SMG_div_x_div, SMG_div_y_div, SMG_div_z_div = mdata.SMG()

#SIM_xx, SIM_yy, SMG_zz, SIM_xy, SIM_xz, SIM_yz, SIM_div_x_div, SIM_div_y_div, SIM_div_z_div = mdata.SIM(vxbar,vybar,vzbar,1)




#%%
alpha = 0.950
s_fL_x,s_fL_y,s_fL_z = mdata.Fractional_Laplacian(alpha)

#alpha = 0.91005000
#s_fL_xbar,s_fL_ybar,s_fL_zbar = mdata.Fractional_Laplacian_bar(vxbar2 , vybar2, vzbar2,alpha)
#sx_div_bar = mdata.box_filter_sim(sx_div,1)



Lambda = 100
s_tfL_x, s_tfL_y, s_tfL_z = mdata.Tempered_Fractional_Laplacian(alpha,Lambda)



s_dfL_x = np.zeros((redsz,redsz,redsz))
s_dfL_y = np.zeros((redsz,redsz,redsz))
s_dfL_z = np.zeros((redsz,redsz,redsz))
alpha = np.linspace(0.4,0.95,50)
len_alpha = len(alpha)
for i in range(len_alpha):
    print(i,alpha[i])
    s_fL_x_t,s_fL_y_t,s_fL_z_t = mdata.Fractional_Laplacian(alpha[i])
    s_dfL_x += dist_phi(alpha[i])* s_fL_x_t
    s_dfL_y += dist_phi(alpha[i])* s_fL_y_t
    s_dfL_z += dist_phi(alpha[i])* s_fL_z_t


sxxy = sxy.reshape(redsz**3)
Sx_div =sx_div.reshape(redsz**3)
#SIM_xxy = SIM_xy.reshape(redsz**3)
SMG_xxy = SMG_xy.reshape(redsz**3)
FL_xxy = s_fL_x.reshape(redsz**3)

print(np.corrcoef(sxxy, SMG_xxy))
#print(np.corrcoef(sxxy, SIM_xxy))
print(np.corrcoef(Sx_div, FL_xxy))

corr_SMG = 0
corr_FL = 0 
corr_SIM = 0
corr_TFL = 0 
corr_DFL = 0 
corr_FL_bar = 0 

for i in range(redsz):
    for j in range(redsz):
        corr_SMG += np.corrcoef(sx_div[i,j,:], SMG_div_x_div[i,j,:])
     #   corr_SIM += np.corrcoef(sx_div[i,j,:], SIM_div_x_div[i,j,:])
        corr_FL += np.corrcoef(sx_div[i,j,:], s_fL_x[i,j,:])
        corr_TFL += np.corrcoef(sx_div[i,j,:], s_tfL_x[i,j,:])
        corr_DFL += np.corrcoef(sx_div[i,j,:], s_dfL_x[i,j,:])
    #    corr_FL_bar += np.corrcoef(sx_div_bar2[i,j,:], s_fL_xbar[i,j,:])
        
corr_SMG = corr_SMG/(redsz)**2
corr_FL = corr_FL/(redsz)**2
#corr_SIM = corr_SIM/(redsz)**2
corr_TFL = corr_TFL/(redsz)**2
corr_DFL = corr_DFL/(redsz)**2
#corr_FL_bar = corr_FL_bar/(redsz)**2

