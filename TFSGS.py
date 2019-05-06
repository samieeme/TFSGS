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
#%%   
class Solver(object):
    
    def __init__(self,filename,R):
        data = np.loadtxt(filename)
        prsz = data.shape[0]
        matsz = int(np.ceil(prsz ** (1./3.)))
        self.mainsz = matsz-1
        
        vx = np.ones(prsz)
        vx = data[:,0]
        vy = data[:,1]
        vz = data[:,2]
        
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
        sxxbar , self.redsz = self.box_filter(self.vxx , self.rad)
        sxybar , self.redsz = self.box_filter(self.vxy , self.rad)
        sxzbar , self.redsz = self.box_filter(self.vxz , self.rad)
        syybar , self.redsz = self.box_filter(self.vyy , self.rad)
        syzbar , self.redsz = self.box_filter(self.vyz , self.rad)
        szzbar , self.redsz = self.box_filter(self.vzz , self.rad)
        sxxbar = sxxbar  - self.vxbar*self.vxbar
        sxybar = sxybar  - self.vxbar*self.vybar
        sxzbar = sxzbar  - self.vxbar*self.vzbar
        syybar = syybar  - self.vybar*self.vybar
        syzbar = syzbar  - self.vybar*self.vzbar
        szzbar = szzbar  - self.vzbar*self.vzbar
        return  self.vxbar, self.vybar, self.vzbar, sxxbar, sxybar, sxzbar, syybar, syzbar, szzbar, self.redsz, self.rrr

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
        sx = self.vxm[1]
         
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
filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/April-15-2019/data_DNS/DrJaberi/DNS3Duvw.dat"

R = 4

solver = Solver(filename,R)
vxbar , vybar, vzbar, sxx, sxy, sxz, syy, syz, szz, redsz ,rrr = solver.get_bar_gaussian()
sx_div = div(sxx,sxy,sxz, redsz)
sy_div = div(sxy,syy,syz, redsz)
sz_div = div(sxz,syz,szz, redsz)

solver.plot(R)


mdata = Model(vxbar , vybar, vzbar)

SMG_xx, SMG_yy, SMG_zz, SMG_xy, SMG_xz, SMG_yz, SMG_div_x_div, SMG_div_y_div, SMG_div_z_div = mdata.SMG()

#%%
alpha = 0.875000
s_fL_x,s_fL_y,s_fL_z = mdata.Fractional_Laplacian(alpha)


#sxxy = sx_div.reshape(redsz**3)
#SMG_xxy = SMG_div_x_div.reshape(redsz**3)
#FL_xxy = s_fL_x.reshape(redsz**3)
#print(np.corrcoef(sxxy, SMG_xxy))
#print(np.corrcoef(sxxy, FL_xxy))
corr_SMG = 0
corr_FL = 0 
for i in range(redsz):
    for j in range(redsz):
        corr_SMG += np.corrcoef(sx_div[i,j,:], SMG_div_x_div[i,j,:])
        corr_FL += np.corrcoef(sx_div[i,j,:], s_fL_x[i,j,:])

corr_SMG = corr_SMG/(redsz)**2
corr_FL = corr_FL/(redsz)**2

