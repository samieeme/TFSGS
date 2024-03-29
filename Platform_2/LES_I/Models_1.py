# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:39:09 2019

@author: samieeme
"""
import numpy as np
from math import pi, gamma
from astropy.convolution import convolve
from functions_1 import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, Reduce_period_JHU
from scipy.fftpack import fftfreq
import scipy.special as sc
from mkl_fft import fftn, ifftn
from mkl import set_num_threads, get_max_threads

set_num_threads(4)

#%%  Reading and preparing data

class Solver_filtered_field_JHU(object):
    
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
        
        self.vxbar = Reduce_period_JHU(vx,matsz)
        self.vybar = Reduce_period_JHU(vy,matsz)
        self.vzbar = Reduce_period_JHU(vz,matsz)
        
        self.sxx = Reduce_period_JHU(sxxm,matsz) - self.vxbar * self.vxbar
        self.sxy = Reduce_period_JHU(sxym,matsz) - self.vxbar * self.vybar
        self.sxz = Reduce_period_JHU(sxzm,matsz) - self.vxbar * self.vzbar
        self.syy = Reduce_period_JHU(syym,matsz) - self.vybar * self.vybar
        self.syz = Reduce_period_JHU(syzm,matsz) - self.vybar * self.vzbar
        self.szz = Reduce_period_JHU(szzm,matsz) - self.vzbar * self.vzbar
        
    def Output (self):
        return self.vxbar, self.vybar, self.vzbar, self.sxx, self.sxy, self.sxz, self.syy, self.syz, self.szz, self.redsz

# class Solver_filtered_field_unified(object):
    
#     def __init__(self,filename,Rs,time):
#         name_V = "/vel-t"+time+"-R"+Rs+".csv"
#         name_sgs = "/sgs-t"+time+"-R"+Rs+".csv"
#         V = np.genfromtxt(filename+name_V,delimiter=",")
#         SGSm = np.genfromtxt(filename+name_sgs,delimiter=",")
#         prsz = V.shape[0]
#         self.matsz = int(np.ceil(prsz ** (1./3.)))
#         self.vx = V[:,0]
#         self.vy = V[:,1]
#         self.vz = V[:,2]
#         self.sxxm = SGSm[:,0]
#         self.sxym = SGSm[:,1]
#         self.sxzm = SGSm[:,2]
#         self.syym = SGSm[:,3]
#         self.syzm = SGSm[:,4]
#         self.szzm = SGSm[:,5]

#     def Output(self):      
#         self.vxbar = Reduce_period(self.vx,self.matsz)
#         self.vybar = Reduce_period(self.vy,self.matsz)
#         self.vzbar = Reduce_period(self.vz,self.matsz)
        
#         self.sxx = Reduce_period(self.sxxm,self.matsz) #- self.vxbar * self.vxbar
#         self.sxy = Reduce_period(self.sxym,self.matsz) #- self.vxbar * self.vybar
#         self.sxz = Reduce_period(self.sxzm,self.matsz) #- self.vxbar * self.vzbar
#         self.syy = Reduce_period(self.syym,self.matsz) #- self.vybar * self.vzbar
#         self.syz = Reduce_period(self.syzm,self.matsz) #- self.vybar * self.vzbar
#         self.szz = Reduce_period(self.szzm,self.matsz) #- self.vzbar * self.vzbar
#         return self.vxbar, self.vybar, self.vzbar, self.sxx, self.sxy, self.sxz, self.syy, self.syz, self.szz, self.matsz

class Solver_filtered_field_sep(object):
    
    def __init__(self,add_in,filename,Rs):
        name_V = filename
        #name_V = "/velx-t1-R"+Rs+".csv"
        Vx = np.genfromtxt(add_in+"/"+name_V[0])
        #name_V = "/vely-t1-R"+Rs+".csv"
        Vy = np.genfromtxt(add_in+"/"+name_V[1])
        #name_V = "/velz-t1-R"+Rs+".csv"
        Vz = np.genfromtxt(add_in+"/"+name_V[2])
        
        prsz = Vx.shape[0]
        self.matsz = int(np.ceil(prsz ** (1./3.)))
        
        SGSm = np.zeros((prsz , 6))
        #name_sgs = "/Sxx-t1-R"+Rs+".csv"
        SGSm[:,0] = np.genfromtxt(add_in+"/"+name_V[3])
        #name_sgs = "/Sxy-t1-R"+Rs+".csv"
        SGSm[:,1] = np.genfromtxt(add_in+"/"+name_V[4])
        #name_sgs = "/Sxz-t1-R"+Rs+".csv"
        SGSm[:,2] = np.genfromtxt(add_in+"/"+name_V[5])
        #name_sgs = "/Syy-t1-R"+Rs+".csv"
        SGSm[:,3] = np.genfromtxt(add_in+"/"+name_V[6])
        #name_sgs = "/Syz-t1-R"+Rs+".csv"
        SGSm[:,4] = np.genfromtxt(add_in+"/"+name_V[7])
        #name_sgs = "/Szz-t1-R"+Rs+".csv"
        SGSm[:,5] = np.genfromtxt(add_in+"/"+name_V[8])
    # def __init__(self,filename,Rs):
    #     #name_V = filename
    #     name_V = "/1_velx-R"+Rs+".csv"
    #     Vx = np.genfromtxt(filename+name_V)
    #     name_V = "/2_vely-R"+Rs+".csv"
    #     Vy = np.genfromtxt(filename+name_V)
    #     name_V = "/3_velz-R"+Rs+".csv"
    #     Vz = np.genfromtxt(filename+name_V)
        
    #     prsz = Vx.shape[0]
    #     self.matsz = int(np.ceil(prsz ** (1./3.)))
        
    #     SGSm = np.zeros((prsz , 6))
    #     name_sgs = "/4_Sxx-R"+Rs+".csv"
    #     SGSm[:,0] = np.genfromtxt(filename+name_sgs)
    #     name_sgs = "/5_Sxy-R"+Rs+".csv"
    #     SGSm[:,1] = np.genfromtxt(filename+name_sgs)
    #     name_sgs = "/6_Sxz-R"+Rs+".csv"
    #     SGSm[:,2] = np.genfromtxt(filename+name_sgs)
    #     name_sgs = "/7_Syy-R"+Rs+".csv"
    #     SGSm[:,3] = np.genfromtxt(filename+name_sgs)
    #     name_sgs = "/8_Syz-R"+Rs+".csv"
    #     SGSm[:,4] = np.genfromtxt(filename+name_sgs)
    #     name_sgs = "/9_Szz-R"+Rs+".csv"
    #     SGSm[:,5] = np.genfromtxt(filename+name_sgs)        
        self.vx = Vx[:]
        self.vy = Vy[:]
        self.vz = Vz[:]
        self.sxxm = SGSm[:,0]
        self.sxym = SGSm[:,1]
        self.sxzm = SGSm[:,2]
        self.syym = SGSm[:,3]
        self.syzm = SGSm[:,4]
        self.szzm = SGSm[:,5]

    def Output(self):      
        self.vxbar = Reduce_period(self.vx,self.matsz)
        self.vybar = Reduce_period(self.vy,self.matsz)
        self.vzbar = Reduce_period(self.vz,self.matsz)
        
        self.sxx = Reduce_period(self.sxxm,self.matsz) #- self.vxbar * self.vxbar
        self.sxy = Reduce_period(self.sxym,self.matsz) #- self.vxbar * self.vybar
        self.sxz = Reduce_period(self.sxzm,self.matsz) #- self.vxbar * self.vzbar
        self.syy = Reduce_period(self.syym,self.matsz) #- self.vybar * self.vzbar
        self.syz = Reduce_period(self.syzm,self.matsz) #- self.vybar * self.vzbar
        self.szz = Reduce_period(self.szzm,self.matsz) #- self.vzbar * self.vzbar
        return self.vxbar, self.vybar, self.vzbar, self.sxx, self.sxy, self.sxz, self.syy, self.syz, self.szz, self.matsz
    
#%%    
class Pre_Models (object):
    
    def __init__(self,vxbar , vybar, vzbar):
        self.Nnod = len(vxbar)
        self.Nmode = (self.Nnod-1)/2 
        self.vxhat = fftn(vxbar)
        self.vyhat = fftn(vybar)
        self.vzhat = fftn(vzbar)
        self.kx = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype=complex)
        self.ky = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype=complex)
        self.kz = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype=complex)
        freq = fftfreq(self.Nnod,d=1/self.Nnod)*complex(0,1)
        unit = np.ones(self.Nnod)
        self.kz = np.kron(np.kron(unit,unit),freq).reshape(self.Nnod,self.Nnod,self.Nnod)
        self.ky = np.kron(np.kron(unit,freq),unit).reshape(self.Nnod,self.Nnod,self.Nnod)
        self.kx = np.kron(np.kron(freq,unit),unit).reshape(self.Nnod,self.Nnod,self.Nnod)
               
    def div(self,v1,v2,v3):
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype = complex)
        divhat = self.kx * v1 + self.ky * v2 + self.kz * v3
        diver_V = np.real(ifftn(divhat))
        return diver_V

    def deriv_x(self,Vhat):
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype = complex)
        divhat = self.kx * Vhat
        diverx_V = np.real(ifftn(divhat))
        return diverx_V
    
    def deriv_y(self,Vhat):
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype = complex)
        divhat = self.ky * Vhat
        diverx_V = np.real(ifftn(divhat))
        return diverx_V
    
    def deriv_z(self,Vhat):
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype = complex)
        divhat = self.kz * Vhat
        diverx_V = np.real(ifftn(divhat))
        return diverx_V
    
#    def box_filter(self,VV,R):
#        redsz = int((self.Nnod)/(2*R))
#    #   Ix = np.zeros(redsz)
#        vxbar = np.zeros((self.Nnod,self.Nnod,self.Nnod))
#        kernel = np.ones((2*R+1,2*R+1,2*R+1))
#        
#        vxbar = convolve(VV , kernel, boundary='wrap') #signal.convolve(vx3 , kernel, mode='same', boundary='wrap')/(2*R+1)**3
#        vx1 = vxbar[range(0,self.Nnod,2*R),:,:]
#        vx2 = vx1[:,range(0,self.Nnod,2*R),:]
#        vx3 = np.transpose(vx2[:,:,range(0,self.Nnod,2*R)])
#        return vx3, redsz
    

    
#%%
class FSGS_Model (Pre_Models):
    
    def Fractional_Laplacian_2(self,Vhat,alpha):
    #    Vhat = np.fft.fftn(v)
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod))
        kz = np.zeros((self.Nnod,self.Nnod,self.Nnod))
        ky = np.zeros((self.Nnod,self.Nnod,self.Nnod))
        kx = np.zeros((self.Nnod,self.Nnod,self.Nnod)) 
        freq = fftfreq(self.Nnod,d=1/self.Nnod)*complex(0,1)
        dis = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype=complex)
        for i in range(self.Nnod):
            for j in range(self.Nnod):
                for k in range(self.Nnod):
                    dis[i,j,k] = np.sqrt(abs(freq[i])**2 + abs(freq[j])**2 + abs(freq[k])**2)

        frac_L  = -(dis**2)**(alpha)     
        divhat = frac_L * Vhat
        diverz_V = np.real(ifftn(divhat))
        return diverz_V
    
    def Fractional_Laplacian(self,alpha,nu):
        tau = (6*nu+1)/2
       #-(alpha - 1.0) * \
        mu =40*alpha**2 *gamma(2*alpha+1)*tau**(alpha-1)  * nu**(alpha)
        vx = self.vxhat[:]
        vy = self.vyhat[:]
        vz = self.vzhat[:]
        s_fL_x = -mu* self.Fractional_Laplacian_2(vx,alpha)
        s_fL_y = -mu* self.Fractional_Laplacian_2(vy,alpha)
        s_fL_z = -mu* self.Fractional_Laplacian_2(vz,alpha)
        return s_fL_x,s_fL_y,s_fL_z
    
    def FSGS_stress (self, alpha, nu):
        s_fL_x,s_fL_y,s_fL_z = self.Fractional_Laplacian(alpha-1.0/2.0,nu)
        sxx = -deriv_x(self.Nnod,s_fL_x)
        syy = -deriv_y(self.Nnod,s_fL_y)
        szz = -deriv_z(self.Nnod,s_fL_z)
        sxy = -1.0/2.0*(deriv_x(self.Nnod,s_fL_y)+deriv_y(self.Nnod,s_fL_x))
        sxz = -1.0/2.0*(deriv_x(self.Nnod,s_fL_z)+deriv_z(self.Nnod,s_fL_x))
        syz = -1.0/2.0*(deriv_z(self.Nnod,s_fL_y)+deriv_y(self.Nnod,s_fL_z))
        return sxx, syy, szz, sxy, sxz, syz
        
#%%
class SMG_Model (Pre_Models):
    
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

#%%
class Tempered_FSGS (Pre_Models):

    def Tempered_Fractional_Laplacian_2(self,Vhat,alpha,Lambda):
    #    Vhat = np.fft.fftn(v)
        divhat = np.zeros((self.Nnod,self.Nnod,self.Nnod))
        kz = np.zeros((self.Nnod,self.Nnod,self.Nnod))
        ky = np.zeros((self.Nnod,self.Nnod,self.Nnod))
        kx = np.zeros((self.Nnod,self.Nnod,self.Nnod)) 
        dis = np.zeros((self.Nnod,self.Nnod,self.Nnod),dtype=complex)
        f_alpha = -1.0/sc.hyp2f1(-alpha,(3+2*alpha-1)/2,3/2,1)
        freq = fftfreq(self.Nnod,d=1/self.Nnod)*complex(0,1)
        for i in range(self.Nnod):
            for j in range(self.Nnod):
                for k in range(self.Nnod):
                    dis[i,j,k] = np.sqrt(abs(freq[i])**2 + abs(freq[j])**2 + abs(freq[k])**2)
        frac_L = f_alpha*((Lambda**2 + dis**2)**(2*alpha/2)*sc.hyp2f1(-alpha,(3+2*alpha-1)/2,3/2,dis**2/(dis**2+Lambda**2))-Lambda**(2*alpha))
        divhat = frac_L * Vhat
        diverz_V = np.real(ifftn(divhat))
        return diverz_V
    
    def Tempered_Fractional_Laplacian(self,alpha,Lambda,nu):
        tau = (6*nu+1)/2
        #nu = 0.000185
        mu = (2*alpha+Lambda)*tau**(alpha-1)*nu**(alpha)
        vx = self.vxhat[:]
        vy = self.vyhat[:]
        vz = self.vzhat[:]
        s_fL_x = -mu * self.Tempered_Fractional_Laplacian_2(vx,alpha,Lambda)
        s_fL_y = -mu * self.Tempered_Fractional_Laplacian_2(vy,alpha,Lambda)
        s_fL_z = -mu * self.Tempered_Fractional_Laplacian_2(vz,alpha,Lambda)
        return s_fL_x,s_fL_y,s_fL_z

#%%
class SIM (Pre_Models):
    
    def box_filter_sim(self,VV,R):
     #   redsz = int((self.Nnod)/(2*R))
    #   Ix = np.zeros(redsz)
        vxbar = np.zeros((self.Nnod,self.Nnod,self.Nnod))
        kernel = np.ones((2*R+1,2*R+1,2*R+1))
        
        vxbar = convolve(VV , kernel, boundary='wrap') #signal.convolve(vx3 , kernel, mode='same', boundary='wrap')/(2*R+1)**3
        vx3 = vxbar[:]
        return vx3
    
    def SIM (self,vx,vy,vz,R):
        vxbar2 = self.box_filter_sim(vx,R)
        vybar2 = self.box_filter_sim(vy,R)
        vzbar2 = self.box_filter_sim(vz,R)
        
        sxxbar2 = self.box_filter_sim(vx*vx,R)- vxbar2*vxbar2
        sxybar2 = self.box_filter_sim(vx*vy,R)- vxbar2*vybar2
        sxzbar2 = self.box_filter_sim(vx*vz,R)- vxbar2*vzbar2
        syzbar2 = self.box_filter_sim(vy*vz,R)- vybar2*vzbar2
        syybar2 = self.box_filter_sim(vy*vy,R)- vybar2*vybar2
        szzbar2 = self.box_filter_sim(vz*vz,R)- vzbar2*vzbar2
        
        SIM_div_x_div = div(sxxbar2,sxybar2,sxzbar2,self.Nnod)
        SIM_div_y_div = div(sxybar2,syybar2,syzbar2,self.Nnod)
        SIM_div_z_div = div(sxzbar2,syzbar2,szzbar2,self.Nnod)
        return SIM_div_x_div, SIM_div_y_div, SIM_div_z_div