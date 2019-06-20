# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:08:16 2019

@author: samieeme
"""
import numpy as np
from math import pi, gamma
from astropy.convolution import convolve
from functions_filtering import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, Find_Neighboar

class Solver(object):
    
    def __init__(self,filename,R):
        data = np.loadtxt(filename)
        prsz = data.shape[0]
        matsz = int(np.ceil(prsz**(1./3.)))
        self.mainsz = matsz - remain(matsz,2*R) 
        
        vx = np.ones(prsz)
        vx = data[:,0]
        vy = data[:,1]
        vz = data[:,2]
        
        vxa = vx.reshape(matsz,matsz,matsz)
        vya = vy.reshape(matsz,matsz,matsz)
        vza = vz.reshape(matsz,matsz,matsz)        

        vx1m = vxa[range(0,self.mainsz),:,:]
        vx2m = vx1m[:,range(0,self.mainsz),:]
        self.vxm = vx2m[:,:,range(0,self.mainsz)]
        
        vx1m = vya[range(0,self.mainsz),:,:]
        vx2m = vx1m[:,range(0,self.mainsz),:]
        self.vym = vx2m[:,:,range(0,self.mainsz)]
        
        vx1m = vza[range(0,self.mainsz),:,:]
        vx2m = vx1m[:,range(0,self.mainsz),:]
        self.vzm = vx2m[:,:,range(0,self.mainsz)]
        
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
        self.sxxbar = self.sxxbar1  - self.vxbar*self.vxbar
        self.sxybar = self.sxybar1  - self.vxbar*self.vybar
        self.sxzbar = self.sxzbar1  - self.vxbar*self.vzbar
        self.syybar = self.syybar1  - self.vybar*self.vybar
        self.syzbar = self.syzbar1  - self.vybar*self.vzbar
        self.szzbar = self.szzbar1  - self.vzbar*self.vzbar
        return  self.vxbar, self.vybar, self.vzbar, self.sxxbar, self.sxybar, self.sxzbar, self.syybar, self.syzbar, self.szzbar, self.redsz, self.rrr

    def output_file(self,V):
        vo = V.reshape(self.redsz**3)
        return vo
    
    def get_outfile(self,fileout,time,R):
        R = self.rad
        vx = self.output_file(self.vxbar)
        np.savetxt(fileout+'velx-t'+str(time)+'-R'+str(R)+'.csv', vx)
        vy = self.output_file(self.vybar)
        np.savetxt(fileout+'vely-t'+str(time)+'-R'+str(R)+'.csv', vy)
        vz = self.output_file(self.vzbar)
        np.savetxt(fileout+'velz-t'+str(time)+'-R'+str(R)+'.csv', vz)
        sxx = self.output_file(self.sxxbar)
        np.savetxt(fileout+'Sxx-t'+str(time)+'-R'+str(R)+'.csv', sxx)
        sxy = self.output_file(self.sxybar)
        np.savetxt(fileout+'Sxy-t'+str(time)+'-R'+str(R)+'.csv', sxy)
        sxz = self.output_file(self.sxzbar)
        np.savetxt(fileout+'Sxz-t'+str(time)+'-R'+str(R)+'.csv', sxz)        
        syy = self.output_file(self.syybar)
        np.savetxt(fileout+'Syy-t'+str(time)+'-R'+str(R)+'.csv', syy)
        syz = self.output_file(self.syzbar)
        np.savetxt(fileout+'Syz-t'+str(time)+'-R'+str(R)+'.csv', syz)
        szz = self.output_file(self.szzbar)
        np.savetxt(fileout+'Szz-t'+str(time)+'-R'+str(R)+'.csv', szz)
#    def plot(self,R):
#        x = np.linspace(0, 2*pi, self.mainsz)
#        y = np.linspace(0, 2*pi, self.mainsz)
#        
#        X, Y = np.meshgrid(x, y)
#        sx = self.vxm[5]
#         
#        plt.contourf(X, Y, sx, 50, cmap='RdGy')
#        plt.colorbar();
#        plt.show()
#        
#        x = np.linspace(0, 2*pi, self.redsz)
#        y = np.linspace(0, 2*pi, self.redsz)
#        
#        X, Y = np.meshgrid(x, y)
#        sx = self.vxbar[1]
#         
#        plt.contourf(X, Y, sx, 50, cmap='RdGy')
#        plt.colorbar()
#        plt.show()