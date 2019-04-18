# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:28:48 2019

@author: samieeme
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi


def box_filter2 (vx,matsz,R):
    
    redsz = int((matsz-1)/(2*R))+1
#    Ix = np.zeros(redsz)
    vbar = np.zeros((redsz,redsz,redsz))
    for i in range(redsz):
        for j in range(redsz):
            for p in range(redsz):
                for it1 in range(1,2*R+2):
                    for it2 in range(1,2*R+2):
                        for it3 in range(1,2*R+2):
                            Ib = remain((i-1)*2*R+1+it1-1-R,matsz-1)
                            Jb = remain((j-1)*2*R+1+it2-1-R,matsz-1)
                            Pb = remain((p-1)*2*R+1+it3-1-R,matsz-1)
                            vbar[i,j,p] += vx[Ib,Jb,Pb]
    vbar = vbar / (2.*R+1.)**3
    return vbar , redsz

def remain (a,b):
    c = a%b
    if c == 0: c = b
    return c

class Solver(object):
    
    def __init__(self,filename):
        data = np.loadtxt(filename)
        prsz = data.shape[0]
        matsz = int(np.ceil(prsz ** (1./3.)))
        self.matsz = matsz
        
        vx = np.ones(prsz)
        vx = data[:,0]
        vy = data[:,1]
        vz = data[:,2]
        
        self.vxm = vx.reshape(matsz,matsz,matsz)
        self.vym = vy.reshape(matsz,matsz,matsz)
        self.vzm = vz.reshape(matsz,matsz,matsz)
        
        self.vxx = self.vxm * self.vxm
        self.vxy = self.vxm * self.vym
        self.vyy = self.vym * self.vym
        self.vzz = self.vzm * self.vzm
        self.vxz = self.vxm * self.vzm
        self.vyz = self.vym * self.vzm


        
    def box_filter(self,R):
        redsz = int((self.matsz-1)/(2*R))+1
    #   Ix = np.zeros(redsz)
        vxbar = np.zeros((redsz,redsz,redsz))
        vybar = np.zeros((redsz,redsz,redsz))
        vzbar = np.zeros((redsz,redsz,redsz))
        vxxx = np.zeros((redsz,redsz,redsz))
        for i in range(redsz):
            for j in range(redsz):
                for p in range(redsz):
                    for it1 in range(1,2*R+2):
                        Ib = remain((i-1)*2*R+1+it1-1-R,self.matsz-1)
                        for it2 in range(1,2*R+2):
                            Jb = remain((j-1)*2*R+1+it2-1-R,self.matsz-1)
                            for it3 in range(1,2*R+2):
                                Pb = remain((p-1)*2*R+1+it3-1-R,self.matsz-1)
#                                vxbar[i,j,p] += self.vxm[Ib,Jb,Pb]
#                                vybar[i,j,p] += self.vym[Ib,Jb,Pb]
#                                vzbar[i,j,p] += self.vzm[Ib,Jb,Pb]
                                vxbar[i,j,p] += self.vxm[Ib,Jb,Pb]
                                vybar[i,j,p] += self.vym[Ib,Jb,Pb]
                                vzbar[i,j,p] += self.vzm[Ib,Jb,Pb]
        vxmx =self.vxm
        vxbar = vxbar / (2.*R+1.)**3
        vybar = vybar / (2.*R+1.)**3
        vzbar = vzbar / (2.*R+1.)**3
        return  vxbar , vybar, vzbar, redsz , vxmx, vxxx 
            
    def get_bar(self):
        self.vxbar , self.vybar, self.vzbar, self.redsz, self.vxmx, vxxx = self.box_filter(2)

    def plot(self,R):
        redsz = int((self.matsz-1)/(2*R))+1
        x = np.linspace(0, 2*pi, self.matsz)
        y = np.linspace(0, 2*pi, self.matsz)
        
        X, Y = np.meshgrid(x, y)
        sx = self.vxm[0]
         
        plt.contourf(X, Y, sx, 50, cmap='RdGy')
        plt.colorbar();
        plt.show()
        
        x = np.linspace(0, 2*pi, redsz)
        y = np.linspace(0, 2*pi, redsz)
        
        X, Y = np.meshgrid(x, y)
        sx = self.vxbar[0]
         
        plt.contourf(X, Y, sx, 50, cmap='RdGy')
        plt.colorbar()
        plt.show()

filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/April-15-2019/data_DNS/DrJaberi/DNS3Duvw.dat"
R = 2
solver = Solver(filename)


R = 2
vxbar , vybar, vzbar, redsz, vxmx,vxxx  = solver.box_filter(R)
aaa = solver.get_bar()
solver.plot(R)
#
vxbar2, sx = box_filter2 (vxmx,129,2)
#
#
#
#x = np.linspace(0, 2*pi, 33)
#y = np.linspace(0, 2*pi, 33)
#
#X, Y = np.meshgrid(x, y)
#sx = vxbar2[0]
# 
#plt.contourf(X, Y, sx, 50, cmap='RdGy')
#plt.colorbar()
#plt.show()
#
#x = np.linspace(0, 2*pi, 129)
#y = np.linspace(0, 2*pi, 129)
#
#X, Y = np.meshgrid(x, y)
#sx = vxmx[0]
# 
#plt.contourf(X, Y, sx, 50, cmap='RdGy')
#plt.colorbar()
#plt.show()