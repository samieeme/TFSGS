# in the name of God
# functions in filtering function

from IPython import get_ipython
#get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
from math import pi, gamma
from astropy.convolution import convolve
import matplotlib.pyplot as plt
import os
import sys
from scipy.io import loadmat
from functions_filtering import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, Find_Neighboar
from Solve_Module import Solver

def data_reshape(add_in,Np):
    v = loadmat(add_in)
    list = []
    for i in v.keys():
        list.append(i)
    for i in range(3):
        del(v[list[i]])
    N = int(np.sqrt(np.size(v[list[3]])/Np))
    u1 = v[list[3]].reshape((Np,N,N))
    u2 = v[list[4]].reshape((Np,N,N))
    u3 = v[list[5]].reshape((Np,N,N))
    return u1, u2, u3, N


#%%
root = sys.argv[1]
add_in = sys.argv[2:] #"/home/zayern/Desktop/Projects/TFSGS/data/DNS_Ali/Sim_1-IC_25/Out_1/" 
Nprocc = 40 #int(sys.argv[0])-1
 
Np = int(320/Nprocc)
# U = np.zeros((Np,N,N))
# V = np.zeros((Np,N,N))
# W = np.zeros((Np,N,N))
#print(np.size(add_in))
U, V, W, N = data_reshape(root+"/"+"Vel320-p_0.mat", Np)

for files in range(Nprocc):
    #print("Vel320-p_"+str(files)+".mat")
    u1, u2, u3, N = data_reshape(root+"/"+"Vel320-p_"+str(files)+".mat", Np)   
    U = np.concatenate((U,u1), axis=0)
    V = np.concatenate((V,u2), axis=0)
    W = np.concatenate((W,u3), axis=0)
    
U = np.transpose(U)
V = np.transpose(V)
W = np.transpose(W)
    
fileR_out=add_in[Nprocc]+"/"


# filename_out = fileR_out

for R in range(2,3):
    print(R)
    solver = Solver(fileR_out,U,V,W,R)
    vxbar, vybar, vzbar, sxx, sxy, sxz, syy, syz, szz, redsz, rrr=solver.get_bar_box()
    solver.get_outfile(fileR_out,1,R)





