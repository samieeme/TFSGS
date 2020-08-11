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
from Solve_Module import Solver
from functions_filtering import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, Find_Neighboar

def data_reshape(add_in,Np,N):
    v = loadmat(add_in)
    list = []
    for i in v.keys():
        list.append(i)
    for i in range(3):
        del(v[list[i]])   
    u1 = v[list[3]].reshape((Np,N,N))
    u2 = v[list[4]].reshape((Np,N,N))
    u3 = v[list[5]].reshape((Np,N,N))
    return u1, u2, u3


N = 320
Np = int(320/40)
U = np.zeros((Np,N,N))
V = np.zeros((Np,N,N))
W = np.zeros((Np,N,N))
add_in = "/home/zayern/Desktop/Projects/TFSGS/data/DNS_Ali/Sim_1-IC_25/Out_1/" 
xlist = np.linspace(0, 2*np.pi, N)
ylist = np.linspace(0, 2*np.pi, N)

U, V, W = data_reshape(add_in+"Vel320-p_0.mat", Np, N)
#%%
for files in range(1,40): #add_in[1:]
    u1, u2, u3 = data_reshape(add_in+"Vel320-p_"+str(files)+".mat", Np, N)   
    U = np.concatenate((U,u1), axis=0)
    V = np.concatenate((V,u2), axis=0)
    W = np.concatenate((W,u3), axis=0)

U = np.transpose(U)
V = np.transpose(V)
W = np.transpose(W)

cp = plt.contourf(ylist, xlist, U[:,:,160])

#%%

os.system("mkdir /home/zayern/Desktop/Projects/TFSGS/data/DNS_Ali/Sim_1-IC_25/Out_1/temp_L-2")


#fileR=   #"/mnt/home/samieeme/FSGS_paper/Decaying/DNS/"


fileR_out="/home/zayern/Desktop/Projects/TFSGS/data/DNS_Ali/Sim_1-IC_25/Out_1/temp_L-2/"




#%%
# time = sys.argv[1]
# filename=fileR+"Up"+str(time)+".csv"

# os.system("mkdir /mnt/home/samieeme/FSGS_paper/Decaying/Filtered/"+str(time)+"/")
# #os.system("rm -rf "+str(time))
# #os.system("mkdir "+str(time))


# filename_out = fileR_out

for R in range(1,2):
    solver = Solver(fileR_out,U,V,W,R)
    vxbar, vybar, vzbar, sxx, sxy, sxz, syy, syz, szz, redsz, rrr=solver.get_bar_box()
    solver.get_outfile(fileR_out,1,R)

#%%
ss = np.size(vxbar,axis=0)
xlist = np.linspace(0, 2*np.pi, ss)
ylist = np.linspace(0, 2*np.pi, ss)
cp = plt.contourf(ylist, xlist, vybar[20,:,:])







