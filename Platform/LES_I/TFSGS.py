# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:28:48 2019

@author: samieeme
"""
import numpy as np
from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period
from Outputs import Output_Corr
import matplotlib.pyplot as plt 
import os
import sys
import time

Rs = sys.argv[1]
add_in = sys.argv[2]
filename = sys.argv[3:12]
fileout = sys.argv[12]
Rs = filename[0][8]
print(add_in)
print(filename)

solver = Output_Corr(add_in,filename,Rs)
corr_smg = solver.SMG_Model()
np.savetxt(fileout+"/"+"output"+"-R"+Rs+"-SMG.csv", corr_smg)


nu = 0.001 #1.0/1600




#%%     
alpha1 = [0.51,0.53,0.55,0.57,0.6,0.63,0.65,0.67,0.7,0.73,0.75,0.77,0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.97,0.99,0.9999]
#alpha1[0] = 0.01
#alpha1[21] = 0.99
#alpha1 = [0.9]
temp_exp = [0.000000000001, 0.01, 0.05, 0.1, 0.4, 0.8, 1, 2, 4, 10]
for j in temp_exp:
    for i in alpha1:
      #  corr_fsgs,test = solver.FSGS_Model(i,nu)
        corr_tfsgs = solver.TFSGS_Model(i,j,nu,2)
      #  np.savetxt(fileout+"/"+"output"+"-R"+Rs+"-FSGS-alp-"+str(int(i*100))+".csv", corr_fsgs)
        np.savetxt(fileout+"/"+"output"+"-R"+Rs+"-TFSGS-alp-"+str(int(i*1000))+"-lmb-"+str(int(j*100))+".csv", corr_tfsgs)


# print(corr_smg)
# print(corr_fsgs)
# print(corr_tfsgs)      



