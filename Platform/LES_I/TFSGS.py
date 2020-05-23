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
print(add_in)
print(filename)

solver = Output_Corr(add_in,filename,Rs)
corr_smg = solver.SMG_Model()
np.savetxt(fileout+"/"+"output"+"-R"+Rs+"-SMG.csv", corr_smg)


nu = 0.001 #1.0/1600




#%%    
alpha1 = [0.99,0.89,0.79,0.69] #np.linspace(0,1,21) #[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
#alpha1[0] = 0.01
#alpha1[21] = 0.99
#alpha1 = [0.9]
temp_exp = 0.05
for i in alpha1:
    corr_fsgs,test = solver.FSGS_Model(i,nu)
    corr_tfsgs = solver.TFSGS_Model(i,temp_exp,nu,2)
    np.savetxt(fileout+"/"+"output"+"-R"+Rs+"-FSGS-alp-"+str(int(i*100))+".csv", corr_fsgs)
    np.savetxt(fileout+"/"+"output"+"-R"+Rs+"-TFSGS-alp-"+str(int(i*100))+"-lmb-"+str(temp_exp)+".csv", corr_tfsgs)


print(corr_smg)
print(corr_fsgs)
print(corr_tfsgs)    



