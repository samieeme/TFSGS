# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:28:48 2019

@author: samieeme
"""
import numpy as np
from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period
from Outputs import Output_Corr
import matplotlib.pyplot as plt 

#del( time, Rs, filename, fileout, corr_smg, corr_fsgs)

time = str(20) #input("Time:  ")
Rs = "1" #input("filter width:  ")


##filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/DNS-data/Nektar/20"
filename= "/home/zayern/Desktop/Projects/TFSGS/data/DNS_Ali/Sim_1-IC_25/Out_1/tmprr_fltrd_DNS"

solver = Output_Corr(filename,Rs)

corr_smg = solver.SMG_Model()
#np.savetxt(fileout+'output-time-'+time+'-R'+Rs+'-SMG.csv', corr_smg)


nu = 0.000181 #1.0/1600




#%%    
alpha1 = [0.39] #np.linspace(0,1,21) #[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
#alpha1[0] = 0.01
#alpha1[21] = 0.99
#alpha1 = [0.9]
for i in alpha1:
    corr_fsgs,test = solver.FSGS_Model(i,nu)
    corr_tfsgs = solver.TFSGS_Model(i,0.05,nu,2)
#    np.savetxt(fileout+'output-time-'+time+'-R'+Rs+'-FSGS-alpha-'+str(int(i*100))+'.csv', corr_fsgs)
    # Lnn = solver.Two_PointCorr()
    # Lnn, Lsmg, Lfsgs = solver.Two_PointCorr()
    
# #%%
# #alpha2 = 0.99
# #Lambda2 = 100
# #corr_tfsgs = solver.TFSGS_Model(alpha2,Lambda2)


# ns = Lnn.shape[0]

# R = np.arange(ns)/ns*2*np.pi

# plt.plot(R,Lnn) 
# #plt.plot(R,Lsmg/10) 
# #plt.plot(R,Lfsgs/500) 



