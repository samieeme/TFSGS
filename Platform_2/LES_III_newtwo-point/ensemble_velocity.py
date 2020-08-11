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


LES_dir =os.getcwd()
os.chdir('../../Filtered_Data')
DNS_root=os.getcwd()
Re_files=os.listdir(DNS_root)
DNS_Vel_Stress_Corr = []
SMG_Vel_Stress_Corr = []
VX = []
VY = []
VZ = []


for i in range(20):
    os.chdir('./'+Re_files[i])
    add_in=os.getcwd()
    os.system('pwd')
    os.chdir('../')
    
    Rs = str(4)
    solver = Output_Corr(add_in,Rs)
    vx,vy,vz, Nsize = solver.Vel_Out()
    VX.append(vx);VY.append(vy);VZ.append(vz);

    
vxv = (sum(VX)/20).reshape(Nsize**3); vyv = (sum(VY)/20).reshape(Nsize**3); vzv = (sum(VZ)/20).reshape(Nsize**3) 
np.savetxt('VX_Ensemble.csv',vxv)
np.savetxt('VY_Ensemble.csv',vyv)
np.savetxt('VZ_Ensemble.csv',vzv)

for j in range(100):
    VZ = []
    for i in range(20):
        os.chdir('./'+Re_files[i])
        print(Re_files[i])
        add_in=os.getcwd()
        os.chdir('../')
        name_V = "/prime_"+str(j)+"_vz-R"+Rs+".csv" 
        Vz = np.genfromtxt(add_in+"/"+name_V)
        print(add_in+"/"+name_V)
        Vbar = Vz.reshape(Nsize,Nsize,Nsize)
        VZ.append(Vbar)
        
    vzv = (sum(VZ)/20).reshape(Nsize**3);
    np.savetxt('Prime_Vz_'+str(j)+'_Ensemble.csv',vzv)    



os.chdir(LES_dir)  
os.chdir('../../Filtered_Data')

        

os.chdir(LES_dir)   



#%%

# add_in = #sys.argv[2]
# filename = sys.argv[3:12]
# print(add_in)
# print(filename)

# start = time.time()
# solver = Output_Corr(add_in,filename,Rs)
# corr_smg = solver.SMG_Model()
# #np.savetxt(fileout+'output-time-'+time+'-R'+Rs+'-SMG.csv', corr_smg)


# nu = 0.001 #1.0/1600




# #%%    
# alpha1 = [0.65] #np.linspace(0,1,21) #[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
# #alpha1[0] = 0.01
# #alpha1[21] = 0.99
# #alpha1 = [0.9]

# for i in alpha1:
#     corr_fsgs,test = solver.FSGS_Model(i,nu)
#     corr_tfsgs = solver.TFSGS_Model(i,0.1,nu,2)
# #    np.savetxt(fileout+'output-time-'+time+'-R'+Rs+'-FSGS-alpha-'+str(int(i*100))+'.csv', corr_fsgs)
#     # Lnn = solver.Two_PointCorr()
#     # Lnn, Lsmg, Lfsgs = solver.Two_PointCorr()
# print(time.time()-start)
# print(corr_smg)
# print(corr_fsgs)
# print(corr_tfsgs)    
# # #%%
# # #alpha2 = 0.99
# # #Lambda2 = 100
# # #corr_tfsgs = solver.TFSGS_Model(alpha2,Lambda2)


# # ns = Lnn.shape[0]

# # R = np.arange(ns)/ns*2*np.pi

# # plt.plot(R,Lnn) 
# # #plt.plot(R,Lsmg/10) 
# # #plt.plot(R,Lfsgs/500) 



