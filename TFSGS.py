# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:28:48 2019

@author: samieeme
"""
import numpy as np
from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period
from Outputs import Output_Corr
#import os
import sys
from IPython import get_ipython


get_ipython().magic('reset -sf')
#del( time, Rs, filename, fileout, corr_smg, corr_fsgs)


#%%
#time = str(100) #input("Time:  ")
#Rs = input("filter width:  ")
#Rs = sys.argv[1]

time_step = "900" #str(sys.argv[1])

for jj in range(1,16):
	Rs = str(jj)
	#filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/DNS-data/Nektar/20"
	#filename="/mnt/home/samieeme/FSGS_paper/t-"+time_step+"/DNS_"+ Rs
	filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/DNS-data/Nektar/20/"
# 	fileout = "/mnt/home/samieeme/FSGS_paper/t-"+time_step+"/DNS_"+ Rs +"/"
# 	filename="/mnt/home/samieeme//FSGS_paper/Decaying/Filtered/"+time_step
# 	fileout ="/mnt/home/samieeme/FSGS_paper/Decaying/Outputs/"+time_step+"/L_"+ Rs +"/"

	solver = Output_Corr(filename,Rs,time_step)

	corr_smg = solver.SMG_Model()
	np.savetxt(fileout+'output-time-'+time_step+'-R'+Rs+'-SMG.csv', corr_smg)

	nu = 0.000181 #1.0/1600
	alpha1 = np.linspace(0,1,21) #[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
	alpha1[0] = 0.05 
	alpha1[20] = 0.9999 
	#alpha1 = [0.9] 
	for i in alpha1: 
    		corr_fsgs,test = solver.FSGS_Model(i,nu)
    		np.savetxt(fileout+'output-time-'+time_step+'-R'+Rs+'-FSGS-alpha-'+str(int(i*100))+'.csv', corr_fsgs)



#%%
#alpha2 = 0.99
#Lambda2 = 100
#corr_tfsgs = solver.TFSGS_Model(alpha2,Lambda2)

