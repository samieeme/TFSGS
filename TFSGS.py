# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:28:48 2019

@author: samieeme
"""
import numpy as np
from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period
from Outputs import Output_Corr

#del( time, Rs, filename, fileout, corr_smg, corr_fsgs)

time = input("Time:  ")
Rs = input("filter width:  ")


filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/DNS-data/Nektar/20"
#filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/DNS-data/t-1000/DNS_"+ Rs
#filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/DNS-data/Nektar/20/"
fileout = 'C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/Semester 9/research/DNS-data/Nektar/20/'



solver = Output_Corr(filename,Rs,time)

corr_smg = solver.SMG_Model()
np.savetxt(fileout+'output-time-'+time+'-R'+Rs+'-SMG.csv', corr_smg)


nu = 1.0/1600
alpha1 = 0.80
corr_fsgs = solver.FSGS_Model(alpha1,nu)
np.savetxt(fileout+'output-time-'+time+'-R'+Rs+'-FSGS-alpha-'+str(int(alpha1*100))+'.csv', corr_fsgs)


#alpha2 = 0.75
#Lambda2 = 10
#corr_tfsgs = solver.TFSGS_Model(alpha2,Lambda2)

