# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:28:48 2019

@author: samieeme
"""
import numpy as np
from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period
from Outputs import Output_Corr
import matplotlib.pyplot as plt 
from matplotlib import rc
import os
import sys
import time
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


#%%
LES_dir =os.getcwd()
os.chdir('../../Filtered_Data')
DNS_root=os.getcwd()
Re_files=os.listdir(DNS_root)
DNS_Vel_Stress_Corr = []
SMG_Vel_Stress_Corr = []
TFSGS_Vel_Stress_Corr1 = []
TFSGS_Vel_Stress_Corr2 = []
TFSGS_Vel_Stress_Corr3 = []
TFSGS_Vel_Stress_Corr4 = []
TFSGS_Vel_Stress_Corr5 = []
TFSGS_Vel_Stress_Corr6 = []
TFSGS_Vel_Stress_Corr7 = []
VXV = np.genfromtxt('VX_Ensemble.csv')
VYV = np.genfromtxt('VY_Ensemble.csv')
VZV = np.genfromtxt('VZ_Ensemble.csv')

N = int((VXV.shape[0])**(1/3))+1
nu = 0.001 #1.0/1600
vx_ens=VXV.reshape(N,N,N); vy_ens=VYV.reshape(N,N,N); vz_ens=VZV.reshape(N,N,N)
Vel_av = []
Diss_SMG=[]
Diss1_TFSGS=[]
Diss2_TFSGS=[]
Diss3_TFSGS=[]
Diss4_TFSGS=[]
Diss5_TFSGS=[]
Diss6_TFSGS=[]
Diss7_TFSGS=[]
Diss_DNS=[]
alpha = 0.63
lambdaa = 0.1
for i in range(20):
    os.chdir('./'+Re_files[i])
    add_in=os.getcwd()
    os.system('pwd')
    os.chdir('../')
    
    Rs = str(4) #(0.635,0.07,nu,2)(0.65,0.1,nu,2); (0.68,0.00000001,nu,2); [(- DDNS_M) / DTFSGS1_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.0115]
    solver = Output_Corr(add_in,Rs)
    solver.V_ZeroMean(vx_ens,vy_ens,vz_ens)
    corr_smg,diss_SMG,diss_DNS = solver.SMG_Model()
    # corr1_tfsgs, diss1_TFSGS = solver.TFSGS_Model(0.61,0.0000001,nu,2)
    # LFSGS1 = solver.Two_PointCorr_TFSGS()
    # # C_SMG = diss_DNS/diss_SMG/(2*int(Rs)**2*(2*np.pi/310)**2)*0.016
    # corr2_tfsgs,diss2_TFSGS = solver.TFSGS_Model(0.61,0.05,nu,2)
    # LFSGS2 = solver.Two_PointCorr_TFSGS()
    # corr3_tfsgs,diss3_TFSGS = solver.TFSGS_Model(0.61,0.6,nu,2)
    # LFSGS3 = solver.Two_PointCorr_TFSGS()
    # corr4_tfsgs,diss4_TFSGS = solver.TFSGS_Model(0.61,3,nu,2)
    # LFSGS4 = solver.Two_PointCorr_TFSGS()
    # corr5_tfsgs,diss5_TFSGS = solver.TFSGS_Model(0.75,0.0000000001,nu,2)
    # LFSGS5 = solver.Two_PointCorr_TFSGS()
    # corr6_tfsgs,diss6_TFSGS = solver.TFSGS_Model(0.68,2,nu,2)
    # LFSGS6 = solver.Two_PointCorr_TFSGS()
    # corr7_tfsgs,diss7_TFSGS = solver.TFSGS_Model(0.61,2,nu,2)
    # LFSGS7 = solver.Two_PointCorr_TFSGS()
    
    Lnn, Lsmg = solver.Two_PointCorr_main()
    if Lnn>0 :
        DNS_Vel_Stress_Corr.append(Lnn)
    SMG_Vel_Stress_Corr.append(Lsmg)
    # TFSGS_Vel_Stress_Corr1.append(LFSGS1)
    # TFSGS_Vel_Stress_Corr2.append(LFSGS2)
    # TFSGS_Vel_Stress_Corr3.append(LFSGS3)
    # TFSGS_Vel_Stress_Corr4.append(LFSGS4)
    # TFSGS_Vel_Stress_Corr5.append(LFSGS5)
    # TFSGS_Vel_Stress_Corr6.append(LFSGS6)
    # TFSGS_Vel_Stress_Corr7.append(LFSGS7)
    
    vvx,vvy,vvz,Nsize = solver.Vel_Out()
    Vel_av.append(vvx)
    Diss_DNS.append(diss_DNS)
    Diss_SMG.append(diss_SMG)
    # Diss1_TFSGS.append(diss1_TFSGS)
    # Diss2_TFSGS.append(diss2_TFSGS)
    # Diss3_TFSGS.append(diss3_TFSGS)
    # Diss4_TFSGS.append(diss4_TFSGS)
    # Diss5_TFSGS.append(diss5_TFSGS)
    # Diss6_TFSGS.append(diss6_TFSGS)
    # Diss7_TFSGS.append(diss7_TFSGS)
#%%
print(np.max(sum(Vel_av)))
ns = len(DNS_Vel_Stress_Corr)

DDNS_M = sum(Diss_DNS); DSMG_M = sum(Diss_SMG); 
DTFSGS1_M = sum(Diss1_TFSGS); DTFSGS2_M = sum(Diss2_TFSGS); DTFSGS3_M = sum(Diss3_TFSGS);
DTFSGS4_M = sum(Diss4_TFSGS); DTFSGS5_M = sum(Diss5_TFSGS); DTFSGS6_M = sum(Diss6_TFSGS); DTFSGS7_M = sum(Diss7_TFSGS); 

#%%
Lnn = DNS_Vel_Stress_Corr
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
alpha = 0.63
lambdaa = 0.1
Nmax = 15
Nmin = 0
# Lnn = sum(DNS_Vel_Stress_Corr[:])/10#/abs(DDNS_M)
# # Lnn=Lnn1-Lnn1[0]
R = np.arange(ns)
# Lsmg = sum(SMG_Vel_Stress_Corr)/10#/abs(DSMG_M) #SMG_Vel_Stress_Corr[9] #
# LFSGS1 = sum(TFSGS_Vel_Stress_Corr1)/10#/abs(DTFSGS1_M)/10
# LFSGS2 = sum(TFSGS_Vel_Stress_Corr2)/10#/abs(DTFSGS2_M)/10
# LFSGS3 = sum(TFSGS_Vel_Stress_Corr3)/10#/abs(DTFSGS3_M)/10
# LFSGS4 = sum(TFSGS_Vel_Stress_Corr4)/10#/abs(DTFSGS4_M)/10
# LFSGS5 = sum(TFSGS_Vel_Stress_Corr5)/10#/abs(DTFSGS5_M)/10
# LFSGS6 = sum(TFSGS_Vel_Stress_Corr6)/abs(DTFSGS6_M)/10
# LFSGS7 = sum(TFSGS_Vel_Stress_Corr7)/abs(DTFSGS7_M)/10
C_DNS = 1.0/Lnn[0]
# C_SMG = 1/Lsmg[0]
# C_alpha_1 = 1/LFSGS1[0] #(- DDNS_M) / DTFSGS1_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.015/5/1.10
# C_alpha_2 = 1/LFSGS2[0] #(- DDNS_M) / DTFSGS2_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.015/5/1.10
# C_alpha_3 = 1/LFSGS3[0] #(- DDNS_M) / DTFSGS3_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.015/5/1.10
# C_alpha_4 = 1/LFSGS4[0] #(- DDNS_M) / DTFSGS4_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.015/5/1.10

fig = plt.figure(figsize=(15,12))
font = {'family':'serif','color':'black',
        'weight': 'normal',
        'size': 30,
        }

plt.xlabel('$r/\Delta$',fontsize=40)
plt.ylabel('$ S_{\\Delta\\Delta} $',fontsize=40)



# plt.plot(R[Nmin:Nmax],C_DNS*Lnn[Nmin:Nmax],color='black', linestyle='solid', linewidth=3, marker='*',
#       markerfacecolor='black', markersize=13, label='Filtered DNS') 
# plt.plot(R[Nmin:Nmax],C_SMG*Lsmg[Nmin:Nmax], color='blue', linestyle='-.', linewidth=3, marker='+',
#       markerfacecolor='green', markersize=16, label='SMG Model') 

# plt.plot(R[Nmin:Nmax],C_alpha_1*LFSGS1[Nmin:Nmax], color='maroon', linestyle='dashed', linewidth=3, marker='s',
#         markerfacecolor='maroon', markersize=11, label='TFSGS, $\\alpha = $'+str(0.6)+', $\\lambda = $'+str(0))
# plt.plot(R[Nmin:Nmax],C_alpha_2*LFSGS2[Nmin:Nmax], color='crimson', linestyle='dashed', linewidth=3, marker='v',
#       markerfacecolor='crimson', markersize=11, label='TFSGS, $\\alpha = $'+str(0.6)+', $\\lambda = $'+str(0.01))
# plt.plot(R[Nmin:Nmax],C_alpha_3*LFSGS3[Nmin:Nmax], color='salmon', linestyle='dashed', linewidth=3, marker='o',
#         markerfacecolor='salmon', markersize=11, label='TFSGS, $\\alpha = $'+str(0.6)+', $\\lambda = $'+str(0.07))
# plt.plot(R[Nmin:Nmax],C_alpha_4*LFSGS4[Nmin:Nmax], color='teal', linestyle='dashed', linewidth=3, marker='d',
#         markerfacecolor='teal', markersize=11, label='TFSGS, $\\alpha = $'+str(0.6)+', $\\lambda = $'+str(0.5))


######################  Semi-Log X: Small r
plt.plot(R[Nmin:Nmax],Lnn[Nmin:Nmax],color='black', linestyle='solid', linewidth=3, marker='*',
      markerfacecolor='black', markersize=18, label='Filtered DNS') 
# plt.plot(R[Nmin:Nmax],C_SMG*Lsmg[Nmin:Nmax], color='blue', linestyle='-.', linewidth=3, marker='+',
#       markerfacecolor='green', markersize=16, label='SMG Model') 

# plt.plot(R[Nmin:Nmax],C_alpha_1*LFSGS1[Nmin:Nmax], color='maroon', linestyle='dashed', linewidth=3, marker='s',
#         markerfacecolor='maroon', markersize=10, label='TFSGS, $\\alpha = $'+str(0.6)+', $\\lambda = $'+str(0))
# plt.plot(R[Nmin:Nmax],C_alpha_2*LFSGS2[Nmin:Nmax], color='crimson', linestyle='dashed', linewidth=3, marker='v',
#       markerfacecolor='crimson', markersize=10, label='TFSGS, $\\alpha = $'+str(0.6)+', $\\lambda = $'+str(0.05))
# plt.plot(R[Nmin:Nmax],C_alpha_3*LFSGS3[Nmin:Nmax], color='salmon', linestyle='dashed', linewidth=3, marker='o',
#         markerfacecolor='salmon', markersize=10, label='TFSGS, $\\alpha = $'+str(0.6)+', $\\lambda = $'+str(0.6))
# plt.plot(R[Nmin:Nmax],C_alpha_4*LFSGS4[Nmin:Nmax], color='teal', linestyle='dashed', linewidth=3, marker='d',
#         markerfacecolor='teal', markersize=10, label='TFSGS, $\\alpha = $'+str(0.6)+', $\\lambda = $'+str(3))

######################  Semi-Log Y: Large r





################### Enlarge Balance Point
# balance_point = C_alpha_1*np.array([LFSGS1[5]])
# balance_R = np.array([R[5]])
# plt.plot(balance_R,balance_point,  marker='s',
#         markerfacecolor='maroon',markeredgewidth=0.0, markersize=23)
# balance_point = C_alpha_2*np.array([LFSGS2[5]])
# balance_R = np.array([R[5]])
# plt.plot(balance_R,balance_point,   marker='v',
#       markerfacecolor='crimson', markeredgewidth=0.0,markersize=23)
# balance_point = C_alpha_3*np.array([LFSGS3[4]])
# balance_R = np.array([R[4]])
# plt.plot(balance_R,balance_point,  marker='o',
#         markerfacecolor='salmon',markeredgewidth=0.0,  markersize=23)
# balance_point = C_alpha_4*np.array([ LFSGS4[3]])
# balance_R = np.array([R[3]])
# plt.plot(balance_R,balance_point, markeredgewidth=0.0, marker='d',
#         markerfacecolor='teal', markersize=23)
# # balance_point = np.array([ C_alpha_1*LFSGS1[4],C_alpha_2*LFSGS2[4],C_alpha_3*LFSGS3[3],C_alpha_4*LFSGS4[2]])
# # balance_R = np.array([R[4],R[4],R[3],R[2]])
# # plt.plot(balance_R,balance_point,linestyle='None', markeredgewidth=0,  marker='o', markerfacecolor='black', markersize=20)
###################3

plt.legend(loc='upper right', fontsize=25)

# plt.ylim(0.0025,0.005) #np.min(C_SMG*Lsmg[1:Nmax])/1.3,np.max(Lnn)*1.08)
plt.xlim(-0.1,Nmax+1)
plt.axes()
# gca().set_xticklabels(['']*10)
#plt.savefig()

os.chdir(LES_dir)





#%%
# plt.rc('xtick', labelsize=30)
# plt.rc('ytick', labelsize=30)
# alpha = 0.63
# lambdaa = 0.1
# Nmax = 20
# Nmin = 1
# Lnn = sum(DNS_Vel_Stress_Corr[:])/10#/abs(DDNS_M)
# # Lnn=Lnn1-Lnn1[0]
# R = np.arange(ns)/ns*2*np.pi
# Lsmg = sum(SMG_Vel_Stress_Corr)/10#/abs(DSMG_M) #SMG_Vel_Stress_Corr[9] #
# LFSGS1 = sum(TFSGS_Vel_Stress_Corr1)/10#/abs(DTFSGS1_M)/10
# LFSGS2 = sum(TFSGS_Vel_Stress_Corr2)/10#/abs(DTFSGS2_M)/10
# LFSGS3 = sum(TFSGS_Vel_Stress_Corr3)/10#/abs(DTFSGS3_M)/10
# LFSGS4 = sum(TFSGS_Vel_Stress_Corr4)/10#/abs(DTFSGS4_M)/10
# LFSGS5 = sum(TFSGS_Vel_Stress_Corr5)/10#/abs(DTFSGS5_M)/10
# # LFSGS6 = sum(TFSGS_Vel_Stress_Corr6)/abs(DTFSGS6_M)/10
# # LFSGS7 = sum(TFSGS_Vel_Stress_Corr7)/abs(DTFSGS7_M)/10

# C_SMG = DDNS_M / DSMG_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.02/5
# C_alpha_1 = (- DDNS_M) / DTFSGS1_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.015/5/1.10
# C_alpha_2 = (- DDNS_M) / DTFSGS2_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.015/5/1.10
# C_alpha_3 = (- DDNS_M) / DTFSGS3_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.015/5/1.10
# C_alpha_4 = (- DDNS_M) / DTFSGS4_M /(2*int(Rs)**2*(2*np.pi/310)**2)*0.015/5/1.10

# fig = plt.figure(figsize=(15,12))
# font = {'family':'serif','color':'black',
#         'weight': 'normal',
#         'size': 30,
#         }

# plt.xlabel('$r/\Delta$',fontsize=40)
# plt.ylabel('$ G_{\\Delta\\Delta} $',fontsize=40)



# plt.plot(R[Nmin:Nmax],Lnn[Nmin:Nmax],color='black', linestyle='solid', linewidth=3, marker='*',
#       markerfacecolor='black', markersize=13, label='Filtered DNS') 
# plt.plot(R[Nmin:Nmax],C_SMG*Lsmg[Nmin:Nmax], color='blue', linestyle='-.', linewidth=3, marker='+',
#       markerfacecolor='green', markersize=16, label='SMG Model') 

# plt.plot(R[Nmin:Nmax],C_alpha_1*LFSGS1[Nmin:Nmax], color='maroon', linestyle='dashed', linewidth=3, marker='s',
#         markerfacecolor='maroon', markersize=11, label='TFSGS, $\\alpha = $'+str(0.7)+', $\\lambda = $'+str(0))
# plt.plot(R[Nmin:Nmax],C_alpha_2*LFSGS2[Nmin:Nmax], color='crimson', linestyle='dashed', linewidth=3, marker='v',
#       markerfacecolor='crimson', markersize=11, label='TFSGS, $\\alpha = $'+str(0.7)+', $\\lambda = $'+str(0.04))
# plt.plot(R[Nmin:Nmax],C_alpha_3*LFSGS3[Nmin:Nmax], color='salmon', linestyle='dashed', linewidth=3, marker='o',
#         markerfacecolor='salmon', markersize=11, label='TFSGS, $\\alpha = $'+str(0.7)+', $\\lambda = $'+str(0.1))
# plt.plot(R[Nmin:Nmax],C_alpha_4*LFSGS4[Nmin:Nmax], color='teal', linestyle='dashed', linewidth=3, marker='d',
#         markerfacecolor='teal', markersize=11, label='TFSGS, $\\alpha = $'+str(0.7)+', $\\lambda = $'+str(0.5))
# ###################3
# balance_point = C_alpha_1*np.array([LFSGS1[5]])
# balance_R = np.array([R[5]])
# plt.plot(balance_R,balance_point,  marker='s',
#         markerfacecolor='maroon',markeredgewidth=0.0, markersize=23)
# balance_point = C_alpha_2*np.array([LFSGS2[5]])
# balance_R = np.array([R[5]])
# plt.plot(balance_R,balance_point,   marker='v',
#       markerfacecolor='crimson', markeredgewidth=0.0,markersize=23)
# balance_point = C_alpha_3*np.array([LFSGS3[4]])
# balance_R = np.array([R[4]])
# plt.plot(balance_R,balance_point,  marker='o',
#         markerfacecolor='salmon',markeredgewidth=0.0,  markersize=23)
# balance_point = C_alpha_4*np.array([ LFSGS4[3]])
# balance_R = np.array([R[3]])
# plt.plot(balance_R,balance_point, markeredgewidth=0.0, marker='d',
#         markerfacecolor='teal', markersize=23)
# # balance_point = np.array([ C_alpha_1*LFSGS1[4],C_alpha_2*LFSGS2[4],C_alpha_3*LFSGS3[3],C_alpha_4*LFSGS4[2]])
# # balance_R = np.array([R[4],R[4],R[3],R[2]])
# # plt.plot(balance_R,balance_point,linestyle='None', markeredgewidth=0,  marker='o', markerfacecolor='black', markersize=20)
# ###################3

# plt.legend(loc='lower left', fontsize=25)

# # plt.ylim(0.0025,0.005) #np.min(C_SMG*Lsmg[1:Nmax])/1.3,np.max(Lnn)*1.08)
# # plt.xlim(0.15,1.0)
# plt.axes()
# # gca().set_xticklabels(['']*10)
# #plt.savefig()

# os.chdir(LES_dir)


