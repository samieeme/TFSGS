# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:11:40 2019

@author: samieeme
"""
from Models import Pre_Models, Solver_filtered_field_sep, SMG_Model, FSGS_Model, Tempered_FSGS, Solver_filtered_field_JHU
from functions import remain, deriv_x, deriv_y, deriv_z, div, get_TwoPointCorr, get_TwoPointCorr_DNS
import numpy as np
from scipy import stats
from math import gamma
   
class Output_Corr (object):
    
    def __init__(self,add_in,Rs):
        solver = Solver_filtered_field_sep(add_in,Rs) 
        #solver = Solver_filtered_field_sep(filename,Rs) 
        self.vxbar , self.vybar, self.vzbar, self.sxx, self.sxy, self.sxz, \
        self.syy, self.syz, self.szz, self.redsz = solver.Output()
        self.sx_div = div(self.sxx,self.sxy,self.sxz, self.redsz)
        self.sy_div = div(self.sxy,self.syy,self.syz, self.redsz)
        self.sz_div = div(self.sxz,self.syz,self.szz, self.redsz)

    
    def Vel_Out(self):
        return self.vxbar , self.vybar, self.vzbar, self.redsz
    
    def V_ZeroMean(self,vx,vy,vz):
        self.vxbar = self.vxbar - vx
        self.vybar = self.vybar - vy
        self.vzbar = self.vzbar - vz
        
    def corr_3d (self,a,b):
        corr_temporary = 0. 
        for i in range(self.redsz):
            for j in range(self.redsz):
                corr_temporary += np.corrcoef(a[i,j,:], b[i,j,:])
        corr_temporary /= (self.redsz)**2
        return corr_temporary[0,1]

    
    def SMG_Model(self):
        mdata = SMG_Model(self.vxbar , self.vybar, self.vzbar)
        self.SMG_xx, self.SMG_yy, self.SMG_zz, self.SMG_xy, self.SMG_xz, self.SMG_yz, SMG_div_x_div, \
        SMG_div_y_div, SMG_div_z_div = mdata.SMG()
        self.strain_xx,self.strain_yy,self.strain_zz,self.strain_xy,self.strain_xz,self.strain_yz = mdata.strain()
        dissipation=self.strain_xx*self.SMG_xx#+self.strain_xy*self.SMG_xy+self.strain_yy*self.SMG_yy + \
            #self.strain_xz*self.SMG_xz + self.strain_yz*self.SMG_yz + self.strain_zz*self.SMG_zz
        dissipation_mean = np.mean(dissipation)
        self.corr_smg = np.zeros(9)
        self.corr_smg[0] = self.corr_3d(SMG_div_x_div,self.sx_div)
        self.corr_smg[1] = self.corr_3d(SMG_div_y_div,self.sy_div)
        self.corr_smg[2] = self.corr_3d(SMG_div_z_div,self.sz_div)
        self.corr_smg[3] = self.corr_3d(self.SMG_xx,self.sxx)         
        self.corr_smg[4] = self.corr_3d(self.SMG_xy,self.sxy) 
        self.corr_smg[5] = self.corr_3d(self.SMG_xz,self.sxz) 
        self.corr_smg[6] = self.corr_3d(self.SMG_yz,self.syz) 
        self.corr_smg[7] = self.corr_3d(self.SMG_yy,self.syy) 
        self.corr_smg[8] = self.corr_3d(self.SMG_zz,self.szz) 
        dissipation=self.strain_xx*self.sxx#+self.strain_xy*self.sxy+self.strain_yy*self.syy + \
            #self.strain_xz*self.sxz + self.strain_yz*self.syz + self.strain_zz*self.szz
        dissipation_mean_DNS = np.mean(dissipation)
        return self.corr_smg,dissipation_mean,dissipation_mean_DNS
        
    def FSGS_Model(self,alpha_fl,nu):
        mdata = FSGS_Model(self.vxbar , self.vybar, self.vzbar)
        s_fL_x ,s_fL_y ,s_fL_z = mdata.Fractional_Laplacian(alpha_fl,nu)
        if alpha_fl> 0.5 :
            self.fl_sxx, fl_syy, fl_szz, self.fl_sxy, fl_sxz, fl_syz = mdata.FSGS_stress (alpha_fl, nu)
        
        V_fl_sx = s_fL_x.reshape(self.redsz**3)
        V_fl_sy = s_fL_y.reshape(self.redsz**3)
        V_fl_sz = s_fL_z.reshape(self.redsz**3)
        sx_div = self.sx_div[:]
        sy_div = self.sy_div[:]
        sz_div = self.sz_div[:]
        V_dns_sx =sx_div.reshape(self.redsz**3)
        V_dns_sy =sy_div.reshape(self.redsz**3)
        V_dns_sz =sz_div.reshape(self.redsz**3)
         
        corr_fsgs = np.zeros(12)
        corr_fsgs[0] = self.corr_3d(s_fL_x,self.sx_div)
        corr_fsgs[1] = self.corr_3d(s_fL_y,self.sy_div)
        corr_fsgs[2] = self.corr_3d(s_fL_z,self.sz_div)
        if alpha_fl> 0.5 :
            corr_fsgs[3] = self.corr_3d(self.fl_sxx,self.sxx)         
            corr_fsgs[4] = self.corr_3d(self.fl_sxy,self.sxy) 
            corr_fsgs[5] = self.corr_3d(fl_sxz,self.sxz) 
            corr_fsgs[6] = self.corr_3d(fl_syz,self.syz) 
            corr_fsgs[7] = self.corr_3d(fl_syy,self.syy) 
            corr_fsgs[8] = self.corr_3d(fl_szz,self.szz)  

        slope, intercept, r_value, p_value, std_err = stats.linregress(V_dns_sx,V_fl_sx)
        corr_fsgs[9] = slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(V_dns_sy,V_fl_sy)
        corr_fsgs[10] = slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(V_dns_sz,V_fl_sz)
        corr_fsgs[11] = slope
        test = np.corrcoef(V_fl_sy, V_dns_sy)
        return corr_fsgs, test

    def TFSGS_Model(self,alpha_tem,Lambda,nu,nterm):  
        mdata = Tempered_FSGS(self.vxbar , self.vybar, self.vzbar)  
        phi =np.zeros((2,2))
        Lambda_v = np.array([0.000000000001,Lambda])#0.01, 0.1, 0.5, 1, 5, 100])#Lambda])
        phi[0,0] = gamma(2*alpha_tem)
        phi[1,0] = gamma(2*alpha_tem) - gamma(2*alpha_tem-1)
        phi[1,1] = gamma(2*alpha_tem-1)
        s_tfL_x = 0
        s_tfL_y = 0
        s_tfL_z = 0
        for i in range(1,2):        
            s_x, s_y, s_z = mdata.Tempered_Fractional_Laplacian(alpha_tem,Lambda_v[i],nu)
            s_tfL_x = s_tfL_x + phi[nterm-1,i] * s_x#s_x/np.max(np.abs(s_x)) #phi[nterm-1,i] * s_x
            s_tfL_y = s_tfL_y + phi[nterm-1,i] * s_y#s_y/np.max(np.abs(s_y)) #phi[nterm-1,i] * s_y
            s_tfL_z = s_tfL_z + phi[nterm-1,i] * s_z#s_z/np.max(np.abs(s_z)) #phi[nterm-1,i] * s_z
        self.TFSGS_xx, self.TFSGS_yy, self.TFSGS_zz, self.TFSGS_xy, self.TFSGS_xz, self.TFSGS_yz = mdata.TFSGS_stress(s_tfL_x,s_tfL_y,s_tfL_z)  
        # dissipation=self.strain_xx*self.TFSGS_xx #+self.strain_xy*self.TFSGS_xy+self.strain_yy*self.TFSGS_yy + \
        #     self.strain_xz*self.TFSGS_xz + self.strain_yz*self.TFSGS_yz + self.strain_zz*self.TFSGS_zz
        # dissipation_mean = np.mean(dissipation)
        corr_tfsgs = np.zeros(3)
        corr_tfsgs[0] = self.corr_3d(s_tfL_x,self.sx_div)
        corr_tfsgs[1] = self.corr_3d(s_tfL_y,self.sy_div)
        corr_tfsgs[2] = self.corr_3d(s_tfL_z,self.sz_div)
#        corr_fsgs[3] = self.corr_3d(SMG_xx,self.sxx)         
#        corr_fsgs[4] = self.corr_3d(SMG_xy,self.sxy) 
#        corr_fsgs[5] = self.corr_3d(SMG_xz,self.sxz) 
#        corr_fsgs[6] = self.corr_3d(SMG_yz,self.syz) 
#        corr_fsgs[7] = self.corr_3d(SMG_yy,self.syy) 
#        corr_fsgs[8] = self.corr_3d(SMG_zz,self.szz)    
        return corr_tfsgs, self.TFSGS_zz     #, dissipation_mean

    def Two_PointCorr_main(self,add_in,t1,t2,t3,t4,ld):
        # L11 = abs(get_TwoPointCorr(self.vzbar,self.szz,self.redsz,3))#/vbar**3.0
        #   #   L11 /=L11[0]
        # Lsmg11 = abs(get_TwoPointCorr(self.vzbar,self.SMG_zz,self.redsz,3))#/vbar**3.0
        # L11 = get_TwoPointCorr(self.strain_zz,self.szz,self.redsz,2)#/vbar**3.0
        L11,Lsmg11, Ltfsgs1, Ltfsgs2, Ltfsgs3, Ltfsgs4 = get_TwoPointCorr_DNS(add_in,self.szz,self.SMG_zz,t1,t2,t3,t4,self.redsz,2,ld)
          #   L11 /=L11[0]
        #Lsmg11 = 1#get_TwoPointCorr(self.strain_zz,self.SMG_zz,self.redsz,3)#/vbar**3.0
            # Lfsgs11 = get_TwoPointCorr(self.vxbar,self.fl_sxx,self.redsz,0)/np.mean(self.vxbar**3.0)
        #Ltfsgs11 = abs(get_TwoPointCorr(self.vzbar,self.TFSGS_zz,self.redsz,3))#/vbar**3.0
        
        #   Lfsgs11 = get_TwoPointCorr(self.vxbar,self.fl_sxx,self.redsz,0)/np.mean(self.vxbar**3.0)
        #    L11 = abs(get_TwoPointCorr(self.vxbar,self.sxx,self.redsz,0))#/vbar**3.0
        # #   L11 /=L11[0]
        #    Lsmg11 = abs(get_TwoPointCorr(self.vxbar,self.SMG_xx,self.redsz,0))#/vbar**3.0
        #   # Lfsgs11 = get_TwoPointCorr(self.vxbar,self.fl_sxx,self.redsz,0)/np.mean(self.vxbar**3.0)
         
        #    Ltfsgs11 = abs(get_TwoPointCorr(self.vxbar,self.TFSGS_xx,self.redsz,0))#/vbar**3.0
        #   # Lfsgs11 = get_TwoPointCorr(self.vxbar,self.fl_sxx,self.redsz,0)/np.mean(self.vxbar**3.0)
        return L11, Lsmg11,Ltfsgs1, Ltfsgs2, Ltfsgs3, Ltfsgs4 #, Ltfsgs11

    def Two_PointCorr_TFSGS(self):
        Ltfsgs11 = 1# get_TwoPointCorr(self.strain_zz,self.TFSGS_zz,self.redsz,3)#/vbar**3.0
        
        #   Lfsgs11 = get_TwoPointCorr(self.vxbar,self.fl_sxx,self.redsz,0)/np.mean(self.vxbar**3.0)
        #    L11 = abs(get_TwoPointCorr(self.vxbar,self.sxx,self.redsz,0))#/vbar**3.0
        # #   L11 /=L11[0]
        #    Lsmg11 = abs(get_TwoPointCorr(self.vxbar,self.SMG_xx,self.redsz,0))#/vbar**3.0
        #   # Lfsgs11 = get_TwoPointCorr(self.vxbar,self.fl_sxx,self.redsz,0)/np.mean(self.vxbar**3.0)
         
        #    Ltfsgs11 = abs(get_TwoPointCorr(self.vxbar,self.TFSGS_xx,self.redsz,0))#/vbar**3.0
        #   # Lfsgs11 = get_TwoPointCorr(self.vxbar,self.fl_sxx,self.redsz,0)/np.mean(self.vxbar**3.0)
        return Ltfsgs11