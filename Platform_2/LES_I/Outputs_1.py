# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:11:40 2019

@author: samieeme
"""
from Models_1 import Pre_Models, Solver_filtered_field_sep, SMG_Model, FSGS_Model, Tempered_FSGS, Solver_filtered_field_JHU
from functions_1 import remain, deriv_x, deriv_y, deriv_z, div, get_TwoPointCorr
import numpy as np
from scipy import stats
from math import gamma
   
class Output_Corr (object):
    
    def __init__(self,add_in,filename,Rs):
        solver = Solver_filtered_field_sep(add_in,filename,Rs) 
        #solver = Solver_filtered_field_sep(filename,Rs) 
        self.vxbar , self.vybar, self.vzbar, self.sxx, self.sxy, self.sxz, \
        self.syy, self.syz, self.szz, self.redsz = solver.Output()
        self.sx_div = div(self.sxx,self.sxy,self.sxz, self.redsz)
        self.sy_div = div(self.sxy,self.syy,self.syz, self.redsz)
        self.sz_div = div(self.sxz,self.syz,self.szz, self.redsz)
        
    def corr_3d (self,a,b):
        corr_temporary = 0. 
        # for i in range(self.redsz):
        #     for j in range(self.redsz):
        #         corr_temporary += np.corrcoef(a[i,j,:], b[i,j,:])
        corr_temporary = np.corrcoef(a.reshape((1,self.redsz**3)), b.reshape((1,self.redsz**3)))
        #corr_temporary /= (self.redsz)**2
        return corr_temporary[0,1]

    
    def SMG_Model(self):
        mdata = SMG_Model(self.vxbar , self.vybar, self.vzbar)
        self.SMG_xx, SMG_yy, SMG_zz, self.SMG_xy, SMG_xz, SMG_yz, SMG_div_x_div, \
        SMG_div_y_div, SMG_div_z_div = mdata.SMG()
        
        self.corr_smg = np.zeros(9)
        self.corr_smg[0] = self.corr_3d(SMG_div_x_div,self.sx_div)
        self.corr_smg[1] = self.corr_3d(SMG_div_y_div,self.sy_div)
        self.corr_smg[2] = self.corr_3d(SMG_div_z_div,self.sz_div)
        self.corr_smg[3] = self.corr_3d(self.SMG_xx,self.sxx)         
        self.corr_smg[4] = self.corr_3d(self.SMG_xy,self.sxy) 
        self.corr_smg[5] = self.corr_3d(SMG_xz,self.sxz) 
        self.corr_smg[6] = self.corr_3d(SMG_yz,self.syz) 
        self.corr_smg[7] = self.corr_3d(SMG_yy,self.syy) 
        self.corr_smg[8] = self.corr_3d(SMG_zz,self.szz) 
        return self.corr_smg
        
    def FSGS_Model(self,alpha_fl,nu):
        mdata = FSGS_Model(self.vxbar , self.vybar, self.vzbar)
        s_fL_x ,s_fL_y ,s_fL_z = mdata.Fractional_Laplacian(alpha_fl,nu)
        # if alpha_fl> 0.5 :
        #     self.fl_sxx, fl_syy, fl_szz, self.fl_sxy, fl_sxz, fl_syz = mdata.FSGS_stress (alpha_fl, nu)
        
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
        # if alpha_fl> 0.5 :
        #     corr_fsgs[3] = self.corr_3d(self.fl_sxx,self.sxx)         
        #     corr_fsgs[4] = self.corr_3d(self.fl_sxy,self.sxy) 
        #     corr_fsgs[5] = self.corr_3d(fl_sxz,self.sxz) 
        #     corr_fsgs[6] = self.corr_3d(fl_syz,self.syz) 
        #     corr_fsgs[7] = self.corr_3d(fl_syy,self.syy) 
        #     corr_fsgs[8] = self.corr_3d(fl_szz,self.szz)  

        # slope, intercept, r_value, p_value, std_err = stats.linregress(V_dns_sx,V_fl_sx)
        # corr_fsgs[9] = slope
        # slope, intercept, r_value, p_value, std_err = stats.linregress(V_dns_sy,V_fl_sy)
        # corr_fsgs[10] = slope
        # slope, intercept, r_value, p_value, std_err = stats.linregress(V_dns_sz,V_fl_sz)
        # corr_fsgs[11] = slope
        test = np.corrcoef(V_fl_sy, V_dns_sy)
        return corr_fsgs, test

    def TFSGS_Model(self,alpha_tem,Lambda,nu,nterm):  
        mdata = Tempered_FSGS(self.vxbar , self.vybar, self.vzbar)  
        phi =np.zeros((2,2))
        Lambda_v = np.array([0.00000000000000001, Lambda])
        phi[0,0] = gamma(2*alpha_tem)*0
        phi[1,0] = gamma(2*alpha_tem) - gamma(2*alpha_tem-1)
        phi[1,1] = gamma(2*alpha_tem-1)
        s_tfL_x = 0
        s_tfL_y = 0
        s_tfL_z = 0
        for i in range(nterm):        
            s_x, s_y, s_z = mdata.Tempered_Fractional_Laplacian(alpha_tem,Lambda_v[i],nu)
            s_tfL_x = s_tfL_x + phi[nterm-1,i] * s_x
            s_tfL_y = s_tfL_y + phi[nterm-1,i] * s_y
            s_tfL_z = s_tfL_z + phi[nterm-1,i] * s_z
            
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
        return corr_tfsgs

    def Two_PointCorr(self):
#        L11 = get_TwoPointCorr(self.vxbar,self.sxy,self.redsz,0)
#     #   L11 /=L11[0]
#        Lsmg11 = get_TwoPointCorr(self.vxbar,self.SMG_xy,self.redsz,0)
#        Lfsgs11 = get_TwoPointCorr(self.vxbar,self.fl_sxy,self.redsz,0)
        L11 = get_TwoPointCorr(self.vxbar,self.sxx,self.redsz,0)/np.mean(self.vxbar**3.0)
     #   L11 /=L11[0]
        Lsmg11 = get_TwoPointCorr(self.vxbar,self.SMG_xx,self.redsz,0)/np.mean(self.vxbar**3.0)
        Lfsgs11 = get_TwoPointCorr(self.vxbar,self.fl_sxx,self.redsz,0)/np.mean(self.vxbar**3.0)
        return L11 , Lsmg11, Lfsgs11
    
    
        
        
        
        
        
        
        