# in the name of God
# functions in filtering function

import numpy as np
import matplotlib.pyplot as plt
from math import pi, gamma
from astropy.convolution import convolve
from functions_filtering import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, Find_Neighboar
from Solve_Module import Solver

filename="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/\
Semester_summer_2019/research/April-15-2019/data_DNS/DrJaberi/DNS3Duvw.dat"

filename_out="C:/Users/samieeme.CMSE020/Desktop/New folder/semesters/PHD MSU/\
Semester_summer_2019/research/April-15-2019/data_DNS/DrJaberi/"

R = 2
time = 0
solver = Solver(filename,R)
vxbar, vybar, vzbar, sxx, sxy, sxz, syy, syz, szz, redsz, rrr = solver.get_bar_box()

solver.get_outfile(filename_out,time,R)








