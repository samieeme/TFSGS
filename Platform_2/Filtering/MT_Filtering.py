## In the nme of Allah
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat May 16 16:28:47 2020

@author: samieeme
"""


import queue
import numpy as np
from threading import Thread
#from scipy import ndimage 
from astropy.convolution import convolve

def Single_Filter(image, k):
    image_filtered = convolve(image, k, boundary='wrap') #ndimage.convolve(image, k, mode='wrap', cval=0.0)
    return image_filtered

def foo(q,im,k,c):
    a = convolve(im, k, boundary='wrap')
    q.put((a,c))
    

def Multi_Filter (image, k, N, R, num_threads):
    print("num_threads = "+str(num_threads))
    image_extended_1 = np.concatenate((np.concatenate((np.reshape(image[N-R:,:,:],(R,N,N)),image[:,:,:]),axis=0)\
                                ,np.reshape(image[0:R,:,:],(R,N,N))),axis=0)
    image_extended = np.concatenate((np.concatenate((image_extended_1[:,N-R:,:],image_extended_1[:,:,:]),axis=1),\
                                 image_extended_1[:,0:R,:],),axis=1)
    del(image_extended_1)
    im_filtered_part = [None]*2*num_threads
    im_part= [None]*2*num_threads

    for i in range(num_threads-1):
        j = 0
        im_part[i+j*num_threads] = image_extended[i*int(N/num_threads):(i+1)*int(N/num_threads)+2*R,0:int(N/2)+2*R,:]
    
        j = 1 
        im_part[i+j*num_threads] = image_extended[i*int(N/num_threads):(i+1)*int(N/num_threads)+2*R,int(N /2):,:]        

    i = num_threads-1
    j = 0
    im_part[i+j*num_threads] = image_extended[i*int(N/num_threads):,0:int(N/2)+2*R,:]
    j = 1
    im_part[i+j*num_threads] = image_extended[i*int(N/num_threads):,int(N /2):,:]
 
    que = queue.Queue()
    thrd =[]
    
    for i in range(num_threads):
        for j in range(2):
            t = Thread(target=foo, args=(que,im_part[i+j*num_threads], k, i+j*num_threads))
            t.start()
            thrd.append(t)
            print(t)
    
    for i in thrd:
        result,c = que.get()
        print(c)
        im_filtered_part[c] = result
        que.task_done()
        
    im_all_filtered = im_filtered_part[0][R:int(N/num_threads)+R,R:int(N/2)+R,:]
    if (num_threads > 2):
        for i in range(1,num_threads-1):
            im_all_filtered = np.concatenate((im_all_filtered,im_filtered_part[i]\
                                              [R:int(N/num_threads)+R,R:int(N/2)+R,:]),axis=0)
    im_all_filtered = np.concatenate((im_all_filtered,im_filtered_part[num_threads-1]\
                                          [R:int(N/num_threads)+R+N%num_threads,R:int(N/2)+R,:]),axis=0)
    image_filtered_MT = im_all_filtered
    del(im_all_filtered)
    
    im_all_filtered = im_filtered_part[num_threads][R:int(N/num_threads)+R,R:int(N/2)+R,:]
    if (num_threads > 2):
        for i in range(1,num_threads-1):
            im_all_filtered = np.concatenate((im_all_filtered,im_filtered_part[i+num_threads]\
                                              [R:int(N/num_threads)+R,R:int(N/2)+R,:]),axis=0)
    im_all_filtered = np.concatenate((im_all_filtered,im_filtered_part[num_threads-1+num_threads]\
                                          [R:int(N/num_threads)+R+N%num_threads,R:int(N/2)+R,:]),axis=0)
    image_filtered_MT = np.concatenate((image_filtered_MT,im_all_filtered),axis=1)
    
    del(im_all_filtered)
    
    return image_filtered_MT

    
    
    
    
    
    
    
    
    
    
    
    
    
