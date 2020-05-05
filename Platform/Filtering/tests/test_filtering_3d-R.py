#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:54:18 2020

@author: zayern
"""


import queue
import numpy as np
from threading import Thread
from scipy import ndimage 
import time

def Filter(image, k):
    image_filtered = ndimage.convolve(image, k, mode='wrap', cval=0.0)
    return image_filtered

def foo(q,im,c):
    #bar = q.get()
    #a = input("Salam = ")
    #print ('hello 0'.format(bar))
    a = ndimage.convolve(im, k, mode='wrap', cval=0.0)
    q.put((a,c))



N = 320
R=2
N -= N%R
# np.random.seed(100)
image = np.random.randn(N,N,N)
k = np.ones((2*R+1,2*R+1,2*R+1))

start = time.time()   
image_filtered = Filter(image, k)
print("Single threading: ",str(time.time()-start))
num_threads = 4
image_extended_1 = np.concatenate((np.concatenate((np.reshape(image[N-R:,:,:],(R,N,N)),image[:,:,:]),axis=0)\
                                ,np.reshape(image[0:R,:,:],(R,N,N))),axis=0)
image_extended = np.concatenate((np.concatenate((image_extended_1[:,N-R:,:],image_extended_1[:,:,:]),axis=1),\
                                 image_extended_1[:,0:R,:],),axis=1)
del(image_extended_1)
#del(image)
# numerate = [None]*2*num_threads
# image_part = [None]*2*num_threads
im_all_filtered = [None]*2*num_threads
# thread_counter = [None]*2*num_threads
im_filtered_part = [None]*2*num_threads
#errorm = [None]*2*num_threads
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

start = time.time()  
que = queue.Queue()

thrd =[]

for i in range(num_threads):
    for j in range(2):
        t = Thread(target=foo, args=(que,im_part[i+j*num_threads],i+j*num_threads))
        t.start()
        thrd.append(t)
      #  t.join()
        print(t)
    
for i in thrd:
    result,c = que.get()
  #  que.task_done()
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
    
#del(image_filtered_part_temp)
#print("multi threading: # thread ",str(num_threads),", time ",str(time.time()-start))      
    
    
print("Multi threading: ",str(time.time()-start))

error = np.max(np.abs(image_filtered_MT-image_filtered))





