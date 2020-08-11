##   In the name of Allah
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:27:40 2020

@author: samieeme
"""


import queue
import threading
import numpy as np
from scipy import ndimage    
import time
import numpy as np


def Filter_SThread(image, k):
    image_filtered = ndimage.convolve(image, k, mode='wrap', cval=0.0)
    return image_filtered
    
def Filter_MThread(image, k,count):
    image_part_temp = ndimage.convolve(image, k, mode='wrap', cval=0.0)
    q.put((image_part_temp,count))

q = queue.PriorityQueue()

def Filter_Thread(image,k,R,N,num_threads_max):
    
    num_threads=num_threads_max
    if num_threads == 1:
        image_filtered_part= Filter_SThread(image, k)
    elif (num_threads > 1):
        numerate = [None]*num_threads
        image_part = [None]*num_threads
        im_all_filtered = [None]*num_threads
        thread_counter = [None]*num_threads
        image_filtered_part_temp = [None]*num_threads
        
        start = time.time()
        for i in range(num_threads):    
            if i == 0:
                image_part[i] = np.concatenate((np.reshape(image[N-R:,:,:],(R,N,N)),image[0:int(N/num_threads)+R,:,:]),axis=0)
            elif i == num_threads-1:
                image_part[i] = np.concatenate((image[i*int(N/num_threads)-R:,:,:],np.reshape(image[0:R,:,:],(R,N,N))),axis=0)
            else:
                image_part[i] = image[i*int(N/num_threads)-R:(i+1)*int(N/num_threads)+R,:,:]
        
        count = 0
        for im in image_part:
            t = threading.Thread(target=Filter_MThread, args = (im,k,count))
            # t = threading.Thread(name=str(count),target=Filter_thread)
            t.daemon = True
            t.start()
            count += 1
        
        for j in range(num_threads):
            im_all_filtered[j],thread_counter[j] = q.get() #
        
        
        for i,j in enumerate(np.argsort(thread_counter)):
            image_filtered_part_temp[i] = im_all_filtered[j]
        
        
        image_filtered_part = image_filtered_part_temp[0][R:int(N/num_threads)+R,:,:]
        for j in range(1,num_threads-1):
            image_filtered_part = np.concatenate((image_filtered_part,image_filtered_part_temp[j][R:int(N/num_threads)+R,:,:]),axis=0)
        image_filtered_part = np.concatenate((image_filtered_part,image_filtered_part_temp[num_threads-1][R:int(N/num_threads)+R+N%num_threads,:,:]),axis=0)
        #print("multi threading: # thread ",str(num_threads),", time ",str(time.time()-start))  
