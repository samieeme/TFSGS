## In the name of Allah
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:23:18 2020

@author: zayern
"""


import queue
import threading
import numpy as np
from scipy import ndimage    
import time
import numpy as np


def Filter(image, k):
    image_filtered = ndimage.convolve(image, k, mode='wrap', cval=0.0)
    return image_filtered
    
def Filter_thread(image, k,count):
    image_part_temp = ndimage.convolve(image, k, mode='wrap', cval=0.0)
    aa=count*np.ones((1,N,N))
    image_filtered = np.concatenate((image_part_temp,aa),axis=0)
    q.put(image_filtered)

N = 320
R = 10
k = np.ones((2*R+1,2*R+1,2*R+1))
image = np.random.randn(N,N,N)
start = time.time()   
# np.random.seed(100)
image_filtered = Filter(image, k)
print("Single threading: ",str(time.time()-start))

num_threads = 4
numerate = [None]*num_threads
image_part = [None]*num_threads
im_all_filtered = [None]*num_threads
image_filtered_part_temp = [None]*num_threads

start = time.time()
for i in range(num_threads):    
    if i == 0:
        image_part[i] = np.concatenate((np.reshape(image[N-R:,:,:],(R,N,N)),image[0:int(N/num_threads)+R,:,:]),axis=0)
    elif i == num_threads-1:
        image_part[i] = np.concatenate((image[i*int(N/num_threads)-R:N,:,:],np.reshape(image[0:R,:,:],(R,N,N))),axis=0)
    else:
        image_part[i] = image[i*int(N/num_threads)-R:(i+1)*int(N/num_threads)+R,:,:]
        
# for i in range(num_threads):    
#     if i == 0:
#         image_part[i] = np.concatenate((np.reshape(image[N-1,:],(1,N)),image[0:int(N/num_threads)+R,:]),axis=0)
#     elif i == num_threads-1:
#         image_part[i] = np.concatenate((image[i*int(N/num_threads)-R:N,:],np.reshape(image[0,:],(1,N))),axis=0)
#     else:
#         image_part[i] = image[i*int(N/num_threads)-R:(i+1)*int(N/num_threads)+R,:]
# image_filtered_2 = ndimage.convolve(image_part[0], k, mode='wrap', cval=0.0)

#im_all_filtered = image_part
q = queue.PriorityQueue()
# print(q.qsize()) 

count = 0
for im in image_part:
    t = threading.Thread(target=Filter_thread, args = (im,k,count))
    # t = threading.Thread(name=str(count),target=Filter_thread)
    t.daemon = True
    t.start()
    count += 1
#print(q.qsize()) 

for j in range(num_threads):
  #   print (threading.currentThread().getName())
    im_all_filtered[j] = q.get()
print("multi threading: # thread ",str(num_threads),", time ",str(time.time()-start))

for i in range(num_threads):
    for j in range(num_threads):
        if (i==im_all_filtered[j][int(N/num_threads)-1+2*R+1,0,0]):
                image_filtered_part_temp[i] = im_all_filtered[j]


image_filtered_part = image_filtered_part_temp[0][R:int(N/num_threads)+R,:,:]
for j in range(1,num_threads):
    image_filtered_part = np.concatenate((image_filtered_part,image_filtered_part_temp[j][R:int(N/num_threads)+R,:,:]),axis=0)



error = np.max(np.abs(image_filtered_part-image_filtered))
