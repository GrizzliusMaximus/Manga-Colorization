###WARNING!!!! REMOVE THIS IF YOUR COMPUTER IS CHEAP!!!###
# INCREASES THE RECURSION LIMIT
import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

##########################################################

import os
import scipy as sp
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal as ssignal
from numpy import matrix
from numpy import amax
from multiprocessing import Pool, Manager



def find(leader,n):
    x,y = n
    if (x,y) != leader[x][y]:
        leader[x][y] = find(leader, leader[x][y])
    return leader[x][y]

def connect(leader,size, a, b):
    la, lb = find(leader, a), find(leader, b)
    if la == lb:
        return
    if la < lb:
        la, lb = lb, la
    (xb,yb) = lb
    (xa,ya) = la
    leader[xb][yb] = la
    size[xa][ya] += size[xb][yb] 

def query(leader, a, b):
    return find(leader, a) == find(leader, b)


def EdgeDetect(image, count):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 1.0
    gray = cv2.GaussianBlur(gray,(3,3),0)
    img1 = gray * 1.0
    kernel = [[0,1,0],[1,-4,1],[0,1,0]]

    for i in range(1,19):
      conv = ssignal.convolve(img1, kernel, mode='valid')

    thres = np.absolute(conv).mean() * 0.75
    np.matrix.clip(conv,0, 255, out=conv)
    conv *= 255/np.max(conv)
    conv  = (amax(conv) - conv)
    
    #Union Find Screentone Removal
    leader = [[(i,j) for j in range(conv.shape[1])] for i in range(conv.shape[0])]
    size = [[1 for j in range(conv.shape[1])] for i in range(conv.shape[0])]
    for i in range(conv.shape[0]-1):
        for j in range(conv.shape[1]-1):
            if conv[i][j] < 250:
                if i != 0 and conv[i-1][j] < 250:
                    connect(leader,size,leader[i][j],leader[i-1][j])
                if j != 0 and conv[i][j-1] < 250:
                    connect(leader,size,leader[i][j],leader[i][j-1])\

    for i in range(conv.shape[0]-1):
        for j in range(conv.shape[1]-1):
            x,y = find(leader,(i,j))
            if (size[x][y] <= 16):
                conv[i][j] = 255
    
    for i in range(conv.shape[0]):
        for j in range(conv.shape[1]):
            if conv[i][j] < 250:
                conv[i][j] *= (conv[i][j]/255)**10
            else: 
                conv[i][j] = 255 
    conv = cv2.GaussianBlur(conv,(3,3),0)
    
    cv2.imwrite(os.path.join("linevideoout","lineframe%d.jpg" % count),conv)
    print("LineArt #%d Complete  :)"  % count)


vidcap = cv2.VideoCapture('bunnygirl.mkv')
success,image = vidcap.read()
count = 0
p = Pool(processes = 10)
while success:
    # cv2.imwrite(os.path.join("videoout","frame%d.jpg" % count), image)     # save frame as JPEG file 
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * 1.0
    # cv2.imwrite(os.path.join("grayvideoout","gray%d.jpg" % count), gray)     # save frame as JPEG file 
    p.apply_async(EdgeDetect,[image,count])
    for i in range(60):
        success,image = vidcap.read()
        if not success:
            break
    
    print('Read a new frame %d: ' % count, success)
    count += 1
p.close()
p.join()




    