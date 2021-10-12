import numpy as np
from torch.autograd import Variable
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import IPython

def matrix2axisAngle(A):
    val = (A[0,0]+A[1,1]+A[2,2]-1)/2
    
    if np.abs(val-1)<0.001:
        return np.array([1,0,0]),0

    theta = np.arccos(val)

    if np.abs(np.sin(theta)) < 0.001:
        return np.array([1,0,0]),0

    e1 = (A[2,1]-A[1,2])/(2*np.sin(theta))
    e2 = (A[0,2]-A[2,0])/(2*np.sin(theta))
    e3 = (A[1,0]-A[0,1])/(2*np.sin(theta))

    return np.array([e1,e2,e3]), theta

def axisAngle2matrix(e, theta):
    ex = np.zeros([3,3])
    ex[0,1] = -e[2]
    ex[1,0] = +e[2]
    ex[0,2] = +e[1]
    ex[2,0] = -e[1]
    ex[1,2] = -e[0]
    ex[2,1] = +e[0]
    A = np.eye(3)*np.cos(theta) + (1-np.cos(theta))*e@e.T + ex*np.sin(theta) 
    return A