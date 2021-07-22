import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

def ohc(y):
    n = len(y)
    p = len(np.unique(y))
    temp = np.zeros((n,p))
    temp[range(n),y[:,0]] = 1.0
    return temp

def get_spiral():
     # Idea: radius -> low...high
    #           (don't start at 0, otherwise points will be "mushed" at origin)
    #       angle = low...high proportional to radius
    #               [0, 2pi/6, 4pi/6, ..., 10pi/6] --> [pi/2, pi/3 + pi/2, ..., ]
    # x = rcos(theta), y = rsin(theta) as usual
    
    radius = np.linspace(1,10,100)
    theta = np.empty((6,100))
    for i in range(6):
        start_angle = np.pi*i/3.0
        end_angle = start_angle + np.pi/2.0
        points = np.linspace(start_angle,end_angle,100)
        theta[i] = points
        
    #cartesian coordinates
    x1 = np.empty((6,100))
    x2 = np.empty((6,100))
    for i in range(6):
        x1[i] = radius*np.cos(theta[i])
        x2[i] = radius*np.sin(theta[i])
    
    #inputs
    X = np.empty((600,2))
    X[:,0] = x1.flatten()
    X[:,1] = x2.flatten()
    
    #add noise
    X += np.random.randn(600,2)*0.5
    
    #target
    Y = np.array([0]*100 + [1]*100 + [0]*100 + [1]*100 + [0]*100 + [1]*100)
    return X,Y


def get_normalized_data():
    data = pd.read_csv('train.csv')
    X = data.iloc[:,1:].values
    Y = data.iloc[:,0].values.reshape(-1,1)
    #normalizing the data
    X = X/255.0
    #covert Y to indicator matrix
    Y_ind = ohc(Y)
    return X,Y_ind

def init_weights(m1,m2):
    w = np.random.rand(m1,m2)*np.sqrt(2.0/m1)
    b = np.zeros(m2,dtype = np.float32)
    return w.astype(np.float32),b