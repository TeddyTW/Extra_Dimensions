import numpy as np
import numpy.random 
import scipy.special
import random 
import math
import matplotlib.pyplot as plt

def random_2d_points(x0, x1, y0, y1, N):
    Points={'x':[], 'y':[]}
    for i in range(0, N):
        x=random.uniform(x0, x1)
        y=random.uniform(y0, y1)
        Points['x'].append(x)
        Points['y'].append(y)
        
    return Points

def random_2d_points_lam(x0, x1, y0, y1, Lam):
    N=numpy.random.poisson(Lam*(x1-x0)*(y1-y0))
    Points={'x':[], 'y':[]}
    for i in range(0, N):
        x=random.uniform(x0, x1)
        y=random.uniform(y0, y1)
        Points['x'].append(x)
        Points['y'].append(y)
        
    return Points


def region_count_square(x0, x1, y0, y1, Points):
    Area={'xrange':[x0, x1], 'yrange':[y0,y1]}
    AreaCount=0
    
    for i in range(0, len(Points['x'])):
        cond=((Points['x'][i]>=Area['xrange'][0]) and (Points['x'][i]<=Area['xrange'][1]) and
             (Points['y'][i]>=Area['yrange'][0]) and (Points['y'][i]<=Area['yrange'][1]))
        if(cond):
            AreaCount+=1
    return AreaCount
            
def region_count_circle(R, C, Points):
    
    AreaCount=0
    for i in range(0, len(Points['x'])):
        shiftx=Points['x'][i]-C[0]
        shifty=Points['y'][i]-C[1]
        r=np.sqrt(shiftx**2+shifty**2)
        #print(r)
        if(r<=R):
            AreaCount+=1
    return AreaCount


def vacancy_ind_square(x0, x1, y0, y1, Points):
    if(region_count_square(x0, x1, y0, y1, Points) < 1):
        return 1
    else:
        return 0
    
def capacity_functional(beta, area):
    return 1 - np.exp(-beta*area)

def contact_dist_2d(beta, r):
    p=np.pi
    return 1 - np.exp(-beta*p*r*r)


def dist_2d(u, X):
    distance=[]
    for i in range(0, len(X['x'])):
        distx=X['x'][i]-u[0]
        disty=X['y'][i]-u[1]
        dist=np.sqrt(disty**2+distx**2)
        distance=np.append(distance, dist)
        
    return(distance.min())


def K_function(t, X, W):
    #W=width of the feild in [x0,y0,x1,y1]
    K=[]
    for i in range(0, len(X['x'])):

        Npoints=region_count_circle(t, [X['x'][i],X['y'][i]], X)-1

        cond=((X['x'][i]>t+W[0]) and (X['x'][i]<W[2]-t) and (X['y'][i]>t+W[1]) and (X['y'][i]<W[3]-t))
        if(cond):
            K=np.append(K, Npoints)

    return K.mean()

def F_function(r, X, W):
    #W=width of the feild in [x0,y0,x1,y1]
    F=[]
    for i in range(W[0], W[2]):
        for j in range(W[1],W[3]):

            if(region_count_circle(r, [i,j] , X)>0):
                F=np.append(F, 1)
            else:
                F=np.append(F, 0)

    return(F.mean())

    def G_function(r, X):
    #W=width of the feild in [x0,y0,x1,y1]
    G=[]
    for i in range(0, len(X['x'])):
        
            if((region_count_circle(r, [X['x'][i],X['y'][i]] , X)-1)>0):
                G=np.append(G, 1)
            else:
                G=np.append(G, 0)

    return(G.mean())