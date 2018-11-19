import utils as ut
import numpy as np
import networkx as nx
import itertools as it
import math
from possibilearn.kernel import GaussianKernel

def get_eta(values, kernel, us, gamma):
    
    '''
    calculates the parameter eta with (27) in Probabilistic C-Means
    '''
    
    b = 1.0/sum(us)
    
    cp = zip(values,us)
    gram_term = b * b * sum([ u1*u2*kernel.compute(x1,x2) for (x1,u1) in cp for x2,u2 in cp])
    
    kernel_term =[]
    kernel_term2 = []
    for i in range(len(us)):
        kernel_term.append(kernel.compute(values[i], values[i]))
        kernel_term2.append( 2.0*b* sum( [us[i]*kernel.compute(values[i],x) for x in values] ))
                            
    return gamma * b * sum([u*(k - k2 + gram_term) for (u,k,k2) in zip (us, kernel_term, kernel_term2)])
        
                            
def update_us(values, kernel, us, eta):
    
    '''
    return the updated u with (26) in Probabilistic C-Means
    '''
    
    b = 1.0/sum(us)
    
    cp = zip(values,us)
    gram_term = b * b * sum([u1*u2*kernel.compute(x1,x2) for (x1,u1) in cp for x2,u2 in cp])
    
    kernel_term =[]
    kernel_term2 = []
    for i in range(len(us)):
        kernel_term.append(kernel.compute(values[i], values[i]))
        kernel_term2.append( 2.0*b* sum( [us[i]*kernel.compute(values[i],x) for x in values] ))
    
    
    return [math.exp(-1.0/eta*( k - k2 + gram_term )) for (u,k,k2) in zip(us, kernel_term, kernel_term2)]


def get_memebrship(x, values, kernel, us, eta, gram_term):
    
    '''
    return the membership of a point with (28) in Probabilistic C-Means
    '''
    
    b = 1.0/sum(us)
    
    kernel_term = kernel.compute(x,x)
    
    kernel_term2 = 2.0*b* sum( [u * kernel.compute(v,x) for (u,v) in zip(us,values)] )
    
    
    return [math.exp(-1.0/eta*( kernel_term - kernel_term2 + gram_term ))]


def get_alfa(us, prc):
    
    '''
    returns the boundary alfa that exclude a certain percentage of points
    
    us: array of u membership values
    prc: percentage of points to be considered outliers
    '''
    
    n=len(us)
    sorted_us = sorted(us)
    i=(n+1)*prc/100
    return sorted_us[i]

    

def build_clusters(x, index_v, index_sv, radius, d):


def clustering(x, kernel, c, labels=[], graph=False):
    
    
    