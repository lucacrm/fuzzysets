import utils as ut
import numpy as np
import networkx as nx
import itertools as it
import math
from possibilearn.kernel import GaussianKernel

def get_eta(values, us, gamma, kernel):
    
    '''
    calculates the parameter eta with (27) in Probabilistic C-Means
    '''
    
    b = 1.0/sum(us)
    l=len(us)
    
    cp = zip(values,us)
    gram_term = b * b * sum([ u1*u2*kernel.compute(x1,x2) for (x1,u1) in cp for x2,u2 in cp])
    
    kernel_term =[]
    kernel_term2 = []
    for i in range(l):
        kernel_term.append(kernel.compute(values[i], values[i]))
        kernel_term2.append( 2.0*b* sum( [ us[j] * kernel.compute(values[i],values[j]) for j in range(l)] ))
                            
    return gamma * b * sum([u*(k - k2 + gram_term) for (u,k,k2) in zip (us, kernel_term, kernel_term2)])
        
                            
def update_us(values, us, eta, kernel):
    
    '''
    return the updated u with (26) in Probabilistic C-Means
    '''
    
    b = 1.0/sum(us)
    bq= b*b
    l=len(us)
    
    #print'b ', b
    #print'b^2', bq
    
    cp = zip(values,us)
    gram_term = bq * sum([u1*u2*kernel.compute(x1,x2) for (x1,u1) in cp for x2,u2 in cp])
    
    kernel_term =[]
    kernel_term2 = []
    for i in range(l):
        kernel_term.append(kernel.compute(values[i], values[i]))
        kernel_term2.append( 2.0*b* sum( [ us[j] * kernel.compute(values[i],values[j]) for j in range(l)] ))
    
    
    return [math.exp(-1.0/eta*( k - k2 + gram_term )) for (k,k2) in zip(kernel_term, kernel_term2)]


def get_memebrship(x, values, us, eta, kernel, gram_term):
    
    '''
    return the membership of a point with (28) in Probabilistic C-Means
    '''
    
    b = 1.0/sum(us)
    
    kernel_term = kernel.compute(x,x)
    
    kernel_term2 = 2.0*b* sum( [u * kernel.compute(v,x) for (u,v) in zip(us,values)] )
    
    
    return math.exp(-1.0/eta*( kernel_term - kernel_term2 + gram_term ))


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


def check_couples(x_start, x_end, alfa, d, discretization_size=20):

    x_start = np.array(x_start)
    x_end = np.array(x_end)
    discretization = np.arange(0., 1+1./discretization_size, 1./discretization_size)
    for x_between in [alpha*x_start + (1-alpha)*x_end for alpha in discretization]:
        if d(x_between) < alfa:
            return 0
    return 1

def build_clusters(values, indexes, alfa, d):
    
    G = nx.Graph()
    G.add_nodes_from(indexes)
    
    couples = list(it.combinations(indexes,2))
    
    i=0
    for c in couples:
        if check_couples(values[c[0]], values[c[1]], alfa, d):
            G.add_edge(c[0],c[1])
        i = i+1
            
    return [list(c) for c in list(nx.connected_components(G))]


def clustering(values, gamma, prc, kernel, epsilon=0.01):
    
    '''
    cluster a set of points by the support vector method
    
    - values: array of points to be clustered
    - kernel: Kernel for the trasformation of data in an high-dimensional space
    - gamma: trade-off parameter
    - epsilon: minimum accetable amount of penalties
    - prc: percentage of outliers
    
    Returns (clusters, ur), where
    
    - clusters: is a list of list. each sublist represent a cluster.
    - alfa: is the membership of a point that lie on the surface
    '''

    n = len(values)
    us = np.zeros(n) + 1.0/n
    eta = get_eta(values, us, gamma, kernel)
    delta = 1 #amount of penalties
    us = np.array(us)
    values = np.array(values)

    i=0
    while(delta >= epsilon and i < 100):

        new_us = np.array(update_us(values, us, eta, kernel))
        diff = us - new_us
        delta = sum( map(math.fabs, diff) )
        us = new_us
        i=i+1

    alfa = get_alfa(us, prc)

    b = 1.0/sum(us)
    cp = zip(values,us)
    gram_term = b * b * sum([u1*u2*kernel.compute(x1,x2) for (x1,u1) in cp for (x2,u2) in cp])

    dst = lambda x_new: get_memebrship(x_new, values, us, eta, kernel, gram_term) #gia una funzione di membership

    point_to_be_clustered=[]
    for i in range(len(us)):
        if us[i] >= alfa:
            point_to_be_clustered.append(i)

    return build_clusters(values, point_to_be_clustered, alfa, dst), alfa, us

def get_membership_function(values, gamma, prc, kernel, epsilon=0.01):
    
    n = len(values)
    us = np.zeros(n) + 1.0/n
    eta = get_eta(values, us, gamma, kernel)
    delta = 1 #amount of penalties
    us = np.array(us)
    values = np.array(values)

    i=0
    while(delta >= epsilon and i<100):

        new_us = np.array(update_us(values, us, eta, kernel))
        diff = us - new_us
        delta = sum( map(math.fabs, diff) )
        us = new_us
        i=i+1

    b = 1.0/sum(us)
    cp = zip(values,us)
    gram_term = b * b * sum([u1*u2*kernel.compute(x1,x2) for (x1,u1) in cp for (x2,u2) in cp])

    mf = lambda x_new: get_memebrship(x_new, values, us, eta, kernel, gram_term)
    
    return mf