import utils as ut
import numpy as np
import pandas as pd
import random
from possibilearn import *
from possibilearn.kernel import GaussianKernel
import math
import itertools as it
import gurobipy as gpy

def solve_wolf(values, k, c):
    
    '''
    Solves the dual optimization problem on the basis of SV clustering

    - x: array containing data to be clustered
    - k: kernel function to be used
    - c: trade-off parameter
    '''

    n=len(values)
    # p = 1.0 / (n*c)    
    
    model = gpy.Model('Wolf')
    model.setParam('OutputFlag', 0)

    for i in range(n):
        model.addVar(name="beta_%d" %i, lb=0, ub=c, vtype=gpy.GRB.CONTINUOUS)

    model.update()
    b = model.getVars()
    
    # obj == - SVC(11) 
    obj = gpy.QuadExpr()
    for i, j in it.product(range(n), range(n)):
        obj.add( b[i] * b[j], k.compute(values[i], values[j]))
    
    for i in range(n):
        obj.add( -1 * b[i] * k.compute(values[i], values[i]))
    
    model.setObjective(obj, gpy.GRB.MINIMIZE)
    
    constEqual = gpy.LinExpr()
    constEqual.add(sum(b), 1.0)

    model.addConstr(constEqual, gpy.GRB.EQUAL, 1)

    model.optimize()
    
    if model.Status != gpy.GRB.OPTIMAL:
        raise ValueError('optimal solution not found!')

    b_opt = [chop(v.x, 0, c) for v in model.getVars()]
    
    return b_opt

def distance_from_center(x_new, x, b_opt, k, gram_term):
    
    '''
    Computes the squared distance between the image of a point and the center of the sphere founded during 1-SVM
    
    - x_new: starting point
    - x: array of points to be clustered
    - b_opt: optimal value of variables in the Wolfe dual problem
    - k: kernel function
    - gram_term: common addend based on the Gram matrix
    '''
    
    d1 = k.compute(x_new, x_new)
    d2 = np.array([k.compute(x_i, x_new) for x_i in x]).dot(b_opt)
    d = d1 - 2 * d2 + gram_term
    return d

def squared_radius_and_distance(x, b_opt, index_sv, k, c, mean=True):
    
    '''
    Computes the squared radius of the found sphere and a function returning
    the squared distance between the image of a generic point and the sphere center
    
    - x: array of points to be clustered
    - b_opt: optimal value of variables in the Wolfe dual problem
    - index_sv: index of the support vector in the dataset
    - k: kernel function
    - c: trade-off parameter
    
    Returns (r, d), where
    
    - r is the squared radius
    - d is the function computing the squared distance
    '''
 
    if(len(index_sv) == 0):
        raise ValueError('No support vector founded')
    
    gram = np.array(np.array([[k.compute(x1, x2) for x1 in x] for x2 in x]))
    gram_term = np.array(b_opt).dot(gram.dot(b_opt))
    d = lambda x_new: distance_from_center(x_new, x, b_opt, k, gram_term)
    x = np.array(x)
    r = [d(sv) for sv in x[[i for i in index_sv]]]
    if len(r) == 0:
        return None
    
    return (np.mean(r), d) if mean else (np.max(r), d)


def check_couple(x_start, x_end, radius, d, discretization_size=20):
    
    x_start = np.array(x_start)
    x_end = np.array(x_end)
    discretization = np.arange(0., 1+1./discretization_size, 1./discretization_size)
    for x_between in [alpha*x_start + (1-alpha)*x_end for alpha in discretization]:
        if d(x_between) > radius:
            return 0
    return 1

def build_clusters(x, index_v, index_sv, radius, d, all_links=False):
    
    '''
    Build clusters as connected component of the graph which its nodes are the non-BSV points in the 
    dataset. There is an edge between nodes i and j if for all points in the segment connecting i and j the
    squared distance from the center of the sphere is < radius
    
    - x: array of points to be clustered
    - index_v: list/array of the indexes of the non-BSV and non-SV points
    - index_sv: list/array of the indexes of the support vectors
    - radius: radius of the sphere
    - d: function computing the squared distance between two points
    '''
    import networkx as nx
    
    G = nx.Graph()
    G.add_nodes_from(index_v)
    G.add_nodes_from(index_sv)
    
    if(all_links):
        indexes = index_v + index_sv
        couples = filter(lambda x: x[0]!=x[1], [[i,j] for i in indexes for j in indexes])
    else:
        couples = [[i,j] for i in index_sv for j in index_v]
    
    for c in couples:
        if check_couple(x[c[0]], x[c[1]], radius, d):
            G.add_edge(c[0],c[1])
            
    return [list(c) for c in list(nx.connected_components(G))]

def clustering(x, kernel, c, all_links):
    
    '''
    cluster a set of points by the support vector method
    
    - x: array of points to be clustered
    - kernel: Kernel for the trasformation of data in an high-dimensional space
    - c: trade-off parameter
    - labels: array of labels
    - graph: if True plot graphs showing which point are classified support vectors
    
    Returns (clusters, radius), where
    
    - clusters is a list of list. each sublist represent a cluster.
    - r is the squared radius
    '''
    
    import matplotlib.pyplot as plt
        
    k = kernel 
    
    betas = solve_wolf(x, k, c)
    
    index_bsv = []
    index_sv = []
    index_v = []
    
    for i in range(len(betas)):
        if 0 < betas[i] < c:
            index_sv.append(i)
        elif betas[i] == c:
            index_bsv.append(i)
        else:
            index_v.append(i)
        
    radius, d = squared_radius_and_distance(x, betas, index_sv, k, c)
        
    return build_clusters(x, index_v, index_sv, radius, d, all_links), radius
    