import numpy as np
import matplotlib.cm as cm
import pandas as pd
import random
from possibilearn import *
from possibilearn.kernel import GaussianKernel
import math
import itertools as it
import gurobipy as gpy

#farla in modo che come estremi utilizzi valori coerenti ai dati passati
def g(m):
    return (-4 + np.random.random(2*m) * 8).reshape((m, 2))

def create_generator(data_set, n_components):
    a = -4 #capire
    b = 8 #capire
    def gen(m):
        return (a + np.random.random(n_components*m) * b).reshape((m, n_components))

def gr_membership_graded(estimated_membership, color_map):
    x = np.arange(-4, 4, .1)
    y = np.arange(-4, 4, .1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([estimated_membership((x, y))
                   for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    plt.imshow(Z, interpolation='bilinear', origin='lower',
               cmap=color_map, alpha=0.99, extent=(-4, 4, -4, 4))
    
def gr_membership_contour(estimated_membership):
    x = np.arange(-4, 4, .1)
    y = np.arange(-4, 4, .1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([estimated_membership((x, y))
                   for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    membership_contour = plt.contour(X, Y, Z,
                                     levels=(.1, .3, .5, .7, .8, .9, .95), colors='k')
    plt.clabel(membership_contour, inline=1)

def gr_dataset(): 
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'green', 'red')):
        plt.scatter(iris_values_2d[iris_labels==lab, 0],
                    iris_values_2d[iris_labels==lab, 1],
                    label=lab,
                    c=col)

def get_different_clusters(clusters, clusters_index, clusters_labels):
    d={}
    c=[]
    l=[]
    i=[]
    for j in range(len(clusters_labels)):
        if clusters_labels[j] not in d:
            d[clusters_labels[j]] = True
            c.append(clusters[j])
            i.append(clusters_index[j])
            l.append(clusters_labels[j])
    return c, i, l

def print_graph(values, labels):
    
    import matplotlib.pyplot as plt

    values = np.array(values)
    
    fig, axs = plt.subplots(1, 2,
    sharey=True, figsize=(15, 4))
    fig.subplots_adjust(wspace=0.2)

    axs[0].scatter(values[:,0],values[:,1]) #grafico di tutti i punti, senza appartenenza
    axs[0].set_xlim(-4,4)
    
    for i in range(len(values)):
        if labels[i] == 'Iris-setosa':
            axs[1].plot(values[i][0],values[i][1],'bo')
        elif labels[i] == 'Iris-virginica':
            axs[1].plot(values[i][0],values[i][1],'go')
        elif labels[i] == 'Iris-versicolor':
            axs[1].plot(values[i][0],values[i][1],'ro')
        else:
            axs[1].plot(values[i][0],values[i][1],'b^')
    
    axs[1].set_xlim(-4,4)   
 
    plt.show()

def get_a_sample(x, n):
    return (np.vstack([iris_values_2d[:n], iris_values_2d[50:50+n], iris_values_2d[100:100+n]]) , 
            np.vstack([iris_labels[:n], iris_labels[50:50+n], iris_labels[100:100+n]]).ravel() )

def chop(x, minimum, maximum, tolerance=1e-4):
    '''Chops a number when it is sufficiently close to the extreme of
   an enclosing interval.

Arguments:

- x: number to be possibily chopped
- minimum: left extreme of the interval containing x
- maximum: right extreme of the interval containing x
- tolerance: maximum distance in order to chop x

Returns: x if it is farther than tolerance by both minimum and maximum;
         minimum if x is closer than tolerance to minimum
         maximum if x is closer than tolerance to maximum

Throws:

- ValueError if minimum > maximum or if x does not belong to [minimum, maximum]

'''
    if minimum > maximum:
        raise ValueError('Chop: interval extremes not sorted')
    if  x < minimum or x > maximum:
        raise ValueError('Chop: value not belonging to interval')

    if x - minimum < tolerance:
        x = 0
    if maximum - x < tolerance:
        x = maximum
    return x

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
    Computes the squared distance between the image of a point and the center of the found sphere
    
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
    Computes the squared squared radius of the found sphere and a function returning
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

def build_clusters(x, index_v, index_sv, radius, d):
    
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
    
    couples = [[i,j] for i in index_v for j in index_sv]
    
    for c in couples:
        if check_couple(x[c[0]], x[c[1]], radius, d):
            G.add_edge(c[0],c[1])
            
    return [list(c) for c in list(nx.connected_components(G))]

def clustering(x, kernel, c, labels=[], graph=False):
    
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
    if(graph):
        print_graph(x, labels)
        
    k = kernel 
    
    betas = solve_wolf(x, k, c)
    
    index_bsv = []
    index_sv = []
    index_v = []
    
    #sistemare
    #---
    for i in range(len(betas)):
        if 0 < betas[i] < c:
            if(graph):
                plt.plot(x[i][0],x[i][1],'bo')
            index_sv.append(i)
        elif betas[i] == c:
            if(graph):
                plt.plot(x[i][0],x[i][1],'ro')
            index_bsv.append(i)
        else:
            if(graph):
                plt.plot(x[i][0],x[i][1],'go')
            index_v.append(i)
     #---
        
    radius, d = squared_radius_and_distance(x, betas, index_sv, k, c)
    
    if(graph):
        plt.show()
        print "number of support vectors: %d" %len(index_sv)
        
    return build_clusters(x, index_v, index_sv, radius, d), radius
    