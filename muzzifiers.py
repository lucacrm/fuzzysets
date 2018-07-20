import numpy as np
from support_vector_clustering import *

class BaseMuzzifier(object):
    def __init__(self):
        pass

        def get_mus(self, x, clusters, kernel, c, gen):
            raise NotImplementedError(
            'the base class does not implement get_mus method')
            
class BinaryMuzzifier(BaseMuzzifier):
    def __init__(self):
        self.name = 'Binary Muzzifier'
        self.latex_name = '$\\hat\\mu_{\\text{binary muzzifier}}$'

    def get_mus(self, x, clusters, kernel, c, gen):
        return [[1 if point in cl else 0 for point in x] for cl in clusters]
        
        
    def __repr__(self):
        return 'BinaryMuzzifier()'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True
            
class LinearMuzzifier(BaseMuzzifier):
    def __init__(self):
        self.name = 'Linear Muzzifier'
        self.latex_name = '$\\hat\\mu_{\\text{linear muzzifier}}$'

    def get_mus(self, x, clusters, kernel, c, gen):
        
        mus = []
        
        sample = gen(1500)
        
        #print 'the sample is ', sample
        
        for cl in clusters:
            #print '\n\ncluster ', cl
            
            betas = solve_wolf(cl, kernel, c)
    
            index_bsv = []
            index_sv = []
            index_v = []

            for i in range(len(betas)):
                if 0 < betas[i] < c:
                    index_sv.append(i)

            radius, d = squared_radius_and_distance(cl, betas, index_sv, kernel, c, mean=False)
            
            #print 'radius is ', radius
            
            max_distance = np.max(map(d,sample))
            #print 'max distance is ', max_distance
            
            '''
            for point in x:
                d_from_cl = d(point)
                print 'point ', point
                print 'distance from cluster is ', d(point)
            '''
            
            curr_mu = [1 if d(point) <= radius else 0 if d(point) >= max_distance 
                       else (max_distance - d(point))/(max_distance - radius) for point in x]
            
            #print '\ncurrent mus ', curr_mu
            
            mus.append(curr_mu)
            
        return mus
                

    def __repr__(self):
        return 'LinearMuzzifier()'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True