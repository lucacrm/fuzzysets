from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def get_principal_components(data, n):
    
    '''
    reduces a data_set in its n principal components
    
    - data: data set upon wich extract the principal components
    - n: number of the wished components
    '''
    
    data = np.array(data)
    
    if len(data[0]) < n or n < 1:
        raise ValueError( 'Invalid number of components. It must be between 0 and data length')

    data_std = StandardScaler().fit_transform(data)
    pca_nd = PCA(n)
    return pca_nd.fit_transform(data_std)


def create_generator(data_set, n_components):
    a = -4 #capire
    b = 8 #capire
    
    def gen(m):
        return (a + np.random.random(n_components*m) * b).reshape((m, n_components))
    
    return gen


def get_random_sets_by_index(n, percentuals):
    
    '''
    create a random permutation of the numbers in the range from 0 to n-1 and divide the permutation in three sets according
    to the passed percentuals.
    
    n: range of the data
    percentuals: tuple (a,b,c) a+b+c must be 1
    '''
    
    if sum(percentuals) !=1:
        raise ValueError('The sum of the elements in the tuple of percentuals must be 1')
    
    indexes = range(n)
    permutation = np.random.permutation(indexes)
    
    perc_train, perc_val, perc_test = percentuals
    
    train_index = permutation[:int(n*perc_train)]
    validation_index = permutation[int(n*perc_train):int(n*(perc_train+perc_val))]
    test_index = permutation[int(n*(perc_train+perc_val)):]
    
    return train_index, validation_index, test_index


def get_random_sets(x, percentuals):
    
    '''
    create a random permutation of the data in x and divide the permutation in three sets according
    to the passed percentuals.
    
    x: data set
    percentuals: tuple (a,b,c) a+b+c must be 1
    '''
    if sum(percentuals) !=1:
        raise ValueError('The sum of the elements in the tuple of percentuals must be 1')
    
    n=len(x)
    permutation = np.random.permutation(x)
    perc_train, perc_val, perc_test = percentuals
    
    train = permutation[:int(n*perc_train)]
    validation = permutation[int(n*perc_train):int(n*(perc_train+perc_val))]
    test = permutation[int(n*(perc_train+perc_val)):]
    
    return train, validation, test


def get_different_labels(labels):
    
    d={}
    ok_index = []
    for j in range(len(labels)):
        if labels[j] not in d:
            ok_index.append(j)
            d[labels[j]] = True
            
    return ok_index


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

def pretty_results(results):
    
    pretty = results['name']+', number of iterations: '+str(results['iterations'])+'\n\n'
    pretty = pretty + 'Parameters info\n'
    pretty = pretty + 'Tradeoffs parameter: '+str(results['cs'])+'\n'
    pretty = pretty + 'Using same parameters for all the clusterization: '+str(results['same_c'])+'\n'
    pretty = pretty + 'Gaussian kernel sigmas: '+str(results['sigmas'])+'\n'
    pretty = pretty + 'Number of principal dimension considerated: '+str(results['dimensions'])+'\n'
    pretty = pretty + 'Dataset length: '+str(results['dataset-length'])+'\n\n'
    pretty = pretty + 'Clusterization info\n'
    d = int(results['dataset-length']*results['min_size'])
    pretty = pretty + 'Minimum size of a cluster: in perc: '+str(results['min_size'])+', in int: '+str(d+1)+'\n'
    pretty = pretty + 'Type of mu: '+results['muzzifier']+'\n'
    pretty = pretty + 'Fuzzifier: '+results['fuzzifier']+'\n'
    pretty = pretty + 'Cluster graph uses all connections: '+str(results['all_links'])+'\n'
    pretty = pretty + 'Cluster to label info\n'
    pretty = pretty + 'Force the number of cluster to be the number of different labels: '+str(results['force_num_fs'])+'\n'
    pretty = pretty + 'Force the labels to be always represent by a cluster: '+str(results['force_num_fs'])+'\n\n'
    pretty = pretty + 'Validation info\n'
    pretty = pretty + 'Conflict resolution: '+results['resolve_conflict']+'\n'
    pretty = pretty + 'Number of cluster considerated for validation (top_k): '+str(results['top_k'])+'\n\n\n'
    pretty = pretty + 'RESULTS\n\n'
    pretty = pretty + 'On TEST set\n\n'
    pretty = pretty + 'Accuracy on test set for every iteration (n_of_the_iter, best_c, best_sigma):\n'
    
    d = sorted(results['test-accuracies'])
    for x in d:
        pretty = pretty + str(x) + ': ' +  str(results['test-accuracies'][x]) +'\n'
        
    pretty = pretty + '\nLoss on test set for every iteration (n_of_the_iter, best_c, best_sigma):\n'
        
    d = sorted(results['test-losses'])
    for x in d:
        pretty = pretty + str(x) + ': ' +  str(results['test-losses'][x]) +'\n'
        
    pretty = pretty + '\nAccuracy mean :'+str(results['test-accuracy-mean'])+'\n'
    pretty = pretty + 'Std deviation :'+str(results['test-accuracy-std'])+'\n'
    pretty = pretty + 'Loss mean :'+str(results['test-loss-mean'])+'\n'
    pretty = pretty + 'Std deviation :'+str(results['test-loss-std'])+'\n\n'
    
    pretty = pretty + 'On TRAINING set\n\n'
    pretty = pretty + 'Accuracy on training set for every iteration (n_of_the_iter, c, sigma):\n'
    
    d = sorted(results['training-accuracies'])
    for x in d:
        pretty = pretty + str(x) + ': ' +  str(results['training-accuracies'][x]) +'\n'
        
    pretty = pretty + 'Loss on training set for every iteration (n_of_the_iter, best_c, best_sigma):\n'
        
    d = sorted(results['training-losses'])
    for x in d:
        pretty = pretty + str(x) + ': ' + str(results['training-losses'][x]) +'\n'
    
    pretty = pretty + '\nAccuracy mean :'+str(results['training-accuracy-mean'])+'\n'
    pretty = pretty + 'Std deviation :'+str(results['training-accuracy-std'])+'\n'
    pretty = pretty + 'Loss mean :'+str(results['training-loss-mean'])+'\n'
    pretty = pretty + 'Std deviation :'+str(results['training-loss-std'])+'\n\n'
    
    pretty = pretty + 'On ALL the sets\n\n'
    pretty = pretty + 'Accuracy on all the set for every iteration, for every couple c,sigma (n_of_the_iter, c, sigma):\n'
    
    d = sorted(results['all-accuracies'])
    for x in d:
        pretty = pretty + str(x) + ': ' +  str(results['all-accuracies'][x]) +'\n'
        
    pretty = pretty + '\nLoss on all the set for every iteration, for every couple c,sigma (n_of_the_iter, best_c, best_sigma):\n'
        
    d = sorted(results['all-losses'])
    for x in d:
        pretty = pretty + str(x) + ': ' + str(results['all-losses'][x]) +'\n'
    
    pretty = pretty + '\nAccuracy mean :'+str(results['all-accuracy-mean'])+'\n'
    pretty = pretty + 'Std deviation :'+str(results['all-accuracy-std'])+'\n'
    pretty = pretty + 'Loss mean :'+str(results['all-loss-mean'])+'\n'
    pretty = pretty + 'Std deviation :'+str(results['all-loss-std'])+'\n\n'
    
    return pretty

def associate_labels_to_colors(y):

    d1={}
    d2={}
    color_list = ['b','g','r','c','m','y','k']
    color_map_list = [cm.Blues, cm.Greens, cm.Reds]
    set_y = sorted(list(set(y)))
    for (a,b) in zip(set_y,color_list):
        d1[a] = b
    for (c,d) in zip(set_y,color_map_list):
        d2[c] = d
    return d1, d2


def gr_dataset(x, y, colors): 
    for e in colors:
        plt.scatter(x[y==e, 0],
                    x[y==e, 1],
                    label=e,
                    c=colors[e])
    
    
def gr_cluster_division(x, y, membership_functions, function_labels, i, colors, sigma, c, c1):
    
    fig = plt.figure(figsize=(12,4))
    plt.text(-4,-6,"sigma: "+str(sigma)+", c: "+str(c)+", c1: "+str(c1))
    plt.suptitle('Cluster division '+str(i))
    
    plt.subplot(121)
    gr_dataset(x, y, colors)
    plt.title('Dataset by known labels')
    plt.legend()
    plt.xlim(-4,4)
    plt.ylim(-4,4)
        
    function_labels = np.array(function_labels)
    new_y = function_labels[[np.argmax([mf(p) for mf in membership_functions]) for p in x]]
    
    plt.subplot(122)
    gr_dataset(x, new_y, colors)
    plt.title('Dataset by obtained clusters')
    plt.legend()
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    
    
def gr_membership_contour(x, y, estimated_membership, label, i, colors, sigma, c, c1):
    
    gr_dataset(x, y, colors)
 
    plt.title('CLUSTER '+str(i)+'  '+label)
    plt.text(-4,-6,"sigma: "+str(sigma)+", c: "+str(c)+", c1: "+str(c1))
    
    a = np.arange(-4, 4, .1)
    b = np.arange(-4, 4, .1)
    A, B = np.meshgrid(a, b)
    zs = np.array([estimated_membership((a, b))
                   for a,b in zip(np.ravel(A), np.ravel(B))])
    Z = zs.reshape(A.shape)

    membership_contour = plt.contour(A, B, Z,
                                     levels=(.1, .3, .5, .7, .8, .9, .95), colors='k')
    plt.clabel(membership_contour, inline=1)
    plt.legend()

def gr_membership_graded(x, y, estimated_membership, label, i, colors, colors_map, sigma, c, c1):
    
    gr_dataset(x, y, colors)
    
    plt.title('CLUSTER '+str(i)+'  '+label)
    plt.text(-4,-6,"sigma: "+str(sigma)+", c: "+str(c)+", c1: "+str(c1))
    
    a = np.arange(-4, 4, .1)
    b = np.arange(-4, 4, .1)
    A, B = np.meshgrid(a, b)
    zs = np.array([estimated_membership((a, b))
                   for a,b in zip(np.ravel(A), np.ravel(B))])
    Z = zs.reshape(A.shape)
    plt.imshow(Z, interpolation='bilinear', origin='lower',
           cmap=colors_map[label], alpha=0.99, extent=(-4, 4, -4, 4))
    plt.legend()
    
def gr_save(filename):
    plt.savefig(filename+".png")
    plt.clf()