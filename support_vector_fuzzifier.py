from support_vector_clustering import *
from possibilearn import *
from possibilearn.kernel import GaussianKernel
import skfuzzy as fuzz
from muzzifiers import *
import matplotlib.pyplot as plt

def get_principal_components(data, n):
    
    '''
    extract the n principal components of a data set
    
    - data: data set upon wich extract the principal components
    - n: number of the wished components
    '''
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    data = np.array(data)
    
    if len(data[0]) < n or n < 1:
        raise ValueError( 'Invalid number of components. It must be between 0 and data length')

    data_std = StandardScaler().fit_transform(data)
    pca_nd = PCA(n)
    return pca_nd.fit_transform(data_std)

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


def learn_fs(x, c=1, kernel=GaussianKernel(1), min_size=0.01, fuzzifier=LinearFuzzifier(), muzzifier=BinaryMuzzifier()):

    '''
    Divides a set of data into clusters and it computes a membership function for all found clusters
    
    - x: set of points
    - y: labels of points
    - c: tradeoff value
    - kernel: kernel for the trasformation of data in an high-dimensional space
    - min_size: lower bound for the dimension of a cluster (in percentage of the size of the data set)
    - fuzzifier: function to be used in order to get membership values of
             points falling outside the crisp set
    - mu_f: function to be used in order to set a value of mu, representing distance of a point to a cluster.
    
    '''
    
    x_len=len(x)
    
    clusters_index, r = clustering(x, kernel, c) #clusterize data using indexes

    clusters_index = [ cl for cl in clusters_index if len(cl) > x_len*min_size ] #exclude small clusters

    clusters_index.sort(key = lambda x: -len(x)) #sort clusters by len in order to get bigger cluster in the associate_fs_to_labels function with force parameters set to true

    clusters = [x[cl] for cl in clusters_index] #clusterize real data (data of x)
    print 'clusters by index ', clusters_index
    print 'number of clusters ', len(clusters)

    #compute initial mus for compute the estimated memberships
    gn = create_generator(x, len(x[0]))
    
    mus = muzzifier.get_mus(x, clusters, kernel, c, gn)
    if None in mus:
        raise ValueError('Unable to calculate mus')
    #print 'mus ', mus
    print 'inferring membership functions'
    
    estimated_memberships = []

    for i in range(len(clusters)):
        estimated_membership, _ = possibility_learn(x,
                                          mus[i],
                                          c=c,
                                          k=kernel,
                                          fuzzifier=fuzzifier,
                                          sample_generator=gn)
        estimated_memberships.append(estimated_membership)
        
    if None in estimated_memberships:
        raise ValueError('Unable to infer membership functions')
        
    return estimated_memberships, clusters_index


def associate_fs_to_labels(x, y, membership_functions, force_num_fs=False, force_labels_repr=False):
    
    '''
    associate every membership function to a fuzzy set
    
    - x: set of data point
    - y: labels of the data points
    - membership_functions: list of membership function
    - force_num_fs: if true force the number of fuzzy sets to be equal to the number of different labels
    - force_labels_repr: if true force each label to have at least one fuzzy set that represent it
    
    '''
    
    labels = set(y)
    #print 'possible labels ', labels
    n_labels = len(labels)
    
    if (len(membership_functions) < n_labels) and force_num_fs:
            raise ValueError('Number of functions is less than the number of labels, cant force the number of functions')
    
    fuzzy_memberships = [[mf(p) for mf in membership_functions] for p in x]
    #print 'fuzzy memberships ', fuzzy_memberships
    
    best_memberships = [np.argmax(m) for m in fuzzy_memberships]
    #print "best memberships per points ", best_memberships
    
    labels_founded = np.array([ [ y[i] for i in range(len(y)) if best_memberships[i] == k ] for k in range(len(membership_functions)) ])
    
    #exclude subclusters / associate same clusters
    membership_functions = [x[0] for x in zip(membership_functions,labels_founded) if len(x[1])>0]
    labels_founded = [x for x in labels_founded if len(x)>0]
    
    #print 'labels founded ', labels_founded
    
    function_labels = [pd.Series(lf).mode()[0] for lf in labels_founded]
    
    if force_labels_repr:
        if len(set(function_labels)) != n_labels:
            raise ValueError('Some of the labels dont represent a cluster')
            
        else:
            if force_num_fs:
                i = get_different_labels(function_labels)
                return list(np.array(membership_functions)[i]), list(np.array(function_labels)[i])
            else:
                return membership_functions, function_labels
             
    if force_num_fs and not(force_labels_repr):
        return membership_functions[:n_labels], function_labels[:n_labels]
        
    if not(force_num_fs) and not(force_labels_repr):
        return membership_functions, function_labels
    
def binary_loss(x,correct_labels,guessed_labels):
    return float(sum(guessed_labels != correct_labels)) / len(x)
    
def validate_fs(x, y, membership_functions, function_labels, resolve_conflict='random', loss=binary_loss):
    
    def best_choice(candidates, function_labels, real_labels):
        choices = []
        for i in range(len(candidates)):
            found = False
            for j in range (len(candidates[i])):
                if function_labels[candidates[i][j]] == real_labels[i] and (not found) :
                    choices.append(function_labels[candidates[i][j]])
                    found = True
            if not found:
                choices.append(function_labels[np.random.choice(candidates[i])])
                
        return np.array(choices)
    
    
    def worst_choice(candidates, function_labels, real_labels):
        choices = []
        for i in range(len(candidates)):
            found = False
            for j in range (len(candidates[i])):
                if function_labels[candidates[i][j]] != real_labels[i] and (not found) :
                    choices.append(function_labels[candidates[i][j]])
                    found = True
            if not found:
                choices.append(function_labels[np.random.choice(candidates[i])])
                
        return np.array(choices)
                           
    function_labels = np.array(function_labels)
    results = np.array([[f(p) for f in membership_functions] for p in x])
    maxs = np.array([np.max(r) for r in results])
    results = [pd.Series(r) for r in results]
    candidates = []
    for i in range(len(results)):
        candidates.append(results[i][results[i]==maxs[i]].keys().values)
        
    if resolve_conflict == 'random':
        guessed_labels = function_labels[[np.random.choice(c) for c in candidates]]
          
    elif resolve_conflict == 'best':
        guessed_labels = best_choice(candidates,function_labels,y)
        
    elif resolve_conflict == 'worst':
        guessed_labels = worst_choice(c,function_labels,y)
    
    else:
        raise ValueError('the value of resolve_conflict must be random, best or worst')

    correct_labels = np.array(y)
    
    acc = float(sum(guessed_labels == correct_labels)) / len(x)
                    
    if loss is None:
        return acc, None
    else:
        lss = loss(x,correct_labels,guessed_labels)
        return acc, lss
    
def iterate_tests(x, y, cs, sigmas, iterations, dim=2, seed=None, min_size=0.01, fuzzifier=LinearFuzzifier(), muzzifier=BinaryMuzzifier(), force_num_fs=False, force_labels_repr=False, resolve_conflict='random', loss=binary_loss, 
                  name='ITERATE TESTS', save_graph=False):
    
    '''
    Iterate the procedure of clustering and membership with different combination of parameters
    
    - x: set of data point
    - y: labels of the data points
    - cs: array of tradeoff values
    - sigmas: array of GaussianKernel parameters
    - n: numbers of iteration
    - dim: number of principal components to consider
    - seed: seed for initialize the numpy random generator, if no seed is specified a random seed is choose
    '''
    
    #get the wished principal components
    values = get_principal_components(x,dim)
    
    #get the couples of c,sigma to test
    couples = [(c,s) for s in sigmas for c in cs]
    
    #initialize the random generator with seed
    if seed is None:
        seed = np.random.randint(0,2**32)   
    np.random.seed(seed)
    
    #initialize parameters for the iterations
    test_accuracies={}
    training_accuracies={}
    test_losses={}
    training_losses={}
    all_accuracies={}
    all_losses={}
    valid_iteration = 0
    
    #associate every label to a color for graphs consistency
    colors, colors_map = associate_labels_to_colors(y)
    
    #iterations
    while(valid_iteration < iterations):
        print '\n\nholdout iteration {}'.format(valid_iteration)
        
        #get a random permutation of data and divide it in train, validation and test set
        train_index, validation_index, test_index = get_random_sets_by_index(len(values),(0.8,0.1,0.1)) #perc as a parameter?
        train_set = values[[i for i in train_index]]
        validation_set = values[[i for i in validation_index]]
        test_set = values[[i for i in test_index]]
        
        # find the couple that have the best accuracy for this permutation
        best_couples = [] #initialization of best couples
        best_accuracy = -1 #initialization of best accuracy
        
        for (c, sigma) in couples:
            print '\nmodel selection: trying parameters c={}, sigma={}'.format(c, sigma)
            print 'starting clusterization'
            
            #clustering and compute a membership function for each cluster
            try:
                mf, clusters_index = learn_fs(train_set, c=c, 
                             kernel=GaussianKernel(sigma), min_size=min_size, fuzzifier=fuzzifier, muzzifier=muzzifier)
                
                #print 'clusters with index ', clusters_index
            
                #associate membership functions to labels
                membership_functions, function_labels = associate_fs_to_labels(train_set, y[[i for i in train_index]], mf, force_num_fs=force_num_fs, force_labels_repr=force_labels_repr)
                
                print 'associated labels', function_labels

                #calculate the accuracy on validation set
                accuracy, lss = validate_fs(validation_set, y[[i for i in validation_index]], 
                    membership_functions, function_labels, loss=loss)
                
                print 'accuracy: ', accuracy
                
                all_accuracies[(valid_iteration, c, sigma)] = accuracy
                all_losses[(valid_iteration, c, sigma)] = lss
                
                #check the best accuracy
                if accuracy == best_accuracy:
                    best_couples.append((c,sigma))
                elif accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_couples = [(c,sigma)]
                   
            except ValueError as e:
                print str(e)
                continue
        
        if len(best_couples) > 0:
            
            #random choice between the couples with best accuracy
            c_best, sigma_best = best_couples[np.random.randint(len(best_couples))]
            print '\nbest couple', (c_best, sigma_best)
            
            #with the couple with the best accuracy infer the membership function merging train and validation set
            new_train_set = np.vstack((train_set, validation_set))
            new_train_index = np.hstack((train_index, validation_index))
            
            try:
                mf, clusters_index = learn_fs(new_train_set, c=c_best, 
                             kernel=GaussianKernel(sigma_best), min_size=min_size, fuzzifier=fuzzifier)
                print 'first clusters', clusters_index
            
                #associate membership functions to labels
                membership_functions, function_labels = associate_fs_to_labels(new_train_set, y[[i for i in new_train_index]], mf, force_num_fs=force_num_fs, force_labels_repr=force_num_fs)
                
                print 'associated labels', function_labels
                
                #graphs
                if save_graph and dim==2:
                    
                    gr_cluster_division(new_train_set, y[[i for i in new_train_index]],
                                        membership_functions, function_labels, valid_iteration, colors, sigma_best, c_best)
                    gr_save(name+"_"+str(valid_iteration)+"it_clusters_division")
                    
                    for (f,l,j) in zip(membership_functions, function_labels, range(len(function_labels))):
                        
                        gr_membership_contour(new_train_set, y[[i for i in new_train_index]], f, l, j,
                                              colors, sigma_best, c_best)
                        gr_save(name+"_"+str(valid_iteration)+"it_cluster_"+str(j)+"_"+l+"_"+"contour")
                        
                        gr_membership_graded(new_train_set, y[[i for i in new_train_index]], f, l, j,
                                             colors, colors_map, sigma_best, c_best)
                        gr_save(name+"_"+str(valid_iteration)+"it_cluster_"+str(j)+"_"+l+"_"+"heat")
                        

                #calculate the accuracy of the best couple on test set
                accuracy, lss = validate_fs(test_set, y[[i for i in test_index]], 
                    membership_functions, function_labels, loss=loss)
                
                print 'test accuracy: ', accuracy
                
                test_accuracies[(valid_iteration, c_best, sigma_best)] = accuracy
                test_losses[(valid_iteration, c_best, sigma_best)] = lss
                
                #calculate the accuracy of the best couple on training set
                accuracy, lss = validate_fs(new_train_set, y[[i for i in new_train_index]], 
                    membership_functions, function_labels)
                
                print 'training accuracy: ', accuracy
                
                training_accuracies[(valid_iteration, c_best, sigma_best)] = accuracy
                training_losses[(valid_iteration, c_best, sigma_best)] = lss
                
                #at this point the iteration is valid
                valid_iteration = valid_iteration + 1
                print 'iteration is valid'
                
            except ValueError as e:
                print str(e)
                continue
            
        else:
            print 'no best couple found iteration invalid '
            continue
            
    #build the results
    results = {}
    results['name'] = name
    results['cs'] = cs
    results['sigmas'] = sigmas
    results['iterations']= iterations
    results['dimensions'] = dim 
    results['min_size'] = min_size
    results['dataset-length'] = len(x)
    results['muzzifier'] = muzzifier.name
    results['fuzzifier'] = fuzzifier.name
    results['force_num_fs'] = force_num_fs
    results['force_labels_repr'] = force_labels_repr
    results['resolve_conflict'] = resolve_conflict
    results['seed'] = seed
    results['test-accuracies'] = test_accuracies
    results['training-accuracies'] = training_accuracies
    results['all-accuracies'] = all_accuracies
    results['test-accuracy-mean'] = np.array(test_accuracies.values()).mean()
    results['test-accuracy-std'] = np.array(test_accuracies.values()).std()
    results['training-accuracy-mean'] = np.array(training_accuracies.values()).mean()
    results['training-accuracy-std'] = np.array(training_accuracies.values()).std()
    results['all-accuracy-mean'] = np.array(all_accuracies.values()).mean()
    results['all-accuracy-std'] = np.array(all_accuracies.values()).std()
    
    if loss is None:
        results['test-losses'] = 'None'
        results['training-losses'] = 'None'
        results['all-losses'] = 'None'
        results['test-loss-mean'] = 'None'
        results['test-loss-std'] = 'None'
        results['training-loss-mean'] = 'None'
        results['training-loss-std'] = 'None'
        results['all-loss-mean'] = 'None'
        results['all-loss-std'] = 'None'
    else:
        results['test-losses'] = test_losses
        results['training-losses'] = training_losses
        results['all-losses'] = all_losses
        results['test-loss-mean'] = np.array(test_losses.values()).mean()
        results['test-loss-std'] = np.array(test_losses.values()).std()
        results['training-loss-mean'] = np.array(training_losses.values()).mean()
        results['training-loss-std'] = np.array(training_losses.values()).std()
        results['all-loss-mean'] = np.array(all_losses.values()).mean()
        results['all-loss-std'] = np.array(all_losses.values()).std()
    
    return results

def pretty_results(results):
    
    pretty = results['name']+', number of iterations: '+str(results['iterations'])+'\n\n'
    pretty = pretty + 'Parameters info\n'
    pretty = pretty + 'Tradeoffs parameter: '+str(results['cs'])+'\n'
    pretty = pretty + 'Gaussian kernel sigmas: '+str(results['sigmas'])+'\n'
    pretty = pretty + 'Number of principal dimension considerated: '+str(results['dimensions'])+'\n'
    pretty = pretty + 'Dataset length: '+str(results['dataset-length'])+'\n\n'
    pretty = pretty + 'Clusterization info\n'
    d = int(results['dataset-length']*results['min_size'])
    pretty = pretty + 'Minimum size of a cluster: in perc: '+str(results['min_size'])+', in int: '+str(d+1)+'\n'
    pretty = pretty + 'Type of mu: '+results['muzzifier']+'\n'
    pretty = pretty + 'Fuzzifier: '+results['fuzzifier']+'\n'
    pretty = pretty + 'Cluster to label info\n'
    pretty = pretty + 'Force the number of cluster to be the number of different labels: '+str(results['force_num_fs'])+'\n'
    pretty = pretty + 'Force the labels to be always represent by a cluster: '+str(results['force_num_fs'])+'\n\n'
    pretty = pretty + 'Validation info\n'
    pretty = pretty + 'Conflict resolution: '+results['resolve_conflict']+'\n\n\n'
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
    
def gr_cluster_division(x, y, membership_functions, function_labels, i, colors, sigma, c):
    
    fig = plt.figure(figsize=(12,4))
    plt.text(-4,-6,"sigma: "+str(sigma)+", c: "+str(c))
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
    
def gr_membership_contour(x, y, estimated_membership, label, i, colors, sigma, c):
    
    gr_dataset(x, y, colors)
 
    plt.title('CLUSTER '+str(i)+'  '+label)
    plt.text(-4,-6,"sigma: "+str(sigma)+", c: "+str(c))
    
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

def gr_membership_graded(x, y, estimated_membership, label, i, colors, colors_map, sigma, c):
    
    gr_dataset(x, y, colors)
    
    plt.title('CLUSTER '+str(i)+'  '+label)
    plt.text(-4,-6,"sigma: "+str(sigma)+", c: "+str(c))
    
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
    
    
'''
def iterate_tests_vs_fuzzycmeans(x, y, cs, sigmas, iterations, dim=2, seed=None, min_size=0.01, fuzzifier=LinearFuzzifier(), mu_function=None, force_num_fs=False, force_labels_repr=False, resolve_conflict='random', loss=None):
    
    #get the wished principal components
    values = get_principal_components(x,dim)
    
    #get the couples of c,sigma to test
    couples = [(c,s) for s in sigmas for c in cs]
    
    #initialize the random generator with seed
    if seed is None:
        seed = np.random.randint(0,2**32)   
    np.random.seed(seed)
    
    #initialize parameters for the iterations
    test_accuracies={}
    training_accuracies={}
    test_accuracies_cmeans={}
    training_accuracies_cmeans={}
    all_accuracies={}
    valid_iteration = 0
    
    #iterations
    while(valid_iteration < iterations):
        print '\n\nholdout iteration {}'.format(valid_iteration)
        
        #get a random permutation of data and divide it in train, validation and test set
        train_index, validation_index, test_index = get_random_sets_by_index(len(values),(0.8,0.1,0.1))
        train_set = values[[i for i in train_index]]
        validation_set = values[[i for i in validation_index]]
        test_set = values[[i for i in test_index]]
        
        # find the couple that have the best accuracy for this permutation
        best_couples = [] #initialization of best couples
        best_accuracy = -1 #initialization of best accuracy
        
        for (c, sigma) in couples:
            print '\nmodel selection: trying parameters c={}, sigma={}'.format(c, sigma)
            print 'starting clusterization'
            
            #clustering and compute a membership function for each cluster
            try:
                mf, clusters_index = learn_fs(train_set, c=c, 
                             kernel=GaussianKernel(sigma), min_size=min_size, fuzzifier=fuzzifier)
                
                print 'first clusters', clusters_index
            
                #associate membership functions to labels
                membership_functions, function_labels = associate_fs_to_labels(train_set, y[[i for i in train_index]], mf, force_num_fs=force_num_fs, force_labels_repr=force_labels_repr)
                
                print 'associated labels', function_labels

                #calculate the accuracy on validation set
                accuracy = validate_fs(validation_set, y[[i for i in validation_index]], 
                    membership_functions, function_labels)
                
                print 'accuracy: ', accuracy
                
                all_accuracies[(valid_iteration, c, sigma)] = accuracy
                
                #check the best accuracy
                if accuracy == best_accuracy:
                    best_couples.append((c,sigma))
                elif accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_couples = [(c,sigma)]
                   
            except ValueError as e:
                print str(e)
                continue
        
        if len(best_couples) > 0:
            
            #random choice between the couples with best accuracy
            c_best, sigma_best = best_couples[np.random.randint(len(best_couples))]
            print '\nbest couple', (c_best, sigma_best)
            
            #with the couple with the best accuracy infer the membership function merging train and validation set
            new_train_set = np.vstack((train_set, validation_set))
            new_train_index = np.hstack((train_index, validation_index))
            
            try:
                mf, clusters_index = learn_fs(new_train_set, c=c_best, 
                             kernel=GaussianKernel(sigma_best), min_size=min_size, fuzzifier=fuzzifier)
                print 'first clusters', clusters_index
            
                #associate membership functions to labels
                membership_functions, function_labels = associate_fs_to_labels(new_train_set, y[[i for i in new_train_index]], mf, force_num_fs=force_num_fs, force_labels_repr=force_num_fs)
                
                print 'associated labels', function_labels

                #calculate the accuracy of the best couple on test set
                accuracy = validate_fs(test_set, y[[i for i in test_index]], 
                    membership_functions, function_labels)
                
                print 'test accuracy: ', accuracy
                
                test_accuracies[(valid_iteration, c_best, sigma_best)] = accuracy
                
                #calculate the accuracy of the best couple on training set
                accuracy = validate_fs(new_train_set, y[[i for i in new_train_index]], 
                    membership_functions, function_labels)
                
                print 'training accuracy: ', accuracy
                
                training_accuracies[(valid_iteration, c_best, sigma_best)] = accuracy
                
                print 'starting model against cmeans'
                
                #vs fuzzycmeans
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    new_train_set.T, len(set(y)), 2, error=0.005, maxiter=1000, init=None)
                
                #assign a label to each cluster
                cluster_membrships = cluster_membership = np.argmax(u, axis=0)
                labelled_clusters = [l.mode()[0] for l in [pd.Series(ls) for ls in [[b for (a,b) in zip(cluster_membership,
                                                y[[j for j in new_train_index]]) if a == i] for i in range(len(set(y)))]]]
                
                #compute accuracy on training set
                accuracy = float(len(new_train_set[[labelled_clusters[i] for i in cluster_membership]
                             == y[[i for i in new_train_index]]])) / float(len(new_train_set))
                
                training_accuracies_cmeans[(valid_iteration, c_best, sigma_best)] = accuracy
                
                print 'training accuracy with fuzzy cmeans: ', accuracy
                
                #predict on test set
                u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
                    test_set.T, cntr, 2, error=0.005, maxiter=1000)
                
                #compute accuracy on test set
                cluster_membership = np.argmax(u, axis=0)
                
                accuracy = float(len(test_set[[labelled_clusters[i] for i in cluster_membership]
                             == y[[j for j in test_index]]])) / float(len(test_set))
                
                test_accuracies_cmeans[(valid_iteration, c_best, sigma_best)] = accuracy
                
                print 'test accuracy with fuzzy cmeans: ', accuracy
                
                #at this point the iteration is valid
                valid_iteration = valid_iteration + 1
                print 'iteration is valid'
                
            except ValueError as e:
                print str(e)
                continue
            
        else:
            print 'no best couple found, iteration invalid '
            continue
            
    #build the results
    results = {}
    results['seed'] = seed
    results['test-accuracies'] = test_accuracies
    results['training-accuracies'] = training_accuracies
    results['test-accuracies-cmeans'] = test_accuracies_cmeans
    results['training-accuracies-cmeans'] = training_accuracies_cmeans
    results['all-accuracies'] = all_accuracies
    results['test-mean'] = np.array(test_accuracies.values()).mean()
    results['test-std'] = np.array(test_accuracies.values()).std()
    results['training-mean'] = np.array(training_accuracies.values()).mean()
    results['training-std'] = np.array(training_accuracies.values()).std()
    results['test-mean-cmeans'] = np.array(test_accuracies_cmeans.values()).mean()
    results['test-std-cmeans'] = np.array(test_accuracies_cmeans.values()).std()
    results['training-mean-cmeans'] = np.array(training_accuracies_cmeans.values()).mean()
    results['training-std-cmeans'] = np.array(training_accuracies_cmeans.values()).std()
    results['all-mean'] = np.array(all_accuracies.values()).mean()
    results['all-std'] = np.array(all_accuracies.values()).std()

    return results
    
'''