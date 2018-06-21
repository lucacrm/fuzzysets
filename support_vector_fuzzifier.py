from support_vector_clustering import *
from possibilearn import *
from possibilearn.kernel import GaussianKernel

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


def learn_fs(x, c=1, kernel=GaussianKernel(1), min_size=0.01, fuzzifier=LinearFuzzifier(), mu_function=None):

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

    #compute initial mus for compute the estimated memberships (update: do it with a fuzzifier)
    mus = [[1 if i in ci else 0 for i in range(x_len)] for ci in clusters_index ]
    
    estimated_memberships = []

    for i in range(len(clusters)):
        estimated_membership, _ = possibility_learn(x,
                                          mus[i],
                                          c=c,
                                          k=kernel,
                                          fuzzifier=fuzzifier,
                                          sample_generator=g)
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
    n_labels = len(labels)
    
    if (len(membership_functions) < n_labels) and force_num_fs:
            raise ValueError('Number of functions is less than the number of labels, cant force the number of functions')
    
    fuzzy_memberships = [[mf(p) for mf in membership_functions] for p in x]
    
    best_memberships = [np.argmax(m) for m in fuzzy_memberships]
    
    labels_founded = np.array([ [ y[i] for i in range(len(y)) if best_memberships[i] == k ] for k in range(len(membership_functions)) ])
    
    function_labels = [pd.Series(lf).mode()[0] for lf in labels_founded]
    
    if force_labels_repr:
        if len(set(function_labels)) != n_labels:
            raise ValueError('Some of the labels dont represent a cluster') #eccezione o forzare assegnamento a altre labels? fino adesso abbiamo fatto eccezione/test invalido
            
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
    
    
def validate_fs(x, y, membership_functions, function_labels, resolve_conflict=None, loss=None):
    
    guessed_labels = np.array([function_labels[np.argmax([ e(t) for e in membership_functions ])] for t in x])

    correct_labels = np.array(y)
                    
    return float(sum(guessed_labels == correct_labels)) / len(x)


def iterate_tests(x, y, cs, sigmas, iterations, dim=2, seed=None, min_size=0.01, fuzzifier=LinearFuzzifier(), mu_function=None, force_num_fs=False, force_labels_repr=False, resolve_conflict=None, loss=None):
    
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
    results['seed'] = seed
    results['test-accuracies'] = test_accuracies
    results['training-accuracies'] = training_accuracies
    results['all-accuracies'] = all_accuracies
    results['test-mean'] = np.array(test_accuracies.values()).mean()
    results['test-std'] = np.array(test_accuracies.values()).std()
    results['training-mean'] = np.array(training_accuracies.values()).mean()
    results['training-std'] = np.array(training_accuracies.values()).std()
    results['all-mean'] = np.array(all_accuracies.values()).mean()
    results['all-std'] = np.array(all_accuracies.values()).std()

    return results