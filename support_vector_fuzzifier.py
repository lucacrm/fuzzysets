import utils as ut
import support_vector_clustering as svc
from muzzifiers import *
from possibilearn import *
from possibilearn.kernel import GaussianKernel

def learn_fs(x, c, c1, kernel, min_size, fuzzifier, muzzifier, all_links):

    '''
    Divides a set of data into clusters and computes a membership function for all of it
    
    - x: set of points
    - c: trade-off value in 1st level clustering
    - c1: trade-off value in 2nd level clustering
    - kernel: kernel function to be used
    - min_size: lower bound for the dimension of a cluster (in percentage of the size of the data set)
    - fuzzifier: function to be used in order to get membership values of points falling outside the crisp set
    - muzzifier: function to be used in order to set a value of mu, representing distance of a point to a cluster.
    
    '''
    
    x_len=len(x)
    
    clusters_index, r = svc.clustering(x, kernel, c, all_links) #clusterize data using indexes

    clusters_index = [ cl for cl in clusters_index if len(cl) > x_len*min_size ] #exclude small clusters

    clusters_index.sort(key = lambda x: -len(x)) #sort clusters by len in order to get bigger cluster in the associate_fs_to_labels function with force parameters set to true

    clusters = [x[cl] for cl in clusters_index] #clusterize real data (data of x)
    print 'clusters by index ', clusters_index
    print 'number of clusters ', len(clusters)

    gn = ut.create_generator(x, len(x[0]))
    print 'inferring mus'
    mus = muzzifier.get_mus(x, clusters, kernel, c1, gn)
    if None in mus:
        raise ValueError('Unable to calculate mus')
        
    print 'inferring membership functions'
    estimated_memberships = []

    for i in range(len(clusters)):
        estimated_membership, _ = possibility_learn(x,
                                          mus[i],
                                          c=c1,
                                          k=kernel,
                                          fuzzifier=fuzzifier,
                                          sample_generator=gn)
        estimated_memberships.append(estimated_membership)
        
    if None in estimated_memberships:
        raise ValueError('Unable to infer membership functions')
        
    return estimated_memberships, clusters_index


def associate_fs_to_labels(x, y, membership_functions, force_num_fs, force_labels_repr):
    
    '''
    associates each membership function to a fuzzy set
    
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
                i = ut.get_different_labels(function_labels)
                return list(np.array(membership_functions)[i]), list(np.array(function_labels)[i])
            else:
                return membership_functions, function_labels
             
    if force_num_fs and not(force_labels_repr):
        return membership_functions[:n_labels], function_labels[:n_labels]
        
    if not(force_num_fs) and not(force_labels_repr):
        return membership_functions, function_labels
    
    
def binary_loss(x, correct_labels, guessed_labels):
    
    '''
    loss function that returns loss as uncorrect/tot
    
    x: set of points
    correct_labels: array of the correct labels for points in x
    guessed_labels: array of the guessed labels for points in x
    '''
    return float(sum(guessed_labels != correct_labels)) / len(x)

def k_binary_loss(x, correct_labels, topk_labels):
    
    '''
    loss function that returns loss as uncorrect/tot
    
    x: set of points
    correct_labels: array of the correct labels for points in x
    guessed_labels: array of the guessed labels for points in x, k labels for each point
    '''
    
    guessed_labels = zip (correct_labels,topk_labels)
    bool_accuracy = np.array([x[0] in x[1] for x in guessed_labels])
    lss = 1- (float(np.sum(bool_accuracy))/len(bool_accuracy))
    return lss
    
def best_choice(candidates, function_labels, real_labels):
    
    '''
    choses the correct label between candidates. if the candidates are all wrong, choses randomly.
    
    candidates: array of candidate point labels, various candidates for each point
    function_labels: array of founded clusters labels
    real_labels: array of known labels for each point
    
    returns the array of chosen labels
    '''
    
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
    
    '''
    chooses the uncorrect label between candidates. if the candidates are all correct, chooses randomly.
    
    candidates: array of candidate point labels, various candidates for each point
    function_labels: array of founded clusters labels
    real_labels: array of known labels for each point
    
    returns the array of chosen labels
    '''
    
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


def validate(x, y, membership_functions, function_labels, resolve_conflict, loss):

    '''
    validate the passed model
    
    x: set of points
    y: set of point labels
    membership_functions: set of membership functions
    function_labels: set of function labels
    resolve_conflict: string in ("best", "worst", "random"), select how resolve conflict
    loss: loss function to be used loss(x, correct_labels, guessed_labels)
    '''
    
    function_labels = np.array(function_labels)
    print 'function_labels\n', function_labels
    print '\n'
    results = np.array([[f(p) for f in membership_functions] for p in x])
    print 'results\n', results
    print '\n'
    maxs = np.array([np.max(r) for r in results])
    print 'maxs\n', maxs
    print'\n'
    results = [pd.Series(r) for r in results]
    print 'results\n', results
    print'\n'
    candidates = []
    for i in range(len(results)):
        candidates.append(results[i][results[i]==maxs[i]].keys().values)
        
    print 'candidates\n', candidates
    print'\n'

    if resolve_conflict == 'random':
        guessed_labels = function_labels[[np.random.choice(c) for c in candidates]]

    elif resolve_conflict == 'best':
        guessed_labels = best_choice(candidates, function_labels, y)

    elif resolve_conflict == 'worst':
        guessed_labels = worst_choice(c, function_labels, y)

    else:
        raise ValueError('the value of resolve_conflict must be random, best or worst')

    correct_labels = np.array(y)

    acc = float(sum(guessed_labels == correct_labels)) / len(x)

    if loss is None:
        return acc, None
    else:
        lss = loss(x,correct_labels,guessed_labels)
        return acc, lss


def validate_k(x, y, membership_functions, function_labels, k, resolve_conflict, loss):
    
    
    '''
    validate the passed model
    
    x: set of points
    y: set of point labels
    membership_functions: set of membership functions
    function_labels: set of function labels
    k: number of most graduated labels to consider
    resolve_conflict: string in ("best", "worst", "random"), select how resolve conflict
    loss: loss function to be used loss(x, correct_labels, guessed_labels)
    '''

    if k > len(membership_functions):
        k = len(membership_functions)
        
    function_labels = np.array(function_labels)
    results = np.array([[f(p) for f in membership_functions] for p in x])
    labelled_results = [zip(x,function_labels) for x in results]
    sorted_results = np.array([sorted(x,reverse=True) for x in labelled_results])
    topk_results = np.array([x[:k] for x in sorted_results])
    topk_labels = np.array([[x[1] for x in r] for r in topk_results])
    guessed_labels = zip(y,topk_labels)
    bool_accuracy = np.array([x[0] in x[1] for x in guessed_labels])
    acc = (float(np.sum(bool_accuracy))/len(bool_accuracy))
    correct_labels=np.array(y)
    lss = loss(x,correct_labels,topk_labels)
    return acc, lss

    
def validate_fs(x, y, membership_functions, function_labels, resolve_conflict, top_k, loss):
    
    '''
    validate the passed model
    
    x: set of points
    y: set of point labels
    membership_functions: set of membership functions
    function_labels: set of function labels
    top_k: number of most graduated labels to consider, if None only the most graduate label is considered
    resolve_conflict: string in ("best", "worst", "random"), select how resolve conflict
    loss: loss function to be used loss(x, correct_labels, guessed_labels)
    '''
    
    if top_k is None:
        return validate(x, y, membership_functions, function_labels, resolve_conflict, loss)
    else:
        return validate_k(x, y, membership_functions, function_labels, top_k, resolve_conflict, loss)

        

def iterate_tests(x, y, cs, sigmas, iterations=10, dim=2, seed=None, min_size=0.01, fuzzifier=LinearFuzzifier(), muzzifier=BinaryMuzzifier(), top_k=None, force_num_fs=False, force_labels_repr=False, same_c=True, resolve_conflict='random', loss=None, all_links=False, name='ITERATE TESTS', save_graph=False, pretty=False):
    
    '''
    Iterate the procedure of model selection and validation
    
    - x: set of data point
    - y: labels of the data points
    - cs: array of tradeoff values
    - sigmas: array of GaussianKernel parameters
    - iterations: numbers of iteration
    - dim: number of principal components to consider
    - seed: seed for initialize the numpy random generator, if no seed is specified a random seed is chosen
    - min_size: lower bound for the dimension of a cluster (in percentage of the size of the data set)
    - fuzzifier: function to be used in order to get membership values of points falling outside the crisp set
    - muzzifier: function to be used in order to set a value of mu, representing distance of a point to a cluster.
    - top_k: top_k: number of most graduated labels to consider in validation. If None only the most graduate label is considered
    - force_num_fs: if true force the number of fuzzy sets to be equal to the number of different labels
    - force_labels_repr: if true force each label to have at least one fuzzy set that represent it
    - same_c: if True in each model 1st level clustering and 2nd level clustering use the same 
      tradeoff parameter, if False alla combination of values in cs builds a model
    - resolve_conflict: resolve_conflict: string in ("best", "worst", "random"), select how resolve conflict in validation
    - loss: loss function to be used
    - all_links: if True in the graph method to build cluster all links between points are controlled. If False only links
      between support vectors and other points are controlled
    - name: string, name of the experiment
    - save_graph: if True and if dim equal to 2 graphs of clusters are saved.
    '''
    
    #get the wished principal components
    values = ut.get_principal_components(x,dim)
    
    #get the couples of c,sigma to test
    if same_c:
        couples = [(c, c, s) for s in sigmas for c in cs]
    else:
        couples = [(c, c1, s) for s in sigmas for c in cs for c1 in cs]
        
    #initialize the loss function if is None
    if loss is None:
        if top_k is None:
            loss=binary_loss
        else:
            loss=k_binary_loss
    
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
    colors, colors_map = ut.associate_labels_to_colors(y)
    
    #iterations
    while(valid_iteration < iterations):
        print '\n\nholdout iteration {}'.format(valid_iteration)
        
        #get a random permutation of data and divide it in train, validation and test set
        train_index, validation_index, test_index = ut.get_random_sets_by_index(len(values),(0.8,0.1,0.1)) #perc as a parameter?
        train_set = values[[i for i in train_index]]
        validation_set = values[[i for i in validation_index]]
        test_set = values[[i for i in test_index]]
        
        # find the couple that have the best accuracy for this permutation
        best_couples = [] #initialization of best couples
        best_accuracy = -1 #initialization of best accuracy
        
        for (c, c1, sigma) in couples:
            print '\nmodel selection: trying parameters c={}, c1={}, sigma={}'.format(c, c1, sigma)
            print 'starting clusterization'
            
            #clustering and compute a membership function for each cluster
            try:
                mf, clusters_index = learn_fs(train_set, c, c1, GaussianKernel(sigma), min_size, fuzzifier, muzzifier, all_links)
                
                #associate membership functions to labels
                membership_functions, function_labels = associate_fs_to_labels(train_set, y[[i for i in train_index]], mf, force_num_fs=force_num_fs, force_labels_repr=force_labels_repr)
                
                print 'associated labels', function_labels
                
                #calculate the accuracy on validation set
                accuracy, lss = validate_fs(validation_set, y[[i for i in validation_index]], 
                                            membership_functions, function_labels, resolve_conflict, top_k, loss)
                
                print 'accuracy: ', accuracy
                print 'loss: ', lss
                
                all_accuracies[(valid_iteration, c, c1, sigma)] = accuracy
                all_losses[(valid_iteration, c, c1, sigma)] = lss
                
                #check the best accuracy
                if accuracy == best_accuracy:
                    best_couples.append((c, c1, sigma))
                elif accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_couples = [(c, c1, sigma)]
                   
            except ValueError as e:
                print str(e)
                continue
        
        if len(best_couples) > 0:
            
            #random choice between the couples with best accuracy
            c_best, c1_best, sigma_best = best_couples[np.random.randint(len(best_couples))]
            print '\nbest couple', (c_best, c1_best, sigma_best)
            
            #with the couple with the best accuracy infer the membership function merging train and validation set
            new_train_set = np.vstack((train_set, validation_set))
            new_train_index = np.hstack((train_index, validation_index))
            
            try:
                mf, clusters_index = learn_fs(new_train_set, c_best, c1_best, GaussianKernel(sigma_best),
                                              min_size, fuzzifier, muzzifier, all_links)
                print 'first clusters', clusters_index
            
                #associate membership functions to labels
                membership_functions, function_labels = associate_fs_to_labels(new_train_set, y[[i for i in new_train_index]], mf, force_num_fs=force_num_fs, force_labels_repr=force_num_fs)
                
                print 'associated labels', function_labels
                
                #graphs
                if save_graph and dim==2:
                    
                    ut.gr_cluster_division(new_train_set, y[[i for i in new_train_index]],
                                        membership_functions, function_labels, valid_iteration, colors, sigma_best, c_best, c1_best)
                    ut.gr_save(name+"_"+str(valid_iteration)+"it_clusters_division")
                    
                    for (f,l,j) in zip(membership_functions, function_labels, range(len(function_labels))):
                        
                        ut.gr_membership_contour(new_train_set, y[[i for i in new_train_index]], f, l, j,
                                              colors, sigma_best, c_best, c1_best)
                        ut.gr_save(name+"_"+str(valid_iteration)+"it_cluster_"+str(j)+"_"+l+"_"+"contour")
                        
                        ut.gr_membership_graded(new_train_set, y[[i for i in new_train_index]], f, l, j,
                                             colors, colors_map, sigma_best, c_best, c1_best)
                        ut.gr_save(name+"_"+str(valid_iteration)+"it_cluster_"+str(j)+"_"+l+"_"+"heat")
                        

                #calculate the accuracy of the best couple on test set
                accuracy, lss = validate_fs(test_set, y[[i for i in test_index]], 
                        membership_functions, function_labels, resolve_conflict, top_k, loss)
                
                print 'test accuracy: ', accuracy
                
                test_accuracies[(valid_iteration, c_best, c1_best, sigma_best)] = accuracy
                test_losses[(valid_iteration, c_best, c1_best, sigma_best)] = lss
                
                #calculate the accuracy of the best couple on training set
                accuracy, lss = validate_fs(new_train_set, y[[i for i in new_train_index]], 
                        membership_functions, function_labels, resolve_conflict, top_k, loss)
                
                print 'training accuracy: ', accuracy
                
                training_accuracies[(valid_iteration, c_best, c1_best, sigma_best)] = accuracy
                training_losses[(valid_iteration, c_best, c1_best, sigma_best)] = lss
                
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
    results['same_c'] = same_c
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
    results['all_links'] = all_links
    results['top_k'] = top_k
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
    results['test-losses'] = test_losses
    results['training-losses'] = training_losses
    results['all-losses'] = all_losses
    results['test-loss-mean'] = np.array(test_losses.values()).mean()
    results['test-loss-std'] = np.array(test_losses.values()).std()
    results['training-loss-mean'] = np.array(training_losses.values()).mean()
    results['training-loss-std'] = np.array(training_losses.values()).std()
    results['all-loss-mean'] = np.array(all_losses.values()).mean()
    results['all-loss-std'] = np.array(all_losses.values()).std()
    
    if(pretty):
        return ut.pretty_results(results)
    else:
        return result