import utils as ut
import possibilistic_clustering as pc
import numpy as np
import pandas as pd
from possibilearn.kernel import GaussianKernel


def learn_fs(x, gamma, prc, kernel, kernel1, min_size):
    
    x_len=len(x)
    
    clusters_index, alfa, ur = pc.clustering(x, gamma, prc, kernel) #clusterize data using indexes
    
    clusters_index = [ cl for cl in clusters_index if len(cl) > x_len*min_size ] #exclude small clusters
    
    clusters_index.sort(key = lambda x: -len(x)) #sort clusters by len in order to get bigger cluster in the associate_fs_to_labels function with force parameters set to true
    
    clusters = [x[cl] for cl in clusters_index] #clusterize real data (data of x)
    print 'clusters by index ', clusters_index
    print 'number of clusters ', len(clusters)
    
    print 'inferring membership functions'
    estimated_memberships = []

    for i in range(len(clusters)):
        estimated_membership = pc.get_membership_function(clusters[i], gamma, prc, kernel1)
        estimated_memberships.append(estimated_membership)
        
    if None in estimated_memberships:
        raise ValueError('Unable to infer membership functions')
        
    return estimated_memberships, clusters_index

    
def associate_fs_to_labels(x, y, membership_functions, force_num_fs, force_labels_repr):
    
    labels=set(y)
    n_labels=len(labels)
    
    if (len(membership_functions) < n_labels) and force_num_fs:
            raise ValueError('Number of functions is less than the number of labels, cant force the number of functions')
            
    fuzzy_memberships = [[mf(p) for mf in membership_functions] for p in x]
    
    best_memberships = [np.argmax(m) for m in fuzzy_memberships]
    
    labels_founded = np.array([ [ y[i] for i in range(len(y)) if best_memberships[i] == k ] for k in range(len(membership_functions)) ])

    #exclude subclusters / associate same clusters
    membership_functions = [x[0] for x in zip(membership_functions, labels_founded) if len(x[1])>0]
    
    labels_founded = [x for x in labels_founded if len(x)>0]
    
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
    
def binary_loss(x,correct_labels,guessed_labels):
    return float(sum(guessed_labels != correct_labels)) / len(x)

def k_binary_loss(x,correct_labels,topk_labels):
    guessed_labels = zip (correct_labels,topk_labels)
    bool_accuracy = np.array([x[0] in x[1] for x in guessed_labels])
    lss = 1- (float(np.sum(bool_accuracy))/len(bool_accuracy))
    return lss
    
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
    results = np.array([[f(p) for f in membership_functions] for p in x])
    maxs = np.array([np.max(r) for r in results])
    results = [pd.Series(r) for r in results]
    candidates = []
    for i in range(len(results)):
        candidates.append(results[i][results[i]==maxs[i]].keys().values)
        
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
    
    if top_k is None:
        return validate(x, y, membership_functions, function_labels, resolve_conflict, loss)
    else:
        return validate_k(x, y, membership_functions, function_labels, top_k, resolve_conflict, loss)


def iterate_tests(x, y, gammas, prcs, sigmas, iterations=10, dim=2, seed=None, min_size=0.01, top_k=None, force_num_fs=False, force_labels_repr=False, same_sigma=False, resolve_conflict='random', loss=None, name='ITERATE TESTS', save_graph=False):
    
    
    #get the wished principal components
    values = ut.get_principal_components(x,dim)
    
    #get the triples of gamma,prc,sigma to test
    if same_sigma:
        triples = [(g, p, s, s) for g in gammas for p in prcs for s in sigmas ]
    else:
        triples = [(g, p, s, s1) for g in gammas for p in prcs for s in sigmas for s1 in sigmas]
    
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
        train_index, validation_index, test_index = ut.get_random_sets_by_index(len(values),(0.8,0.1,0.1))
        
        train_set = values[[i for i in train_index]]
        validation_set = values[[i for i in validation_index]]
        test_set = values[[i for i in test_index]]
        
        best_triples = [] #initialization of best couples
        best_accuracy = -1 #initialization of best accuracy
        
        for (gamma, prc, sigma, sigma1) in triples:
            print '\nmodel selection: trying parameters gamma={}, prc={}, sigma={}, sigma1={}'.format(gamma, prc, sigma, sigma1)
            print 'starting clusterization'
            
            #clustering and compute a membership function for each cluster
            
            mf, clusters_index = learn_fs(train_set, gamma, prc, GaussianKernel(sigma), GaussianKernel(sigma1), min_size)

            membership_functions, function_labels = associate_fs_to_labels(train_set, y[[i for i in train_index]],
                                                                           mf, force_num_fs=force_num_fs,
                                                                           force_labels_repr=force_labels_repr)
            
            print 'associated labels', function_labels
            
            #calculate the accuracy on validation set
            accuracy, lss = validate_fs(validation_set, y[[i for i in validation_index]], 
                                            membership_functions, function_labels, resolve_conflict, top_k, loss)
            
            print 'accuracy: ', accuracy
            print 'loss: ', lss

            all_accuracies[(valid_iteration, gamma, prc, sigma, sigma1)] = accuracy
            all_losses[(valid_iteration, gamma, prc, sigma, sigma1)] = lss

            #check the best accuracy
            if accuracy == best_accuracy:
                best_triples.append((gamma, prc, sigma, sigma1))
            elif accuracy > best_accuracy:
                best_accuracy = accuracy
                best_couples = [(gamma, prc, sigma, sigma1)]
                
        if len(best_couples) > 0:

            #random choice between the couples with best accuracy
            gamma_best, prc_best, sigma_best, sigma1_best = best_couples[np.random.randint(len(best_couples))]
            print '\nbest triple', (gamma_best, prc_best, sigma_best, sigma1_best)

            #with the couple with the best accuracy infer the membership function merging train and validation set
            new_train_set = np.vstack((train_set, validation_set))
            new_train_index = np.hstack((train_index, validation_index))

            mf, clusters_index = learn_fs(new_train_set, gamma_best, prc_best, GaussianKernel(sigma_best),
                                          GaussianKernel(sigma1_best), min_size)
            print 'first clusters', clusters_index
            
            #associate membership functions to labels
            membership_functions, function_labels = associate_fs_to_labels(new_train_set, y[[i for i in new_train_index]],
                                                                           mf, force_num_fs=force_num_fs,
                                                                           force_labels_repr=force_num_fs)

            print 'associated labels', function_labels

            #graphs
            if save_graph and dim==2:

                ut.gr_cluster_division(new_train_set, y[[i for i in new_train_index]],
                                    membership_functions, function_labels, valid_iteration, colors, sigma_best,
                                       prc_best, gamma_best)
                ut.gr_save(name+"_"+str(valid_iteration)+"it_clusters_division")
                    
                for (f,l,j) in zip(membership_functions, function_labels, range(len(function_labels))):

                    ut.gr_membership_contour(new_train_set, y[[i for i in new_train_index]], f, l, j,
                                          colors, sigma_best, prc_best, gamma_best)
                    ut.gr_save(name+"_"+str(valid_iteration)+"it_cluster_"+str(j)+"_"+l+"_"+"contour")

                    ut.gr_membership_graded(new_train_set, y[[i for i in new_train_index]], f, l, j,
                                         colors, colors_map, sigma_best, prc_best, gamma_best)
                    ut.gr_save(name+"_"+str(valid_iteration)+"it_cluster_"+str(j)+"_"+l+"_"+"heat")
                        

            #calculate the accuracy of the best couple on test set
            accuracy, lss = validate_fs(test_set, y[[i for i in test_index]], 
                    membership_functions, function_labels, resolve_conflict, top_k, loss)

            print 'test accuracy: ', accuracy

            test_accuracies[(valid_iteration, gamma_best, prc_best, sigma_best, sigma1_best)] = accuracy
            test_losses[(valid_iteration, gamma_best, prc_best, sigma_best, sigma1_best)] = lss

            #calculate the accuracy of the best couple on training set
            accuracy, lss = validate_fs(new_train_set, y[[i for i in new_train_index]], 
                    membership_functions, function_labels, resolve_conflict, top_k, loss)

            print 'training accuracy: ', accuracy

            training_accuracies[(valid_iteration, gamma_best, prc_best, sigma_best, sigma1_best)] = accuracy
            training_losses[(valid_iteration, gamma_best, prc_best, sigma_best, sigma1_best)] = lss

            #at this point the iteration is valid
            valid_iteration = valid_iteration + 1
            print 'iteration is valid'
        
        else:
            print 'no best couple found iteration invalid '
            continue
            
    #build the results
    results = {}
    results['name'] = name
    results['gammas'] = gammas
    results['prc'] = prcs
    results['same_c'] = same_c
    results['sigmas'] = sigmas
    results['iterations']= iterations
    results['dimensions'] = dim 
    results['min_size'] = min_size
    results['dataset-length'] = len(x)
    #results['muzzifier'] = muzzifier.name
    #results['fuzzifier'] = fuzzifier.name
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