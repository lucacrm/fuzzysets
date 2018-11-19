from support_vector_clustering import *
from possibilearn import *
from possibilearn.kernel import GaussianKernel
import skfuzzy as fuzz
from muzzifiers import *
import matplotlib.pyplot as plt


def learn_fs(x, c, c1, kernel, min_size, fuzzifier, muzzifier):

    
def associate_fs_to_labels(x, y, membership_functions, force_num_fs, force_labels_repr):
    
    
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

    

def validate_k(x, y, membership_functions, function_labels, k, resolve_conflict, loss):

    
def validate_fs(x, y, membership_functions, function_labels, resolve_conflict, top_k, loss):

        

def iterate_tests(x, y, cs, sigmas, iterations, dim=2, seed=None, min_size=0.01, fuzzifier=LinearFuzzifier(), muzzifier=BinaryMuzzifier(), top_k=None, force_num_fs=False, force_labels_repr=False, same_c=True, resolve_conflict='random', loss=None, name='ITERATE TESTS', save_graph=False):
    