import numpy as np
import pandas as pd


import os

from igraph import *
import networkx as nx
import community

import scipy as sc
from scipy.sparse import csr_matrix

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances

from sklearn import metrics
#from sklearn.cross_validation import StratifiedKFold, cross_val_score, LeaveOneOut
from time import time
from sklearn import svm

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from scipy import interp


# %install_ext https://raw.github.com/cpcloud/ipython-autotime/master/autotime.py
# %load_ext autotime

def convert_all(array_of_vectors, size=68):
    array_of_matrices = np.zeros((array_of_vectors.shape[0], size, size))

    for idx, vector in enumerate(array_of_vectors):
        array_of_matrices[idx] = convert(vector, size)
    return array_of_matrices


def convert(data, size=68, mode='vec2mat'):  # diag=0,
    '''
    Convert data from upper triangle vector to square matrix or vice versa
    depending on mode.

    INPUT : 

    data - vector or square matrix depending on mode
    size - preffered square matrix size (given by formula :
           (1+sqrt(1+8k)/2, where k = len(data), when data is vector)
    diag - how to fill diagonal for vec2mat mode
    mode - possible values 'vec2mat', 'mat2vec'

    OUTPUT : 

    square matrix or 1D vector 

    EXAMPLE :

    a = np.array([[1,2,3],[2,1,4],[3,4,1]])
    vec_a = convert(a, mode='mat2vec')
    print(vec_a)
    >>> array([2, 3, 4])

    convert(vec_a, size = 3, diag = 1, mode = vec2mat)
    >>> matrix([[1, 2, 3],
                [2, 1, 4],
                [3, 4, 1]], dtype=int64)

    '''

    if mode == 'mat2vec':

        mat = data.copy()
        rows, cols = np.triu_indices(data.shape[0], k=0)
        vec = mat[rows, cols]

        return vec

    elif mode == 'vec2mat':

        vec = data.copy()
        rows, cols = np.triu_indices(size, k=0)
        mat = csr_matrix((vec, (rows, cols)), shape=(size, size)).todense()
        mat = mat + mat.T  # symmetric matrix
        # np.fill_diagonal(mat, diag)
        np.fill_diagonal(mat, np.diag(mat) / 2)
        return mat


def NewmanEig(data, cl_n=None):
    '''
    Compute partiton for a given set of graphs
    using Newman leading eigvector method (aprx 0.19 sec per graph);
    with different prespecified number of clusters

    INPUT:

    data - n_obj x n_vertices x n_vertices numpy array (94x264x264, 54x110x110)
    cl_n - number of clusters to find (None by default)

    OUTPUT:

    part_all - n_obj x n_vertices numpy array of labels

    '''

    part_all = np.zeros((data.shape[0], data.shape[1]))

    for idx, matr in enumerate(data):
        # print(idx)
        g0 = Graph.Weighted_Adjacency(
            matr.tolist(), mode=ADJ_MAX, attr='weight')
        membership = g0.community_leading_eigenvector(
            clusters=cl_n, weights='weight').membership
        part_all[idx] = np.array(membership)

    return part_all


def FastGreedy(data):
    '''
    Compute partiton for a given set of graphs
    using igraph community_fastgreedy method
    (greedy optimization of modularity) 

    INPUT:

    data - n_obj x n_vertices x n_vertices numpy array (94x264x264, 54x110x110)

    OUTPUT:

    part_all - n_obj x n_vertices numpy array of labels

    '''

    part_all = np.zeros((data.shape[0], data.shape[1]))

    for idx, matr in enumerate(data):
        # print(str(idx),'/',str(data.shape[0]))
        g0 = Graph.Weighted_Adjacency(
            matr.tolist(), mode=ADJ_MAX, attr='weight')
        dendrogram = g0.community_fastgreedy(weights='weight')
        clusters = dendrogram.as_clustering()
        part_all[idx] = np.array(clusters.membership)

    return part_all


def CommunityLouvian(data):

    part_all = []

    for idx, mat in enumerate(data):
        # print(str(idx),'/',str(data.shape[0]))
        g = nx.Graph(mat)
        partition = community.best_partition(g)
        part_all.append(list(partition.values()))

    part_all = np.array(part_all)

    return part_all


def metric_adj(all_lbl, mode='AMI'):

    if mode == 'AMI':
        metric_ami = np.diag(np.ones(all_lbl.shape[0]))
        rows, cols = np.triu_indices(metric_ami.shape[0], 1)

        for i, j in zip(rows, cols):
            sim_ami = adjusted_mutual_info_score(all_lbl[i], all_lbl[j])
            metric_ami[i, j] = sim_ami

        mat_ami = metric_ami + metric_ami.T
        np.fill_diagonal(mat_ami, 1)

        return mat_ami
    else:
        metric_ami = np.diag(np.ones(all_lbl.shape[0]))
        rows, cols = np.triu_indices(metric_ami.shape[0], 1)
        metric_ari = metric_ami.copy()

        for i, j in zip(rows, cols):
            sim_ari = adjusted_rand_score(all_lbl[i], all_lbl[j])
            metric_ari[i, j] = sim_ari
            sim_ami = adjusted_mutual_info_score(all_lbl[i], all_lbl[j])
            metric_ami[i, j] = sim_ami

        mat_ari = metric_ari + metric_ari.T
        np.fill_diagonal(mat_ari, 1)

        mat_ami = metric_ami + metric_ami.T
        np.fill_diagonal(mat_ami, 1)

        return mat_ari, mat_ami


#def _kernel(part, a):
#
#    m = np.ones(part.shape)
#    m = a * (m - part)
#    m = np.exp(-m)
#
#    return m


#def repeatSVM_labeled(gram_matrix, full_target_vector, full_labels,
#                      target_unique, names_unique, n_folds=10,
#                      n_repetitions=50, start_state=0, penalty=1, LOO=False):
#    '''
#    This function computes ROC AUC values of SVM with precomputed kernel
#    over several runs of k-fold cross-validation. It works with the multiple
#    objects for the same labels (participants) and produces stratified Kfold
#    in terms of unique participants. It averages prediction for a participant
#    over the respective objects and computes ROC AUC on this participant-based
#    prediction
#
#    Accepts:
#    gram_matrix - precomputed Gram matrix for all
#                  classification objects
#    full_target_vector - target vector of class
#                         labels for all classification objects
#
#    full_labels - vector of labels
#    array_unique - an array of unique labels (first column)
#                   and their classes (second column)
#
#    n_folds - number of folds for cross-validation
#    n_repetitions - number of repetitions of k-fold cross-validation procedure
#                    (with random splitting into folds)
#    penalty - parameter C of the sklearn.svm.SVC 
#              (penalty parameter of the error term)
#
#    Returns:
#    roc_auc - list of length n_repetitions wherein each value
#              is a ROC AUC based on aggregated over folds prediction.
#    decisions - list of length n_repetitions wherein each element is
#                an array of the respective aggragated outputs of the decision function
#    Note: decision function output is used for prediction to avoid
#          problems with Platt calibration in SVM predict_proba
#     '''
#
#    X, y = gram_matrix, np.array(full_target_vector)
#    labels = np.array(full_labels)
#
#    clf = svm.SVC(C=penalty, kernel='precomputed', random_state=0)
#    overall_roc_auc = []
#
#    for rep in range(0, n_repetitions):
#        if LOO is not True:
#            CV = StratifiedKFold(np.array(target_unique), n_folds,
#                                 shuffle=True, random_state=start_state + rep)
#        else:
#            CV = LeaveOneOut(np.array(target_unique).shape[0])
#        decision_predicted = np.zeros(y.shape[0])
#        for train, test in CV:
#            train_labels = np.array(names_unique)[train]
#            test_labels = np.array(names_unique)[test]
#            train_idx = np.in1d(labels, train_labels)
#            test_idx = np.in1d(labels, test_labels)
#
#            clf.fit(X[train_idx][:, train_idx], y[train_idx])
#            decision_output = clf.decision_function(X[test_idx][:, train_idx])
#            decision_predicted[test_idx] = decision_output
#
#        subject_prediction = np.zeros(np.array(target_unique).shape[0])
#
#        for i in range(0, np.array(target_unique).shape[0]):
#            lab = np.array(names_unique)[i]
#            idx = np.where(labels == lab)[0]
#            value = np.mean(decision_predicted[idx])
#            subject_prediction[i] = value
#
#        roc_auc_value = metrics.roc_auc_score(
#            np.array(target_unique), subject_prediction)
#        overall_roc_auc.append(roc_auc_value)
#
#    return np.array(overall_roc_auc)
#
#
#def repeatSVM(gram_matrix, target_vector, n_folds=10,
#              n_repetitions=50, penalty=1, LOO=False):
#    '''
#    This function computes ROC AUC values of SVM with precomputed
#    kernel over several runs of k-fold cross-validation
#
#    Accepts:
#    gram_matrix - precomputed Gram matrix
#    target_vector - target vector of class labels
#    n_folds - number of folds for cross-validation
#    n_repetitions - number of repetitions of k-fold
#                    cross-validation procedure (with random splitting into folds)
#    penalty - parameter C of the sklearn.svm.SVC
#              (penalty parameter of the error term)
#    Returns:
#    roc_auc - list of length n_repetitions wherein each
#              value is a ROC AUC based on aggregated over folds prediction.
#    decisions - list of length n_repetitions wherein each element
#                is an array of the respective aggragated outputs of
#                the decision function
#    Note: decision function output is used for prediction
#          to avoid problems with Platt calibration in SVM predict_proba
#     '''
#    X, y = gram_matrix, np.array(target_vector)
#    clf = svm.SVC(C=penalty, kernel='precomputed', random_state=0)
#    overall_roc_auc = []
#    overall_decisions = []
#    # overall_predict_proba = []
#    overall_predict = np.zeros(108)
#
#    roc_c = []
#
#    for rep in range(0, n_repetitions):
#        if LOO is not True:
#            CV = StratifiedKFold(target_vector, n_folds,
#                                 shuffle=True, random_state=rep)
#        else:
#            CV = LeaveOneOut(y.shape[0])
#        decision_predicted = np.zeros(y.shape[0])
#        metrics_dict = {'ROC curve': []}
#        for train, test in CV:
#            clf.fit(X[train][:, train], y[train])
#            decision_output = clf.decision_function(X[test][:, train])
#            decision_predicted[test] = decision_output
#            # predicted_proba = clf.predicted_proba(X[test][:, train])
#            predicted = clf.predict(X[test][:, train])
#
#            metrics_dict['ROC curve'].append(
#                roc_curve(y[test], decision_output))
#            roc_c.append(metrics_dict)
#
#        roc_auc_value = roc_auc_score(y, decision_predicted)
#        overall_roc_auc.append(roc_auc_value)
#        overall_decisions.append(decision_predicted)
#        # overall_predict_proba.append(predicted_proba)
#        overall_predict[test] = predicted
#    # , np.array(overall_predict_proba)
#    return np.array(overall_roc_auc), np.array(overall_decisions), np.array(overall_predict), roc_c
