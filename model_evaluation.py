
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


class ManualGroupKFold():
    """ K-fold iterator variant with non-overlapping groups.
        The same group will not appear in two different folds (the number of
        distinct groups has to be at least equal to the number of folds).
        The folds are approximately balanced in the sense that the number of
        distinct targets is approximately the same in each fold.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.

    random_state : None, int
        Pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Example 
    -------
    >>> target = np.array([1]*10+ [0]*10)
    >>> groups = np.array([i//2 for i in range(20)])
    >>> X = np.random.random((20,3))

    >>> mgf = ManualGroupKFold(n_splits = 3, random_state = 52)

    >>> print('Target {}, Groups {}'.format(target, groups))
    >>> for train, test in mgf.split(X, target, groups):
    ...    print('-----------------------------------------')
    ...    print('Train : {}, Test : {}'.format(train, test))
    ...    print('Target train : {}, Target test : {}'.format(target[train], target[test]))
    ...    print('Groups train : {}, Groups test : {}'.format(groups[train], groups[test]))

    Target [1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0], Groups [0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9]
    -----------------------------------------
    Train : [ 0  1  6  7  8  9 12 13 16 17 18 19], Test : [ 2  3  4  5 10 11 14 15]
    Target train : [1 1 1 1 1 1 0 0 0 0 0 0], Target test : [1 1 1 1 0 0 0 0]
    Groups train : [0 0 3 3 4 4 6 6 8 8 9 9], Groups test : [1 1 2 2 5 5 7 7]
    -----------------------------------------
    Train : [ 2  3  4  5  8  9 10 11 14 15 18 19], Test : [ 0  1  6  7 12 13 16 17]
    Target train : [1 1 1 1 1 1 0 0 0 0 0 0], Target test : [1 1 1 1 0 0 0 0]
    Groups train : [1 1 2 2 4 4 5 5 7 7 9 9], Groups test : [0 0 3 3 6 6 8 8]
    -----------------------------------------
    Train : [ 0  1  2  3  4  5  6  7 10 11 12 13 14 15 16 17], Test : [ 8  9 18 19]
    Target train : [1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0], Target test : [1 1 0 0]
    Groups train : [0 0 1 1 2 2 3 3 5 5 6 6 7 7 8 8], Groups test : [4 4 9 9]
    """
    def __init__(self, n_splits = 3, random_state = None):

        self.n_splits = n_splits
        self.random_state = random_state
        
    def get_n_splits(self, ):

        return self.n_splits
    
    def split(self, X, target, groups):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : numpy ndarray
            of shape (object, features) data object

        target : numpy ndarray
            of shape (object, ) target variable,
            folds are approximately balanced by this variable

        groups : numpy ndarray
            of shape (object, ) characteristic variable,
            objects from the same group will occur in the same fold

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        skf = StratifiedKFold(n_splits = self.n_splits,
                             shuffle = True, random_state = self.random_state)
        target_unique = np.array([target[groups == elem][0] for elem in np.unique(groups)])
        names_unique = np.unique(groups)
        idx = np.arange(X.shape[0])
               
        for train, test in skf.split(np.zeros(target_unique.shape[0]), target_unique):
            
            train_labels = np.array(names_unique)[train]
            test_labels = np.array(names_unique)[test]
            train_idx = np.in1d(groups, train_labels)
            test_idx = np.in1d(groups, test_labels)
            
            yield idx[train_idx], idx[test_idx]


def exp_kernel(pseudo_kernel, a):
    """This function generate
        almost legit kernel (depending on
        parameter a) from pseudo kernel.
        Under legit we understand that
        this matrix will be semi-positive definite

        

    Parameters
    ----------
    pseudo_kernel : ndarray of shape (n_samples, n_samples)
                pseudo kernel, in case of clustering approach on
                i'th, j'th position of this matrix stands a
                pairwise ARI or AMI coefficient between
                precomputed partition of sample i and sample j

    a : float
        kernel multiplicator
    
    Returns
    -------
    legit_kernel : ndarray of shape (n_samples, n_samples)
                Legit kernel generates as exp{-(1 - pseudo_kernel)}
                Since pseudo_kernel contains measure of simularity
                we do (1 - pseudo_kernel) to obtain measure of distance

    """
    legit_kernel = np.exp(-a*(np.ones(pseudo_kernel.shape) - pseudo_kernel))

    return legit_kernel

def SVC_grid_search(p_kernel, target, groups, n_splits, params, penalties, make_kernel = exp_kernel, random_state = 100):
    """Grid search over model parameters to estimate the best one
        THIS IS MODEL SELECTION
    Parameters
    -----------

    p_kernel : ndarray of size (n_samples, n_samples)
            pseudo kernel

    target : numpy ndarray
            of shape (object, ) target variable

    groups : numpy ndarray
            of shape (object, ) characteristic variable,
            objects from the same group will always occur in the same fold

    n_splits : int
            number of folds for cross validation

    params : list or ndarray unfixed size
            parameters of kernel

    penalties : list or ndarray unfixed size
            parameters of SVC

    make_kernel : python function, default exp_kernel
            function that produces legi kernel from pseudo kernel
            using one of parameters from params
            default  exp_kernel

    random_state : None, int
        Pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Returns
    -------

    best_params : python dictionary 
                  {'Kernel Parameter' : best_param, 'SVC Parameter' : best_penalty}

    best_auc : float
            averaged auc achieved on train folds for best_params

    train_auc_mean : ndarray of size (len(params), len(penalties))
            averaged auc achieved for all model parameters

    train_auc_std ; ndarray of size (len(params), len(penalties))
            std of auc achieved for all model parameters

    Comments 
    --------

    I. There is 2 ways to validate/choose best params:

    1. Compute auc for each train fold, average it
    2. Create vector consists of trains prediction,
        compute auc on it and averages over different random states

    Here I choose the first one

    II. I use fixed random states for both cross validation and SVC classification
    to ensure that differencies between aucs are all depend on model parameters
    """

    train_auc_mean = np.zeros((len(params), len(penalties)))
    train_auc_std = np.zeros((len(params), len(penalties)))
    
    tr_m = np.zeros((len(params), len(penalties)))

    for rs in range(random_state):
        for kidx, kernel_parameter in enumerate(params):
            for sidx, svc_parameter in enumerate(penalties):

                cv = ManualGroupKFold(n_splits = n_splits,
                                     random_state = rs)

                clf = SVC(C = svc_parameter, kernel = 'precomputed',
                             random_state = random_state) 

                kernel = make_kernel(p_kernel, kernel_parameter) 
                cv_auc = []
                for train, test in cv.split(kernel, target, groups):

                    kernel_train, kernel_test = kernel[train][:, train], kernel[test][:, train]
                    y_train, y_test = target[train], target[test]
                    clf.fit(kernel_train, y_train)
                    y_predicted = clf.decision_function(kernel_test)
                    cv_auc.append(roc_auc_score(y_test, y_predicted))

                train_auc_mean[kidx, sidx] = np.mean(cv_auc)
                train_auc_std[kidx, sidx] = np.std(cv_auc)
        tr_m += train_auc_mean       
         
    tr_m /= random_state
    i, j = np.unravel_index(tr_m.argmax(), tr_m.shape)
    best_params = {'Kernel Parameter' : params[i], 'SVC Parameter' : penalties[j]}
    best_auc = tr_m[i, j]


    return best_params, best_auc, tr_m





def SVC_score (p_kernel, target, groups, n_splits = 10, n_repetitions = 100, param = 1, penalty = 1, make_kernel = exp_kernel, random_state = 100):
    """SVC classification score for given parameters,
        averaged over multiple repetitions of cross validation 
        with different random states. For each repetition it creates
        vector consists of trains prediction, compute auc on it.

    THIS IS MODEL EVALUATION

    Parameters
    -----------

    p_kernel : ndarray of size (n_samples, n_samples)
            pseudo kernel

    target : numpy ndarray
            of shape (object, ) target variable

    groups : numpy ndarray
            of shape (object, ) characteristic variable,
            objects from the same group will always occur in the same fold

    n_splits : int, default 10
            number of folds for cross validation

    n_repetitions : int, default 100
            number of repetitions to average over
            to obtain mean and std of the auc score

    param : float, default 1
            kernel parameter, used to obtain
            legit kernel from pseudo kernel

    penalty : float, default 1
            SVC regularization parameter

    make_kernel : python function, default exp_kernel
            function that produces legit kernel from pseudo kernel
            using param, default  exp_kernel

    random_state : int, default 0
        starting Pseudo-random number generator state used for
        shuffling. Final score is computed over multiple
        repetition, each repetition uses different random state

    Returns
    -------
    aucs : ndarray, of shape (n_repetitions, )
        aucs obtained for different different 
        cross validaton splits
    """
    kernel = make_kernel(p_kernel, param)
    clf = SVC(C = penalty, kernel = 'precomputed',
                     random_state = random_state) 
    aucs = np.zeros(n_repetitions)

    for repetition in range(n_repetitions):

        cv = ManualGroupKFold(n_splits = n_splits,
                             random_state = random_state + repetition)
        y_predicted = np.zeros(target.shape[0])
        for train, test in cv.split(kernel, target, groups):

            kernel_train, kernel_test = kernel[train][:, train], kernel[test][:, train]
            y_train, y_test = target[train], target[test]
            clf.fit(kernel_train, y_train)
            y_predicted[test] = clf.decision_function(kernel_test)
        aucs[repetition] = roc_auc_score(target, y_predicted)

    return aucs