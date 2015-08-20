import sys
import numpy as np
import datasets
import theano
import operator
import math
import evals
from sklearn.decomposition import NMF
M = 6040
N = 3952
if __name__=='__main__':
    (trainI, testI) = datasets.read_instances(sys.argv[1], sys.argv[2])
    X = np.zeros((M, N))
    testX = np.zeros((M, N))
    for ins in trainI:
        X[ins[0], ins[1]] = ins[2]
    for ins in testI:
        testX[ins[0], ins[1]] = ins[2]
    K = 20 #dimension length of latent features
    model = NMF(n_components = 20 ,init = 'random', sparseness = 'data',random_state = 0)
    U = model.fit_transform(X)
    V = model.components_
    """
    score = -1
    for k in range(10):
        print k
        matrix_factorization(trainI, U, V , K)
        P = np.dot(U, V) 
        (score_in, score_out) = eval_AUC(P, X, testX)
        if score_in > score:
            score = score_in
            np.save('/tmp/bpr_U.npy', U)
            np.save('/tmp/bpr_V.npy', V)
    """
    #U = np.load('/tmp/bpr_U.npy')
    #V = np.load('/tmp/bpr_V.npy')
    P = np.dot(U, V) 
    #print evals.AUC(P, X, testX)
    #print evals.precision_recall_atK(5 , P, X, testX)
    #print evals.precision_recall_atK(10 , P, X, testX)
    #print evals.precision_recall_atK(20 , P, X, testX)
    print evals.MAP_MRR_atK(5 , P, X, testX)
    print evals.MAP_MRR_atK(10 , P, X, testX)
    print evals.MAP_MRR_atK(20 , P, X, testX)

 
