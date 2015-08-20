import sys
import numpy as np
import datasets
import theano
import operator
import math
import evals
M = 6040
N = 3952

def der_exp(x):
    return 1.0/(1+np.exp(x))

def matrix_factorization(Train, P, Q, K,alpha=0.002,beta=0.02):
    for i in xrange(len(Train)):
        record = Train[i]
        userid = record[0]
        itemid1 = record[1]
        itemid2 = np.random.randint(0, N-1)
        Xuij = np.dot(P[userid,:],Q[:,itemid1]) - np.dot(P[userid,:],Q[:,itemid2]) #error between real rate and the predicted rate
        #print userid,itemid,rate
        #print P[userid,:]
        #print Q[:,itemid]
        P[userid] -= alpha*(der_exp(Xuij)* (Q[:,itemid1]-Q[:,itemid2])+beta*P[userid])
        Q[:,itemid1] -= alpha*(der_exp(Xuij)* ( -P[userid]+beta*Q[:,itemid1]) )
        Q[:,itemid2] -= alpha*(der_exp(Xuij)* ( P[userid]+beta*Q[:,itemid2]) )

if __name__=='__main__':
    (trainI, testI) = datasets.read_instances(sys.argv[1], sys.argv[2])
    X = np.zeros((M, N))
    testX = np.zeros((M, N))
    for ins in trainI:
        X[ins[0], ins[1]] = ins[2]
    for ins in testI:
        testX[ins[0], ins[1]] = ins[2]
    K = 20 #dimension length of latent features
    """
    U = np.random.rand(M,K)
    V = np.random.rand(K,N)
    score = -1
    for k in range(20):
        print k
        matrix_factorization(trainI, U, V , K)
        P = np.dot(U, V) 
        (score_in, score_out) = eval_AUC(P, X, testX)
        if score_in > score:
            score = score_in
            np.save('/tmp/bpr_U.npy', U)
            np.save('/tmp/bpr_V.npy', V)
    """
    U = np.load('/tmp/bpr_U.npy')
    V = np.load('/tmp/bpr_V.npy')
    P = np.dot(U, V) 
    #print evals.AUC(P, X, testX)
    #print evals.precision_recall_atK(5 , P, X, testX)
    #print evals.precision_recall_atK(10 , P, X, testX)
    #print evals.precision_recall_atK(20 , P, X, testX)

 
    print evals.MAP_MRR_atK(5 , P, X, testX)
    print evals.MAP_MRR_atK(10 , P, X, testX)
    print evals.MAP_MRR_atK(20 , P, X, testX)
