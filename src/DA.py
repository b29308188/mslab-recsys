import sys
import numpy as np
import datasets
import theano
import evals
from keras.models import Sequential
from keras.layers import containers
from keras.optimizers import SGD
from keras.layers.core import Dense, AutoEncoder

M = 6040
N = 3952
if __name__ == "__main__":
    (trainI, testI) = datasets.read_instances(sys.argv[1], sys.argv[2])
    X = np.zeros((M, N))
    testX = np.zeros((M, N))
    trainS = [set() for i in range(M)]
    for ins in trainI:
        X[ins[0], ins[1]] = ins[2]
        trainS[ins[0]].add(ins[1])
    for ins in testI:
        testX[ins[0], ins[1]] = ins[2]
    #X = np.array([[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4],[0, 1, 5, 4]] )
    encoder = containers.Sequential([Dense(3952, 300, activation = 'sigmoid'), Dense(300, 100, activation = 'sigmoid')])
    decoder = containers.Sequential([Dense(100, 300, activation = 'sigmoid'), Dense(300, 3952, activation = 'sigmoid')])
    #encoder = containers.Sequential([Dense(3952, 100)])
    #decoder = containers.Sequential([Dense(100, 3952)])
    model = Sequential()
    model.add(AutoEncoder(encoder=encoder, decoder=decoder))
    sgd = SGD(lr = 0.2)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    #model.load_weights("/tmp/DA_weights.mod.hdf5")
    """
    score = -1
    for k in range(60):
        print k
        model.fit(X, X, nb_epoch = 1, batch_size = 128)
        P = model.predict(X)
        (score_in, score_out) = eval_AUC(P, X, testX)
        print score_in, score_out
        if score_in > score:
            score = score_in
            model.save_weights("/tmp/DA_weights.mod.hdf5", overwrite = True)
    """
    model.load_weights("/tmp/DA_weights.mod.hdf5")
    P = model.predict(X)
    #print evals.AUC(P, X, testX)
    #print evals.precision_recall_atK(5 , P, X, testX)
    #print evals.precision_recall_atK(10 , P, X, testX)
    #print evals.precision_recall_atK(20 , P, X, testX)
    print evals.MAP_MRR_atK(5 , P, X, testX)
    print evals.MAP_MRR_atK(10 , P, X, testX)
    print evals.MAP_MRR_atK(20 , P, X, testX)
