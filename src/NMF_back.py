import sys
import numpy as np
import datasets
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as MSE
M = 6040
N = 3952
if __name__ == "__main__":
    (trainI, testI) = datasets.read_instances(sys.argv[1], sys.argv[2])
    X = np.zeros((M, N))
    for ins in trainI:
        X[ins[0], ins[1]] = ins[2]
    #X = np.array([[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4],[0, 1, 5, 4]] )
    model = NMF(n_components = 100 ,init = 'random', sparseness = 'data',random_state = 0)
    U = model.fit_transform(X)
    V = model.components_
    print U.shape, V.shape
    Y = np.array([ins[2] for ins in trainI])
    predY = np.array([np.dot(U[ins[0],:], V[:,ins[1]]) for ins in trainI])
    print np.sqrt(MSE(Y, predY))
    #print np.dot(U, V)
    print model.reconstruction_err_
    print model.n_iter_

