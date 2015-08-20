import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import auc, roc_curve, roc_auc_score

def AUC(P, X ,testX = None):
    score_in = []
    score_out = []
    for i in range(X.shape[0]):
        Y = X[i]
        predY = P[i]
        try:
            score_in.append(roc_auc_score(Y, predY))
        except:
            pass
        
        Y = testX[i]
        if testX is not None:
            try:
                score_out.append(roc_auc_score(Y, predY))
            except:
                pass
        else:
            score_in = [0]

    return np.mean(score_in), np.mean(score_out)

def MAP_MRR_atK(k, P, X, testX = None):
    MAP = []
    MRR = []
    for i in range(X.shape[0]):
        nnz = [j for j in range(testX.shape[1]) if testX[i, j] != 0]
        if len(nnz) > 0:
            top = sorted(range(len(P[i])), key = lambda j: P[i, j], reverse = True)
            topk = []
            for t in top:
                if X[i, t] == 0:
                    topk.append(t)
                if len(topk) >= k:
                    break
            ap = 0.0
            rr = 0.0
            hit = 0.0
            
            #ap
            for (cnt, t) in enumerate(topk):
                if testX[i, t] == 1:
                    hit += 1
                    ap += (hit/(cnt+1))/len(nnz)
            #rr
            for (cnt, t) in enumerate(topk):
                if testX[i, t] == 1:
                    rr = 1.0/(cnt+1)
                    break
            MAP.append(ap)
            MRR.append(rr)

    return np.mean(MAP), np.mean(MRR)

def precision_recall_atK(k, P, X, testX = None):
    precision = []
    recall = []
    for i in range(X.shape[0]):
        nnz = [j for j in range(testX.shape[1]) if testX[i, j] != 0]
        if len(nnz) > 0:
            top = sorted(range(len(P[i])), key = lambda j: P[i, j], reverse = True)
            topk = []
            for t in top:
                if X[i, t] == 0:
                    topk.append(t)
                if len(topk) >= k:
                    break
            hit = set(topk) & set(nnz)
            p = float(len(hit))/k
            r = float(len(hit))/ len(nnz)
            precision.append(p)
            recall.append(r)
    return np.mean(precision), np.mean(recall)
