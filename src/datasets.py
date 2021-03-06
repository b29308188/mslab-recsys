import sys
import numpy as np

def train_test_split_order(input_file, output_train, output_test):
    A = []
    for line in open(input_file, "r"):
        tokens = line.strip().split("::")
        A.append((tokens[0], tokens[1], tokens[2], int(tokens[3])))
    A.sort(key = lambda x: x[3])
    train_A = A[:int(len(A)*0.8)]
    test_A = A[int(len(A)*0.8):]
    with open(output_train, "w") as f:
        for a in train_A:
            f.write("%s %s %s %d\n"%(a[0], a[1], a[2], a[3]))
    with open(output_test, "w") as f:
        for a in test_A:
            f.write("%s %s %s %d\n"%(a[0], a[1], a[2], a[3]))

def read_instances(train_file, test_file):
    trainI = []
    testI = []
    for line in open(train_file):
        tokens = line.strip().split()
        #if float(tokens[2]) >= 3:
        trainI.append((int(tokens[0]) -1 , int(tokens[1]) -1, 1))
    for line in open(test_file):
        tokens = line.strip().split()
        #if float(tokens[2]) >= 3:
        testI.append((int(tokens[0]) - 1, int(tokens[1]) - 1, 1))
    return trainI, testI
