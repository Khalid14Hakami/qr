
import random as ran
import numpy as np
from scipy.stats import ortho_group
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.sparse as sparse


def create_upper_matrix(size):
    upper = np.zeros((size, size))
    upper_size = int((size*(size+1))/2)
    arr =  np.random.rand(upper_size)
    upper[np.triu_indices(size, 0)] = arr
    return(upper)

def sprandsym(n, density):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X) 
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result

def sparce(m, n, density):
    rvs = stats.norm().rvs
    X = sparse.random(m, n, density=density, data_rvs=rvs)
    return X


def generate_Q_R(length):
    for size in range(2, length+1, 20):
    
        r = create_upper_matrix(size)
        q = ortho_group.rvs(size)
        A = np.dot(q,r)

        result = (q, r, A)

        np.save(f'.data/q_r_data/{str(size)}.npy', result, allow_pickle=True)

def generate_saprce(m, n):
    print("generating sparce data")
    for size in range(2, n+1, 20):
        print(size)
        A = sparce(m, n, 0.05).toarray()
        np.save(f'.data/sparce_data/{str(size)}.npy', A, allow_pickle=True)

def generate_saprce_m(m):
    print("generating sparce data")
    for size in range(2, m+1, 20):
        print(size)
        A = sparce(m, size, 0.05).toarray()
        np.save(f'.data/sparce_data/fixed_m/{str(size)}.npy', A, allow_pickle=True)

def generate_saprce_n(n):
    print("generating sparce data")
    for size in range(2, n+1000, 20):
        print(size)
        A = sparce(size, n, 0.05).toarray()
        np.save(f'.data/sparce_data/fixed_n/{str(size)}.npy', A, allow_pickle=True)

def generate_dense(m, n):
    print("generating dense data")
    for size in range(2, n+1, 20):
        print(size)
        A = sparce(m, n, 1).toarray()
        np.save(f'.data/dense_data/{str(size)}.npy', A, allow_pickle=True)

def generate_dense_m(m):
    print("generating dense data")
    for size in range(2, m+1, 20):
        print(size)
        A = sparce(m, size, 1).toarray()
        np.save(f'.data/dense_data/fixed_m/{str(size)}.npy', A, allow_pickle=True)

def generate_dense_n(n):
    print("generating dense data")
    for size in range(2, n+1000, 20):
        print(size)
        A = sparce(size, n, 1).toarray()
        np.save(f'.data/dense_data/fixed_n/{str(size)}.npy', A, allow_pickle=True)

generate_Q_R(1222)

generate_saprce(1222, 1222)
generate_dense(1222, 1222)

generate_saprce_m(1222)
generate_dense_m(1222)

generate_saprce_n(2)
generate_dense_n(2)


