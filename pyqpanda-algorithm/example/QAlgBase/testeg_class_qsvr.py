from pyqpanda_alg.QFinance import class_qsvr
import sympy as sp
import numpy as np
import pyqpanda as pq

if __name__ == '__main__':
    
    n_samples = 100
    n_features = 2
    X = np.random.rand(n_samples, n_features) * 10
    y = (2 * np.sin(X[:, 0]) +
         1.5 * np.cos(X[:, 1]))
    class_qsvr.Quantum_SVR(X, y).show_res()