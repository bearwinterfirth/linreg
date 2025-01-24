import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        b = np.linalg.pinv(X.T @ X) @ X.T @ y
        return b
    
    def dimension(self, d):
        dim = len(d) - 1
        return dim
    
    def sample_size(self, y):
        n = y.shape[0]
        return n
    
    def variance(self, y, X, b, n, d):
        SSE = np.sum(np.square(y - (X @ b)))
        var = SSE/(n-d-1)
        return var
