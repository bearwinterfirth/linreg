import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        # beräkna β0, β1, ... βi för en linjär regressionsmodell
        b = np.linalg.pinv(X.T @ X) @ X.T @ y
        return b
    
    def dimension(self, d):
        # beräkna dimensionen av vår modell
        dim = len(d) - 1
        return dim
    
    def sample_size(self, y):
        # beräkna storleken på stickprovet
        n = y.shape[0]
        return n
    
    def variance(self, y, X, b, n, d):
        # beräkna variansen av y-värden
        SSE = np.sum(np.square(y - (X @ b)))
        var = SSE/(n-d-1)
        return var
    
    def deviation(self, var):
        # beräkna standardavvikelsen
        stdev = np.sqrt(var)
        return stdev
