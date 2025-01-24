import numpy as np
import scipy.stats as stats

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
    
    def SSE(self, y, X, b):
        # beräkna SSE, sum of square errors
        SSE = np.sum(np.square(y - (X @ b)))
        return SSE
    
    def variance(self, SSE, n, d):
        # beräkna variansen
        var = SSE/(n-d-1)
        return var
    
    def deviation(self, var):
        # beräkna standardavvikelsen
        stdev = np.sqrt(var)
        return stdev
    
    def Syy(self, n, y):
        # beräkna variansen i y
        Syy = (n*np.sum(np.square(y))- np.square(np.sum(y)))/n
        return Syy
    
    def SSR(self, Syy, SSE):
        # beräkna sum of square residuals
        SSR = Syy - SSE
        return SSR
    
    def Fstatistic(self, SSR, d, var):
        # beräkna F-statistika
        F = (SSR/d)/var
        return F

    def Rsq(self, SSR, Syy):
        # beräkna R_squared
        Rsq = SSR/Syy
        return Rsq
    
    def Pearsonr(self, X, a, b)
        # beräkna Pearson-r
        r = stats.pearsonr(X[:, a], X[:, b])
        return r