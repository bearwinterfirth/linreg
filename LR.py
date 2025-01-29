import numpy as np
import scipy.stats as stats

class LinearRegression:
    def __init__(self, X, y):
        self.y = y
        self.X = X
        self.n = y.shape[0]
        self.d = self.X.shape[1] - 1
      
    def fit(self):
        # beräkna β0, β1, ... βi för en linjär regressionsmodell
        b = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y
        return b
    
    def dimension(self, b):
        # beräkna dimensionen av vår modell
        d = len(b) - 1
        return d
    
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
    
    def Pearsonr(self, X, a, b):
        # beräkna Pearson-r
        r = stats.pearsonr(X[:, a], X[:, b])
        return r
    
    def var_covar(self, X, var):
        # beräkna varians/kovarians-matris
        c = np.linalg.pinv(X.T @ X)*var
        return c
    
    def significance(self, i, b, c, S):
        # beräkna signifikans för enskild β-parameter
        sig=b[i]/(S*np.sqrt(c[i,i]))
        return sig
    
    def relevance(self, sig, i, n, d):
        # beräkna relevans för enskild β-parameter
        rel=2*min(stats.t.cdf(sig[i], n-d-1), stats.t.sf(sig[i], n-d-1))
        return rel
    
    def confidence_interval(self, n, d, var, c, i):
        # beräkna konfidensintervallet för enskild β-parameter
        ci=stats.t.ppf(1-0.05/2,n-d-1)*var*np.sqrt(c[i,i])
        return ci