from numpy.linalg import inv
import numpy as np


class CustomLinearReg:
    def __init__(self):
        self.a=0
        self.b=0
        
        
    def fit(self, X, y):
        R=inv(X.T@X)@X.T@y
        self.a=R[0]
        self.b=R[1]
    
    def predict(self, X):
        return X@np.array([self.a,self.b])


