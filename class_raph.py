class Complex:
    def __init__(self, ls):
        assert len(ls) == 2
        self.real, self.imag = ls

    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __repr__(self, other):
        return f"{repr(self.real)} + {repr(self.imag)}i"

    def fit(X, y):
        b = np.linalg.pinv(...)
        return b
    