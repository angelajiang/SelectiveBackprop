import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn import gaussian_process

class Predictor():

    def update(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

class GPPredictor(Predictor):

    def __init__(self):
        #super(GPPredictor, self).__init__()
        self.kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
        self.gp = gaussian_process.GaussianProcessRegressor(kernel=self.kernel)

    def update(self, X, ys):
        self.gp.fit(X, ys)

    def predict(self, x):
        X = np.array([x]).reshape(-1, 1)
        y, std = self.gp.predict(X, return_std=True)
        return y, std

class RTOPredictor(Predictor):

    def __init__(self):
        self.kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
        self.predictor = gaussian_process.GaussianProcessRegressor(kernel=self.kernel)
        self.mean = 1
        self.std = 0
        self.first = True
        self.alpha = 1./8.
        self.beta = 1./4.

    def update(self, Xs, y):
        # RTTVAR <- (1 - beta) * RTTVAR + beta * |SRTT - R'|
        # SRTT <- (1 - alpha) * SRTT + alpha * R'
        if self.first:
            self.mean = y
            self.std = 0
            self.first = False
        else:
            self.std = (1 - self.beta) * self.std + self.beta * abs(self.mean - y)
            self.mean = (1 - self.alpha) * self.mean + self.alpha * y

    def predict(self, x):
        return self.mean, self.std

