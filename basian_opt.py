import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import random


class GPoptimizer():
    def _optim(self,func, x0, fprime=None, args=(), approx_grad=0,
                  bounds=None, m=10, factr=1e7, pgtol=1e-5,
                  epsilon=1e-8, maxfun=15000, maxiter=15000, disp=None,
                  callback=None, maxls=20):
        theta_opt, func_min, _ = \
            fmin_l_bfgs_b(func, x0, fprime=fprime, args=args,
                      approx_grad=approx_grad,
                      bounds=bounds, m=m, factr=factr, pgtol=pgtol,
                      epsilon=epsilon,
                      iprint=1, maxfun=maxfun, maxiter=maxiter, disp=disp,
                      callback=callback, maxls=maxls)
        return theta_opt, func_min

    def __init__(self,parameter_conf,f):
        self.scores = []
        self.parameters = []
        self.blackbox = f
        self.parameter_dim = len(parameter_conf)
        self.configs = parameter_conf
        self.kernel  = ConstantKernel() * Matern(length_scale=np.ones(shape=(self.parameter_dim,))) + WhiteKernel()
        self.pipe = Pipeline([('scaler', StandardScaler()),
                 ('gpr', GaussianProcessRegressor(kernel=self.kernel,
                                                  normalize_y=False,
                                                  optimizer=self._optim,
                                                  n_restarts_optimizer=0,
                                                  random_state=1))])
        self.max = None

    def random_sample(self,num):
        samples = []
        for _ in range(num):
            sample = []
            for parameter_conf in self.configs:
                if parameter_conf[0] == 'int':
                    value = random.randrange(parameter_conf[1],parameter_conf[2]+1)
                if parameter_conf[0] == 'float':
                    value = random.random() * (parameter_conf[2] - parameter_conf[1]) + parameter_conf[1]
                sample.append(value)
            samples.append(sample)
        return samples
        
    def update_model(self,parameters,scores):
        self.parameters = self.parameters + parameters
        self.scores = self.scores + scores
        self.max = max(self.scores)
        self.pipe.fit(self.parameters, self.scores)

    def compute_PI(self,candidates): #probablilty improvement
        import functools
        self.pipe.named_steps['gpr'].predict = functools.partial(self.pipe.named_steps['gpr'].predict, return_std=True)
        y_predict, y_std = self.pipe.predict(candidates)
        probability_improvement = norm.cdf((y_predict - self.max)  / y_std)
        return probability_improvement