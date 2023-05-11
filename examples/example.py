"""
This is an example of SPKF from Plett's book "Battery Management System. vol 2".

x_{k+1} = f(x_k, u_k, w_k) = sqrt(5+x_k) + w_k
y_k = h(x_k, u_k, v_k) = x_3 ** 3 + v_k

"""

import numpy as np
import scipy.linalg

from kf.spkf import SPKF


def f_func(x_k, u_k, w_k):
    return np.sqrt(5+x_k) + w_k

def h_func(x_k, u_k, v_k):
    return x_k**3 + v_k

# Define the size of variables in the model
Nx = 1
Ny = 1

# Initialize simulation variables
xhat = 2
SigmaX = 1
SigmaW = 1
SigmaV = 2
max_iter = 10

# simulate true signals
xtrue = 2 + np.random.normal()

# setup spkf object
spkf_obj = SPKF(xhat=xhat, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func, h_func=h_func)

# storage arrays
xstore = np.zeros(max_iter+1)
xstore[0] = xtrue
xhatstore = np.zeros(max_iter)
SigmaXstore = np.zeros(max_iter)

for i in range(max_iter):
    w = (np.transpose(scipy.linalg.cholesky(SigmaW)) * np.random.normal())[0, 0]
    v = (scipy.linalg.cholesky(SigmaV) * np.random.normal())[0, 0]
    ytrue = xtrue ** 3 + v
    xtrue = np.sqrt(5 + xtrue) + w

    # update SPKF object variables
    spkf_obj.solve(u=0, ytrue=ytrue)
    xhatstore[i] = spkf_obj.xhat
    SigmaXstore[i] = spkf_obj.SigmaX
    xstore[i + 1] = xtrue

# Plots
spkf_obj.plot(t_array=range(0, max_iter), measurement_array=xhatstore, sigma_array=SigmaXstore, truth_array=xstore[:-1])