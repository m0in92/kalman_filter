"""
This example is of using equivalent circuit model (ECM) for modeling the state-of-charge(SoC) and terminal voltage of the
lithium-ion batteries.
"""
import numpy as np
import scipy.linalg

from kf.spkf import SPKF


# ECM Parameters
R0 = 0.1
R1 = 0.1
C1 = 0.1
delta_t = 1
Q = 3600 * 1.5 # 1.5 A.hr which is converted to A.s

# state equation
def f_func(x_k, u_k, w_k):
    m1 = np.array([[1,0],[0,np.exp(-delta_t/(R1*C1))]])
    m2 = np.array([[-delta_t/Q],[1-np.exp(-delta_t/(R1*C1))]])
    return m1 @ x_k + m2 * (u_k + w_k)

# output equation
def h_func(x_k, u_k, v_k):
    def OCV(SOC):
        return 3.5 + 0.7 * SOC
    return OCV(x_k[0,0]) - R1 * x_k[1,0] - R0 * u_k + v_k

# SPKF parameters and create SPKF object.
x_hat_int = np.array([[0.5], [0]])
Ny = 1
SigmaX = np.array([[1e-6, 0],[0, 1e-8]])
SigmaW, SigmaV = 0.002, 0.002

spkf_obj = SPKF(xhat=x_hat_int, Ny=Ny, SigmaX=SigmaX, SigmaW=SigmaW, SigmaV=SigmaV, f_func=f_func, h_func=h_func)

# simulation
max_iter = 1000
time_array = np.arange(0, max_iter)
ztrue_array, zhat_array, SigmaZ_array  = np.zeros(max_iter), np.zeros(max_iter), np.zeros(max_iter)
xtrue = np.array([[0.5 + SigmaX[0,0] * np.random.normal()], [0 + SigmaX[1,0] * np.random.normal()]])

for t_ in range(max_iter):
    u = 1 + SigmaX[0,0] * np.random.normal()  # simulate input signal (electric current signal)
    w = (np.transpose(scipy.linalg.cholesky(SigmaW)) * np.random.normal())[0, 0]
    v = (scipy.linalg.cholesky(SigmaV) * np.random.normal())[0, 0]
    ytrue = h_func(x_k=xtrue, u_k=u, v_k=v)
    xtrue = f_func(x_k=xtrue, u_k=u, w_k=w)

    spkf_obj.solve(u=u, ytrue=ytrue)
    ztrue_array[t_] = xtrue[0,0]
    zhat_array[t_] = spkf_obj.xhat[0,0]
    SigmaZ_array[t_] = spkf_obj.SigmaX[0,0]

spkf_obj.plot(t_array=time_array, measurement_array=zhat_array, truth_array=zhat_array, sigma_array=SigmaZ_array)

