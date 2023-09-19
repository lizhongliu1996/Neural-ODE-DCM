import math
import numpy as np

from scipy.stats import gamma
from math import ceil
from scipy.signal import convolve
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import leastsq

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable

def ode_solve(z0, t0, t1, u, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z_temp = f(z, u, t)
        z = z + h *z_temp
        t = t + h
    return z

class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
    
        return 

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, u, t, flat_parameters, func):
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1],u, func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, u, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, u, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]
        
class LinearODEF(ODEF):
    def __init__(self, W, C):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(W.shape[0], W.shape[1], bias=False)
        self.lin.weight = nn.Parameter(W)
        self.sti = nn.Linear(C.shape[0], C.shape[1], bias=False)
        self.sti.weight = nn.Parameter(C)

    def forward(self, x, u, t):
        return self.lin(x) + self.sti(u)

def to_np(x):
    return x.detach().cpu().numpy()

def gammaHRF(TR, paras=None, len_of_seconds=32, onset_seconds=0):
    
    if paras is None:
        paras = [6,16,1,1,6]

    dt = TR/16
    u = np.array(range(0, ceil(len_of_seconds/dt))) - ceil(onset_seconds/dt)
    
    tmp1 = gamma.pdf(u, a = paras[0]/paras[2], scale = 1/(dt/paras[2])) 
    tmp2 = gamma.pdf(u, a = paras[1]/paras[3], scale = 1/(dt/paras[3]))/paras[4]       
    hrf = tmp1 - tmp2

    hrf = hrf[np.array(range(0, ceil(len_of_seconds/TR)))*16 + 1]

    hrf = hrf/sum(hrf)

    return hrf

def get_z_y(A, C, U, z0, t_max = 30, n_points = 80):
    
    index_np = np.arange(0, n_points, 1, dtype=np.int)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]])
    
    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    ode_true = NeuralODE(LinearODEF(A, C))

    obs = ode_true(z0, U, times, return_whole_sequence=True)
    z_np = obs.detach().numpy().reshape(n_points, -1)

    hrf = gammaHRF(1.0, len_of_seconds = n_points)

    dim1 = z_np.shape[0]
    dim2 = z_np.shape[1]
    y = np.zeros(shape=(dim1, dim2))
    for i in range(dim2):
        temp = convolve(hrf, z_np[:,i], mode = "same")
        y[:,i] = temp

    return obs, z_np, y, times

def f_c(z0, t, A, C, U):
    z = np.matmul(A, z0) + np.matmul(U, C)
    return z[0]

def loss(C, z0, t, A, U, z):
    z_hat = odeint(f_c, z0, t, args=(A, C.reshape(2,3), U))
    loss_total = []
    for v1,v2 in zip(z_hat, z):
        loss_total.append(mse(v1, v2))
    loss_total = np.array(loss_total)
    return loss_total

def lstsq_C(z, t, A, C, U):
    z0 = z[0]
    para = leastsq(loss, C, args = (z0, t, A, U, z))[0]
    return para.reshape(2, 3)
