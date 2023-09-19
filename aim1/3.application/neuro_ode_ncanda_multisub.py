import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn import preprocessing

def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z

class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None


class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]

class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(264, 264, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)

class RandomLinearODEF(LinearODEF):
    def __init__(self):
        super(RandomLinearODEF, self).__init__(torch.randn(264, 264)/5.0)


def to_np(x):
    return x.detach().cpu().numpy()

def conduct_experiment(obs, ode_trained, n_steps, epoch=10):
    # Create data
    z0 = Variable(torch.Tensor([[-0.3611,  0.4918, -0.5540,  0.3499, -0.9785,  0.7034,  2.9509,  1.9345,
        -1.6317, -0.0879,  0.0508, -1.3766,  1.3198,  0.6977, -0.3382, -0.6307,
        -0.0203, -0.6318, -1.6345, -0.9259,  0.7250, -1.0283, -0.0036, -0.8241,
         0.0083, -1.1871,  1.5538, -0.6260,  0.6042,  1.5055,  0.0880,  0.1972,
        -0.2271,  1.4975,  0.8703, -1.4955, -0.3549, -1.5776,  0.1272, -1.0559,
        -0.6383,  1.0138, -0.1557,  1.0369, -0.3808,  0.3126, -1.9406,  0.9570,
        -0.1706,  1.3011, -1.6338,  1.3933,  0.1429, -0.6198, -0.0911, -0.1486,
         0.1556,  2.3167, -1.4948, -2.2918,  2.1877, -1.3840,  0.5380, -0.7230,
        -0.1120, -0.7997,  1.6869,  0.0984,  0.3853, -1.1027, -0.6760, -0.0278,
         1.0903,  0.8943,  0.9145,  0.0092,  1.9206,  1.3139,  0.5482,  0.6147,
        -1.1204,  0.7206, -0.3276,  1.6444, -0.1027,  1.6611,  1.1672, -1.1570,
         2.8939, -2.0304, -1.2154, -0.5305,  0.7983,  0.4083, -0.2376, -1.0552,
        -0.4090, -0.3477,  0.7967, -0.3949,  0.8473, -1.7827, -0.2723,  0.0048,
        -1.0050, -2.1996, -0.2127, -0.7669,  1.1794,  1.3060,  0.5461,  1.0095,
        -0.1751, -1.7064,  0.2025, -0.2361, -0.8079, -0.1765, -0.5715, -0.3173,
        -0.3266, -0.4069, -1.3615,  0.7795, -0.7621, -0.0130, -0.7490,  0.8126,
         1.3014, -0.9542,  0.5709, -1.0170, -0.5333, -0.3012, -0.7838,  0.0822,
        -1.8561, -0.7537,  0.1899,  1.4471,  1.2453, -0.1046, -1.9616, -0.0444,
        -0.7931,  0.0700, -0.8177, -0.4806, -1.6357,  0.8500,  2.4234,  0.2799,
        -0.3490, -0.5876, -0.1143,  0.1640, -0.3860,  0.6427, -1.5696, -2.1643,
         0.6146,  1.5466, -0.2383, -1.7386,  0.9204,  0.4483,  0.5360, -0.0748,
         0.9627,  1.3090,  2.0138, -0.2901, -0.2484, -0.4070,  0.9661, -2.0419,
         0.2550,  0.9964,  1.3163,  0.2980,  0.8942,  0.5523, -1.2995,  0.2312,
        -1.0494, -0.1110, -1.3554,  0.4107, -0.0161,  0.6258, -0.0351, -0.4947,
        -0.2776, -0.0985,  1.1404,  0.5095,  0.1049, -0.0077, -1.0381,  0.8684,
        -0.5758,  2.2912, -0.4177,  0.7538,  0.4172, -1.0972, -0.9410,  1.5769,
         1.8089,  0.9451, -1.0868, -2.2048,  0.7410,  0.3239, -0.0316, -0.8639,
         0.7459,  1.0891,  0.4473,  2.4007, -0.2739,  0.4405, -0.7605, -0.4677,
         1.0531,  0.7599,  0.7273,  0.3023,  0.8011,  1.0491, -1.4288, -0.1575,
        -0.1670,  0.6337,  1.9694, -0.9721, -0.0125,  1.1740, -0.6497,  1.0803,
         0.3517, -1.3987,  1.8817, -0.9399,  0.3920,  1.1844,  0.0914,  1.5539,
        -0.3659,  0.4428, -0.3209, -0.3586, -0.4489,  0.0193, -0.8406, -0.2745,
         1.0056, -2.4178,  0.0348,  0.6969, -1.0461, -0.6531, -0.7384, -0.6365]]))

    t_max = 6.29*5
    n_points = 269

    print(f"Training Epoch {epoch}...")

    index_np = np.arange(0, n_points, 1, dtype=np.int)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, t_max, num=n_points)
    times_np = np.hstack([times_np[:, None]])

    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    
    # Get trajectory of random timespan
    min_delta_time = 1.0
    max_delta_time = 5.0
    max_points_num = 32
    def create_batch():
        t0 = np.random.uniform(0, t_max - max_delta_time)
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

        idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

        obs_ = obs[idx]
        ts_ = times[idx]
        return obs_, ts_

    # Train Neural ODE
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 2, verbose = True)
    train_losses = []
    for i in range(n_steps):
        obs_, ts_ = create_batch()
        z_ = ode_trained(obs_[0], ts_, return_whole_sequence=True)
        loss = F.mse_loss(z_, obs_.detach())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        train_losses.append(loss.item())
        
    mean_loss = sum(train_losses)/len(train_losses)
    scheduler.step(mean_loss)
    final_pars = list(ode_trained.parameters())
    return train_losses, final_pars, ode_trained

###import files
mat = np.load('ncanda/ncanda_train.npy', allow_pickle = True)

n_epochs = 10
Loss_list = []

#early stop
n_epochs_stop = 1
epochs_no_improve = 0
early_stop = False
min_val_loss = np.Inf

for i in range(407):
    pars_ls = []
    print(f"subject{i}")
    mat_i = mat[i]
    obs = torch.from_numpy(mat_i)
    obs = np.reshape(obs, (269, 1, 264))
    ode_trained = NeuralODE(RandomLinearODEF())
    pars_ls = []

    for epoch in range(1, n_epochs + 1):
        train_losses, final_pars, model  = conduct_experiment(obs, ode_trained, 3000, epoch = epoch)
        mean_train_losses = np.mean(train_losses)
        if mean_train_losses < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = mean_train_losses
        else:
            epochs_no_improve += 1
    
        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            early_stop = True
            break
        print('Mean Train loss: {:.5f}'.format(mean_train_losses))       
        Loss_list.append(np.mean(train_losses))
        pars = final_pars[0].detach()
        pars_ls.append(pars)
        path = f"ncanda/out/model{i}.pt"
        torch.save(model, path)
        
    par = pars_ls[-1].numpy()
    result_df = pd.DataFrame(par)
    result_df.to_csv(f"ncanda/out/ncanda_result{i}.csv")
