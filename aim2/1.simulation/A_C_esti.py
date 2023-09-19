import torch
from torch.autograd import Variable
from torch import Tensor
import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
import simu_Z_Y
import neural_utils

A_gt = Tensor([[1.5, 0.4, -0.6], [-1.0, 0.3, 0.5], [0.2, -0.4, 0.1]])
C_gt = Tensor([[-0.4, 0], [0, 0], [0, 0]])
U = Tensor([[0.5, 0]])

t_max = 30
n_points = 80
z0 = Variable(torch.Tensor([[0.5, 0.3, 0.1]]))

print("step 0: generating and saving simulation z with ground truth A and C")
obs, z, y, times = simu_Z_Y.get_z_y(A_gt, C_gt, U, z0, t_max = 30, n_points = 80)
#torch.save([obs, times], 'obs_t.pt')

print("step 0.5. loading simulation data")
para = torch.load("obs_t.pt")
obs, times = para[0], para[1]
z = obs.detach().numpy().reshape(n_points, -1)
U = U.numpy()
z0 = z[0]

iter_num = 30
data_t = times.numpy().reshape(1,-1)[0]
A_init = np.zeros((3,3))
C_init = np.array([[1, 0, 0], [0, 0, 0]])

A_gt_norm = A_gt/A_gt[0,0]
C_gt_norm = C_gt/C_gt[0,0]
A_gt_norm = A_gt_norm.detach().numpy()
C_gt_norm = C_gt_norm.detach().numpy()
C_gt_norm = C_gt_norm.reshape(2,3)

#placeholders for results
A_list = []
C_list = []
A_dist_list = []
C_dist_list = []

#para for earlystop
n_iters_stop = 2
epochs_no_improve = 0
min_val_loss = np.Inf

print("step 1: compute C using lstsq with all zeros")
C_hat = simu_Z_Y.lstsq_C(z, data_t, A_init, C_init, U)
A_hat = A_init
for i in range(iter_num):
    print(f"============ iter {i} ============")
    print("step 2. use C_hat and current A to get the Z_hat")
    z_hat = odeint(simu_Z_Y.f_c, z0, data_t, args=(A_hat, C_hat, U))

    print("step 3. Use regression to find coefficient to make z_residual smallest")
    regression = LinearRegression()
    linear_model = regression.fit(z_hat, z)
    beta0 = linear_model.intercept_
    beta1 = linear_model.coef_
    z_pred = linear_model.predict(z_hat)
    z_res = z - z_pred
    # z_res = z - beta0 - np.matmul(z_hat, beta1)


    print("step 4: fit the z_res into NeuralODE and extract A_pred")
    z_res = z_res.reshape(80, 1, 3).astype("float32")
    z_res = torch.from_numpy(z_res)
    A_pred = neural_utils.Neural(z_res, n_iter = 3000, n_epochs = 10, lr = 0.001)


    print("step 5: use the A_pred to replace A_init, use A_pred to cal new C_pred")
    A_hat = A_pred
    C_hat = simu_Z_Y.lstsq_C(z, data_t, A_pred, C_init, U)
    A_list.append(A_hat)
    C_list.append(C_hat)


    print("step 6: calculate forbius norm distance, back to step 2")
    A_norm = A_pred/A_pred[0,0]
    C_norm = C_hat/C_hat[0,0]

    diff = A_norm - A_gt_norm
    diff2 = C_norm - C_gt_norm
    fro_norm = np.linalg.norm(diff, 'fro')
    fro_norm2 = np.linalg.norm(diff2, 'fro')
    A_dist_list.append(fro_norm)
    C_dist_list.append(fro_norm2)

    #if norm not decreasing, early stop
    if fro_norm < min_val_loss:
        epochs_no_improve = 0
        min_val_loss = fro_norm
    else:
        epochs_no_improve += 1
        
    if i > 5 and epochs_no_improve == n_iters_stop:
            print('------Early stopping!')
            break
    

np.save("A_esti_result.npy", A_list)
np.save("C_esti_result.npy", C_list)
np.save("A_dist_result.npy", A_dist_list)
np.save("C_dist_result.npy", C_dist_list)





