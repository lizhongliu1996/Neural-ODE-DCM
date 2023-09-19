import os
import torch
from torch.autograd import Variable
from torch import Tensor
import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
import simu_Z_Y_abcd
import neural_utils
import pandas as pd


TR = 0.8
print("step 0: load fmri data and event data")
dim = len(os.listdir('abcd_small/'))
for sub in range(0, 10):
    print(f"running algorithm for subject {sub}")
    fmri_name = 'abcd_small/' + os.listdir('abcd_small/')[sub]
    event_name = 'train_event/' + os.listdir('train_event/')[sub]
    fmri_test_name = 'abcd_run2_deconv/' + os.listdir('abcd_run2_deconv/')[sub]
    event_test_name = 'event2_small/' + os.listdir('event2_small/')[sub]
    
    fmri = pd.read_csv(fmri_name)
    event = pd.read_csv(event_name, header=0)
    fmri_test = pd.read_csv(fmri_test_name)
    event_test = pd.read_csv(event_test_name, header=0)
 
    obs = np.array(fmri)
    n_points = fmri.shape[0]
    z = obs.reshape(n_points, -1)
    U = np.array(event)
    z0 = torch.from_numpy(z[0])
    
    obs_test = np.array(fmri_test)
    n_points2 = fmri_test.shape[0]
    z_test = obs_test.reshape(n_points2, -1)
    U_test = np.array(event_test)
    z0_test = torch.from_numpy(z_test[0])
    
    #append U to have same length as z
    U_exdim = n_points - U.shape[0]
    U_new_rows = np.zeros((U_exdim, U.shape[1]))
    U_new = np.concatenate([U, U_new_rows])
    U_exdim2 = n_points2 - U_test.shape[0]
    U_new_rows2 = np.zeros((U_exdim2, U_test.shape[1]))
    U_new2 = np.concatenate([U_test, U_new_rows2])
    
    dim1, dim2 = fmri.shape
    dim3, dim4 = event.shape
    A_init = np.zeros((dim2, dim2))
    #C_init = np.random.uniform(-1, 1, (dim4, dim2))
    #C_init = np.random.uniform(0, 1, (dim4, dim2))
    C_init = np.zeros((dim4, dim2))
    
    t_max = TR * (n_points)
    times_np = np.linspace(0, t_max, num = n_points)
    times_np = np.hstack([times_np[:, None]])
    times = torch.from_numpy(times_np[:, :, None]).to(z0)
    
    t_max2 = TR * (n_points2)
    times_np2 = np.linspace(0, t_max2, num = n_points2)
    times_np2 = np.hstack([times_np2[:, None]])
    times2 = torch.from_numpy(times_np2[:, :, None]).to(z0_test)
    
    iter_num = 30
    data_t = times.numpy().reshape(1,-1)[0]
    data_t2 = times2.numpy().reshape(1,-1)[0]
    
    #placeholders for results
    A_list = []
    C_list = []
    A_dist_list = []
    C_dist_list = []
    loss_list = []

    #para for earlystop
    n_iters_stop = 2
    epochs_no_improve = 0
    min_val_loss = np.Inf 
    
    print("step 1: compute C using lstsq with set A all zeros")
    C_hat = simu_Z_Y_abcd.lstsq_C(z, data_t, A_init, C_init, U_new)
    A_hat = A_init
    for i in range(iter_num):
        print(f"============ iter {i} ============")
        print("step 2. use C_hat and current A to get the Z_hat")
        z_hat = odeint(simu_Z_Y_abcd.f_c, z0, data_t[:n_points], args=(A_hat, C_hat, U_new))

        print("step 3. Use regression to find coefficient to make z_residual smallest")
        regression = LinearRegression()
        linear_model = regression.fit(z_hat, z)
        z_pred = linear_model.predict(z_hat)
        z_res = z - z_pred

        print("step 4: fit the z_res into NeuralODE and extract A_pred")
        z_res = z_res.reshape(dim1, 1, dim2).astype("float32")
        z_res = torch.from_numpy(z_res)
        A_pred = neural_utils.Neural(z_res, n_iter = 3000, n_epochs = 10, lr = 0.005)
        print("finish training")

        print("step 5: use the A_pred to replace A_init, use A_pred to cal new C_pred")
        A_hat = A_pred
        C_hat = simu_Z_Y_abcd.lstsq_C(z, data_t, A_pred, C_hat, U_new)

        #load run-02 data and run neural agagin with new z, return loss use that as a early-stop indicator.(write a test function)
        print("step 6: testing using run2 data")
        loss = simu_Z_Y_abcd.test(z_test, U_new2, A_hat, C_hat, data_t2)
        A_norm = A_pred/A_pred[0,0]
        C_norm = C_hat/C_hat[0,0]
        
        if i >= 1:
            diff = A_norm - A_list[-1]
            diff2 = C_norm - C_list[-1]
            fro_norm = np.linalg.norm(diff, 'fro')
            fro_norm2 = np.linalg.norm(diff2, 'fro')
            A_dist_list.append(fro_norm)
            C_dist_list.append(fro_norm2)
            loss_list.append(loss)
            
            #if norm not decreasing, early stop
            if loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = loss
            else:
                epochs_no_improve += 1
                
        else:
            A_dist_list.append(0)
            C_dist_list.append(0)
            
        A_list.append(A_norm)
        C_list.append(C_norm)
        loss_list.append(loss)
        
        if i > 5 and epochs_no_improve == n_iters_stop:
            print(f'------Early stopping!, done for subject {sub}')
            break
    
    np.save(f"abcd_result2/A_esti_result{sub}.npy", A_list)
    np.save(f"abcd_result2/C_esti_result{sub}.npy", C_list)
    np.save(f"abcd_result2/A_dist{sub}.npy", A_dist_list)
    np.save(f"abcd_result2/C_dist{sub}.npy", C_dist_list)
    np.save(f"abcd_result2/loss_result{sub}.npy", loss)
print("done for all subjects")




