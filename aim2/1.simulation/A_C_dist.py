import numpy as np
import copy

#load data
A_simu = np.load("A_result3.npy")
C_simu = np.load("C_result3.npy")

#normalize data
A_norm = copy.deepcopy(A_simu)
C_norm = copy.deepcopy(C_simu)

for i in range(len(A_simu)):
    scaler1 = A_simu[i][0,0]
    scaler2 = C_simu[i][0,0]

    A_norm[i] = A_simu[i]/scaler1
    C_norm[i] = C_simu[i]/scaler2


#normaliza gt
A_gt = np.array([[-0.6, 0.7, -0.8], [-0.2, 0.3, 1.1], [0.2, -0.5, -0.5]])
C_gt = np.array([[-0.4, 0, 0], [0, 0, 0]])

A_gt_norm = A_gt/A_gt[0,0]
C_gt_norm = C_gt/C_gt[0,0]

#generate data graph
dist = []
for i in range(len(A_norm)):
    
    diff = A_norm[i] - A_gt_norm
    diff2 = C_norm[i] - C_gt_norm
    fro_norm = np.linalg.norm(diff, 'fro')
    fro_norm2 = np.linalg.norm(diff2, 'fro')
    dist.append([fro_norm, fro_norm2])

    
np.save("AC_dist3.npy", dist)