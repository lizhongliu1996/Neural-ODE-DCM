import shutil
import os
import numpy as np

name_ls = np.load("run2name_ls.npy")
print(name_ls)

for name in name_ls:
    fname = "abcd_sst_ho/" + name
    shutil.copy(fname,"/Users/lliu/abcd_run2/")