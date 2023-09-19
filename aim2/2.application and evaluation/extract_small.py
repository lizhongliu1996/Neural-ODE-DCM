import shutil
import os
import numpy as np

name_ls = np.load("small_ls.npy")

for name in name_ls:
    shutil.copy(name,"/User/lliu/run2_small/")