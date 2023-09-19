import shutil
import os
import numpy as np

name_ls = np.load("namelist.npy")


for name in name_ls:
    fname = "sub-" + name + "/"
    os.chdir('sstv3/' + fname + 'ses-baselineYear1Arm1/func')
    for filename in os.listdir():
        if filename.endswith("run-02_events.tsv"):
            shutil.copy(filename,"/Users/lliu/events2/")
    os.chdir('../../../..')
