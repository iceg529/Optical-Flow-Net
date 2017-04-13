import h5py
from scipy import misc
import numpy as np
import os 
import sys

f = h5py.File("trainData1.h5", 'r')
dset1 = f['data1']
dset2 = f['flow']
print(dset1.shape)
print(dset2.shape)
f.close()