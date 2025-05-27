import math
import os
import autograd.numpy as np
import hdf5_helper as helper
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import skewnorm
from utils import substract_linear_BG, define_resonances_tp, get_value_range, get_single_map, find_hdf5_files, get_slopes, exponential_model
from scipy import constants
from auto_fit import fit_with_derivative
from SingleMap import SingleMap
from HDF5Data import HDF5Data

current_dir = os.getcwd()
file_dir = os.path.join(current_dir, "bias_maps")
file = os.path.join(file_dir, "792_(1,-1)_(0,0)_zoom_vs_Bpara_vs_bias.hdf5")

hdf5data = HDF5Data(readpath=file)
hdf5data.set_arrays()
hdf5data.set_array_tags()
#hdf5data.set_data()
hdf5data.set_measure_dim()

print(hdf5data.array_tags)
print(hdf5data.arrays.shape)



FG12_pos = hdf5data.arrays[0, :, 2*263:3*263]    #.reshape(hdf5data.measure_dim)
FG12_neg = hdf5data.arrays[0, :, 3*263:4*263]

FG14_pos = hdf5data.arrays[1, :, 2*263:3*263]
FG14_neg = hdf5data.arrays[1, :, 3*263:4*263]

DemR_pos = hdf5data.arrays[4, :, 2*263:3*263]

map = SingleMap(FG12=FG14_pos, FG14=FG12_pos, tread=0.1, pulse_dir=-1, Demod1R=DemR_pos, file_dir=file_dir)
#map.set_centers((5.16, 5.084), (5.135, 5.088))
map.subtract_background()
map.plot_map()
map.save_map()
