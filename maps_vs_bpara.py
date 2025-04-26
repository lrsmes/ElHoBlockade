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

def load_data(file_name, group, dataset, channels, all_maps_pos, all_maps_neg):
    channel_data = helper.read_file(file_name, group, dataset, channels, information=False)

    FG12_raw = channel_data['FG_12']
    FG14_raw = channel_data['FG_14']

    Bpara = channel_data['Triton 3 - Bz']
    Bpara_unique = np.unique(Bpara)

    pulse_direction = channel_data['triple_pulse_direction']

    demR_raw = channel_data['UHFLI - Demod1R']

    maps_pos = all_maps_pos
    maps_neg = all_maps_neg

    for bpara in Bpara_unique:
        demR, FG12, FG14 = get_single_map(bpara, FG12_raw, FG14_raw, Bpara, demR_raw, pulse_direction, 1)
        demR_neg, FG12_neg, FG14_neg = get_single_map(bpara, FG12_raw, FG14_raw, Bpara, demR_raw, pulse_direction, -1)
        maps_pos.append((demR, FG12, FG14, bpara))
        maps_neg.append((demR_neg, FG12_neg, FG14_neg, bpara))
        print(bpara)

    return maps_pos, maps_neg

# Get the current working directory
current_dir = os.getcwd()
file_dir = os.path.join(current_dir, "linecut_400mT")
file_names = find_hdf5_files(file_dir)

group = "Data"
dataset = "Data/Data"
channels = "Data/Channel names"

all_maps_blockade = []
all_maps_transport = []
single_maps_blockade = []
single_maps_transport = []

for file in file_names:
    # Load the data in the given file
    if 'maps' in file:
        all_maps_blockade, all_maps_transport = load_data(file, group, dataset, channels, all_maps_blockade,
                                                          all_maps_transport)

###########
# Blockade
###########
print('###################### Blockade #########################')
for map in all_maps_blockade:
    print(map[3])
    map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3]*1e3, 1, 1, file_dir, )
    single_maps_blockade.append(map_obj)

single_maps_blockade[0].set_centers((5.19, 5.325), (5.18, 5.34))
single_maps_blockade[1].set_centers((5.19, 5.325), (5.18, 5.34))
single_maps_blockade[2].set_centers((5.19, 5.325), (5.18, 5.34))
single_maps_blockade[3].set_centers((5.19, 5.325), (5.18, 5.34))
single_maps_blockade[4].set_centers((5.19, 5.325), (5.18, 5.34))

for map_obj in single_maps_blockade:
    map_obj.set_comp_fac(0.01)
    map_obj.subtract_background()
    map_obj.plot_map()
    map_obj.save_map()


###########
# Transport
###########
print('###################### Transport #########################')
for map in all_maps_transport:
    print(map[3])
    map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3]*1e3, -1, 1, file_dir, )
    single_maps_transport.append(map_obj)

single_maps_transport[0].set_centers((5.19, 5.325), (5.18, 5.34))
single_maps_transport[1].set_centers((5.19, 5.325), (5.18, 5.34))
single_maps_transport[2].set_centers((5.19, 5.325), (5.18, 5.34))
single_maps_transport[3].set_centers((5.19, 5.325), (5.18, 5.34))
single_maps_transport[4].set_centers((5.19, 5.325), (5.18, 5.34))

for map_obj in single_maps_transport:
    map_obj.set_comp_fac(0.01)
    map_obj.subtract_background()
    map_obj.plot_map()
    map_obj.save_map()
