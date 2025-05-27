import math
import os
import autograd.numpy as np
import hdf5_helper as helper
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import substract_linear_BG, define_resonances_tp, get_value_range, get_single_map, find_hdf5_files, get_slopes, exponential_model
from scipy import constants
from auto_fit import fit_with_derivative
from SingleMap import SingleMap

def poly_model(x, a, b, c):
    return a*x**-b+c

def load_data(file_name, group, dataset, channels, all_maps_pos):
    channel_data = helper.read_file(file_name, group, dataset, channels, information=False)

    FG12_raw = channel_data['FG_12']
    FG14_raw = channel_data['FG_14']

    PG13 = channel_data['PG_13']
    PG13_unique = np.unique(PG13)

    pulse_direction = 1#channel_data['triple_pulse_direction']

    demR_raw = channel_data['UHFLI - Demod1R']

    maps_pos = all_maps_pos

    for pg13 in PG13_unique:
        demR, FG12, FG14 = get_single_map(pg13, FG12_raw, FG14_raw, PG13, demR_raw, pulse_direction, 1)
        #demR_neg, FG12_neg, FG14_neg = get_single_map(pg13, FG12_raw, FG14_raw, PG13, demR_raw, pulse_direction, -1)
        maps_pos.append((demR, FG12, FG14, pg13))
        #maps_neg.append((demR_neg, FG12_neg, FG14_neg, pg13))
        print(pg13)

    return maps_pos

# Get the current working directory
current_dir = os.getcwd()
file_dir = os.path.join(current_dir, "pg13")
file_names = find_hdf5_files(file_dir)

group = "Data"
dataset = "Data/Data"
channels = "Data/Channel names"

all_maps_blockade = []
single_maps_blockade = []

ratios = []
ratios_err = []

for file in file_names:
    # Load the data in the given file
    all_maps_blockade = load_data(file, group, dataset, channels, all_maps_blockade)

###########
# Blockade
###########
print('###################### Blockade #########################')
for map in all_maps_blockade:
    print(map[3])
    map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3], 1, 1, file_dir)
    single_maps_blockade.append(map_obj)

single_maps_blockade[6].set_centers((5.17, 5.35), (5.188, 5.3))
#single_maps_blockade[6].set_comp_fac(0.011)
single_maps_blockade[5].set_centers((5.17, 5.35), (5.19, 5.305))
#single_maps_blockade[5].set_comp_fac(0.011)
single_maps_blockade[4].set_centers((5.17, 5.35), (5.195, 5.31))
#single_maps_blockade[4].set_comp_fac(0.011)
single_maps_blockade[3].set_centers((5.17, 5.36), (5.2, 5.31))
#single_maps_blockade[3].set_comp_fac(0.011)
single_maps_blockade[2].set_centers((5.17, 5.365), (5.2, 5.32))
#single_maps_blockade[2].set_comp_fac(0.011)
single_maps_blockade[1].set_centers((5.18, 5.365), (5.2, 5.32))
single_maps_blockade[0].set_centers((5.18, 5.365), (5.2, 5.32))

for map_obj in single_maps_blockade:
    map_obj.set_comp_fac(0.011, 0.00035)
    map_obj.subtract_background()
    #map_obj.detect_lines()


#5.9
single_maps_blockade[0].add_horizontal_line(-0.9, 5.3748)
single_maps_blockade[0].add_horizontal_line(-0.9, 5.3828)
single_maps_blockade[0].add_vertical_line(-11, 5.1852)
single_maps_blockade[0].add_vertical_line(-11, 5.1898)

#5.95
single_maps_blockade[1].add_horizontal_line(-0.9, 5.3648)
single_maps_blockade[1].add_horizontal_line(-0.9, 5.3751)
single_maps_blockade[1].add_vertical_line(-11, 5.1838)
single_maps_blockade[1].add_vertical_line(-11, 5.1894)

#6
single_maps_blockade[2].add_horizontal_line(-0.9, 5.3588)
single_maps_blockade[2].add_horizontal_line(-0.9, 5.368)
single_maps_blockade[2].add_vertical_line(-11, 5.1823)
single_maps_blockade[2].add_vertical_line(-11, 5.18789)

#6.05
single_maps_blockade[3].add_horizontal_line(-0.9, 5.3506)
single_maps_blockade[3].add_horizontal_line(-0.9, 5.3597)
single_maps_blockade[3].add_vertical_line(-11, 5.1812)
single_maps_blockade[3].add_vertical_line(-11, 5.1865)

#6.1
single_maps_blockade[4].add_horizontal_line(-0.9, 5.3411)
single_maps_blockade[4].add_horizontal_line(-0.9, 5.3502)
single_maps_blockade[4].add_vertical_line(-11, 5.1797)
single_maps_blockade[4].add_vertical_line(-11, 5.1852)

#6.15
single_maps_blockade[5].add_horizontal_line(-0.9, 5.3314)
single_maps_blockade[5].add_horizontal_line(-0.9, 5.3412)
single_maps_blockade[5].add_vertical_line(-11, 5.178)
single_maps_blockade[5].add_vertical_line(-11, 5.1838)

# 6.2
single_maps_blockade[6].add_horizontal_line(-0.9, 5.3225)
single_maps_blockade[6].add_horizontal_line(-0.9, 5.3317)
single_maps_blockade[6].add_vertical_line(-13, 5.1787)
single_maps_blockade[6].add_vertical_line(-13, 5.1828)



for map_obj in single_maps_blockade:
    map_obj.add_triangle()
    map_obj.plot_map()
    map_obj.save_map()
    ratio, sigma_ratio = map_obj.get_ratio()
    ratios.append(ratio)
    ratios_err.append(sigma_ratio)

# Blockade
plt.figure(figsize=(20, 12))
ls = 'dashed'
pg13 = []
for elem in all_maps_blockade:
    pg13.append(elem[3])
print(pg13)
plt.scatter(pg13, ratios,
            facecolors='none', edgecolors='orangered')
# plt.errorbar(t_read_s, ratios_blockade,
#             ratios_err_blockade,
#             linestyle='None', marker='.',
#             color='mediumvioletred', elinewidth=0.5)
np.save(os.path.join(file_dir, 'pg13.npy'), pg13)
np.save(os.path.join(file_dir, 'ratios_blockade.npy'), ratios)
np.save(os.path.join(file_dir, 'ratios_err_blockade.npy'), ratios_err)

popt, pcov = curve_fit(poly_model,
                        ydata=ratios,
                        xdata=pg13,
                        sigma=ratios_err)
                        #p0=[1, 2])

# popt, pcov = fit_with_derivative(exponential_model, t_read_s, ratios_blockade, p0=[1, 0.5, 0.5])

x = np.linspace(0, max(pg13), 100)
print(popt, "\n", pcov)
sigma = np.sqrt(np.diag(pcov[0]))

plt.plot(x, exponential_model(x, *popt),
         label=rf'$\tau_b$ : {popt[0]} $\pm$ {sigma}$\mu$s',
         color='mediumblue', linestyle=ls, alpha=0.8)

plt.yscale('log')

#plt.legend()
plt.ylabel('Intensity ratio')
plt.xlabel(r'$V_{PG}$')
plt.tight_layout()
plt.savefig(os.path.join(file_dir, f'prop_vs_pg13.png'))

plt.show()
