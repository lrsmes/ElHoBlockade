import os
import autograd.numpy as np
import hdf5_helper as helper
import matplotlib.pyplot as plt
from scipy import constants
from auto_fit import fit_with_derivative
from utils import find_hdf5_files
from HDF5Data import HDF5Data
from sklearn.preprocessing import normalize, Normalizer
from scipy.signal import detrend, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

mu_b = constants.physical_constants['Bohr magneton in eV/T']


def load_data(files, dir):
    FG14 = []
    Bx = []
    DemodR = []

    for file in files:
        hdf5data = HDF5Data(readpath=file)
        hdf5data.set_arrays()
        hdf5data.set_array_tags()
        FG14.append(hdf5data.arrays[0].T[0])
        Bx.append(hdf5data.arrays[1][0])
        DemodR.append(np.flip(hdf5data.arrays[3], axis=0))

    return FG14, Bx, DemodR



def stitch_maps(arr1, arr2, vals, FG14):
    first = np.argmin(np.abs(FG14[0]-vals[0]))
    second = np.argmin(np.abs(FG14[0]-vals[1]))

    position = first - second
    arr1_height, arr1_width = arr1.shape
    arr2_height, arr2_width = arr2.shape

    # Create a new array to hold the result, initialized with zeros
    result = np.full((arr1_height, arr1_width+arr2_width), np.mean(arr2))

    # Determine the range for stitching arr1 and arr2 based on the shift (position)
    if position > 0:
        # arr2 is shifted down, pad at the top
        result[:, :arr1_width] = arr1  # Copy arr1 top part into result
        result[:arr1_height - position, arr1_width:] = arr2[position:, :]  # Copy shifted arr2 into result
    elif position < 0:
        # arr2 is shifted up, pad at the bottom
        abs_pos = abs(position)
        result[:, :arr1_width] = arr1  # Copy shifted arr2 into result
        result[abs_pos:, arr1_width:] = arr2[:arr1_height - abs_pos, :]  # Copy arr1 bottom part
    else:
        # No shift, simply concatenate
        result = arr1 + arr2

    return result

def f(x, m, b):
    return m*x+b

def calc_peak_distance_tuple(arr1, arr2, y_threshold=0):
    distance = []
    for i, x in arr1:
        for j, y in arr2:
            if i == j and i > y_threshold:
                diff = np.abs(x-y)
                distance.append((i, diff))
    return distance

def g(x, g, b):
    return 0.5*g*mu_b[0]*x+b


def main():
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "in_plane_linecuts")
    file_names = find_hdf5_files(file_dir)

    FG14, Bx, DemodR = load_data(file_names, current_dir)
    DemodR[1] = np.flip(DemodR[1], axis=1)

    # substract background
    DemodR[1] = np.array([trace - np.mean(trace) for trace in DemodR[1].T]).T
    DemodR[0] = np.array([trace - np.mean(trace) for trace in DemodR[0].T]).T
    DemodR[2] = np.array([trace - np.mean(trace) for trace in DemodR[2][:, :-1].T]).T

    for i, map in enumerate(DemodR):
        DemodR[i] = gaussian_filter1d(map, 1, axis=0)

    # stitch maps to one
    start_second = np.argmin(np.abs(np.flip(Bx[1])-0.640))
    end_second = np.argmin(np.abs(np.flip(Bx[1])-0.735))
    end_third = np.argmin(np.abs(np.flip(Bx[1])-0.907))
    second_jump_map = DemodR[1][:, start_second:end_second]
    third_jump_map = DemodR[1][:, end_second:end_third]
    fourth_jump_map = DemodR[1][:, end_third:]
    map = stitch_maps(DemodR[0], DemodR[2], (5.18989, 5.189896), FG14)
    map = stitch_maps(map, second_jump_map, (5.1899, 5.19052), FG14)
    map = stitch_maps(map, third_jump_map, (5.18995, 5.18992), FG14)
    map = stitch_maps(map, fourth_jump_map, (5.18994, 5.18979), FG14)

    map = map[55:-10]
    FG14[0] = FG14[0][55:-10]

    # find peaks
    Bx_full = np.linspace(0, 1.5, map.shape[1])
    ny_peaks = []
    upper_peaks = []
    lower_peaks = []
    for i, row in enumerate(map.T):
        peaks, properties = find_peaks(-row, height=0.6e-5, width=22)
        for peak in peaks:
            if peak > 185 and peak < 215:
                ny_peaks.append((i, peak))
        peaks, properties = find_peaks(-row, height=-0.075e-5, width=16)
        for peak in peaks:
            if FG14[0][peak] > f(Bx_full[i], 0.00035, 5.19155) and FG14[0][peak] < f(Bx_full[i], 0.00035, 5.192):
                lower_peaks.append((i, peak))
        peaks, properties = find_peaks(-row, height=-0.2e-5, width=12)
        for peak in peaks:
            if FG14[0][peak] > f(Bx_full[i], -0.00035, 5.19055) and FG14[0][peak] < f(Bx_full[i], -0.00035, 5.19095):
                #print(f(Bx_full[i], -0.00035, 5.19095), FG14[0][peak], peak)
                upper_peaks.append((i, peak))

    # plotting
    ny_rows, ny_cols = zip(*ny_peaks)
    upper_rows, upper_cols = zip(*upper_peaks)
    lower_rows, lower_cols = zip(*lower_peaks)

    # calculate distances
    ny_upper_dist = calc_peak_distance_tuple(ny_peaks, upper_peaks)
    ny_lower_dist = calc_peak_distance_tuple(ny_peaks, lower_peaks)

    dFG = FG14[0][1] - FG14[0][0]

    ny_upper_rows, ny_upper_cols = zip(*ny_upper_dist)
    ny_lower_rows, ny_lower_cols = zip(*ny_lower_dist)

    leverarmFG12 = 0.5798 #eV/V
    leverarmFG14 = leverarmFG12*(5.196/5.157)
    a = -1.73301
    ny_lower_cols = (np.array(ny_lower_cols)*dFG)*leverarmFG14*np.sqrt(1+(1/a)**2)*10**6 #mueV
    ny_upper_cols = (np.array(ny_upper_cols)*dFG)*leverarmFG14*np.sqrt(1+(1/a)**2)*10**6 #mueV

    # fitting
    popt_upper, pcov_upper = curve_fit(g, Bx_full[list(ny_upper_rows)], ny_upper_cols)
    popt_lower, pcov_lower = curve_fit(g, Bx_full[list(ny_lower_rows)], ny_lower_cols)

    print(f'g-Factor upper: {popt_upper[0]*10**-6}; b upper: {popt_upper[1]}')
    print(f'g-Factor lower: {popt_lower[0]*10**-6}; b lower: {popt_lower[1]}')


    fig = plt.figure(figsize=(12, 6))
    im = plt.pcolormesh(Bx_full, FG14[0],  map, vmin=-10e-6, vmax=10e-6)
    plt.scatter(Bx_full[list(ny_rows)], FG14[0][list(ny_cols)], facecolors='none', edgecolors='red', alpha=0.5)
    plt.scatter(Bx_full[list(upper_rows)], FG14[0][list(upper_cols)], facecolors='none', edgecolors='orange', alpha=0.5)
    plt.scatter(Bx_full[list(lower_rows)], FG14[0][list(lower_cols)], facecolors='none', edgecolors='magenta', alpha=0.5)
    #plt.axhline(FG14[0][185], color='red')
    #plt.axhline(FG14[0][215], color='red')
    #plt.plot(Bx_full, f(Bx_full, 0.00035, 5.19155), color='orange')
    #plt.plot(Bx_full, f(Bx_full, 0.00035, 5.192), color='orange')
    #plt.plot(Bx_full, f(Bx_full, -0.00035, 5.19055), color='orange')
    #plt.plot(Bx_full, f(Bx_full, -0.00035, 5.19095), color='orange')
    plt.ylim(5.1928, 5.189)
    fig.colorbar(im)

    plt.figure(figsize=(12, 8))
    plt.scatter(Bx_full[list(ny_upper_rows)[::3]], ny_upper_cols[::3], color='orangered', marker='.')
    plt.plot(Bx_full, g(Bx_full, *popt_upper), color='orangered', linestyle='--')

    plt.figure(figsize=(12, 8))
    plt.scatter(Bx_full[list(ny_lower_rows)[::3]], ny_lower_cols[::3], color='mediumblue', marker='.')
    plt.plot(Bx_full, g(Bx_full, *popt_lower), color='mediumblue', linestyle='--')
    """
    plt.figure(figsize=(12, 8))
    for y in map.T[:20]:
        plt.plot(FG14[0][55:-10], y, c='red', alpha=0.3)
    """

    plt.show()


if __name__ == "__main__":
    main()