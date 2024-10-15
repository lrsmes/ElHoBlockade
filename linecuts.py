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

    # find peaks
    peak_list = []
    for i, row in enumerate(map.T):
        peaks, properties = find_peaks(-row, height=0.01e-5, prominence=0.05e-5, distance=70)
        for peak in peaks:
            if peak > 80:
                peak_list.append((i, peak))

    # plotting
    peak_rows, peak_cols = zip(*peak_list)

    Bx_full = np.linspace(0, 1.5, map.shape[1])
    fig = plt.figure(figsize=(12, 6))
    im = plt.pcolormesh(Bx_full, FG14[0][55:-10],  map, vmin=-10e-6, vmax=10e-6)
    plt.scatter(Bx_full[list(peak_rows)], FG14[0][55:-10][list(peak_cols)], facecolors='none', edgecolors='red', alpha=0.5)
    plt.axhline(FG14[0][55:-10][80])
    plt.ylim(5.1928, 5.189)
    fig.colorbar(im)
    plt.show()


if __name__ == "__main__":
    main()