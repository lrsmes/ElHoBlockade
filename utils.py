import math
import os
import numpy as np
import hdf5_helper as helper
import matplotlib.pyplot as plt
from scipy import stats, ndimage
#from skimage import filters
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, detrend
from scipy.stats import skewnorm
#from scipy.integrate import cumtrapz
#from skimage.measure import find_contours
from Data_analysis_and_transforms import correct_median_diff

from scipy import constants

##################################
# Simple functions
##################################
def linear_model(x, m, b):
    return m*x + b

def exponential_model(t, tau, A, B):
    return A * np.exp(-t/tau) + B

# Differentiation function to remove linear background
def differentiate_2d(Z, dx, axis=0):
    """
    Differentiate 2D data along a given axis (0 = x, 1 = y).
    """
    padded_Z = np.pad(Z, pad_width=1, mode='wrap')
    dZ = np.zeros_like(Z)

    if axis == 0:  # Differentiate along x-axis
        dZ = (padded_Z[2:, 1:-1] - padded_Z[:-2, 1:-1]) / (2 * dx)

    elif axis == 1:  # Differentiate along y-axis
        dZ = (padded_Z[1:-1, 2:] - padded_Z[1:-1, :-2]) / (2 * dx)

    return dZ[1:-1, 1:-1]

# Integration function to reconstruct after background removal
def integrate_2d(dZ, dx, axis=0):
    """
    Integrate 2D data along a given axis (0 = x, 1 = y).
    """
    Z_int = 0
    if axis == 0:  # Integrate along x-axis
        pass
        #Z_int = cumtrapz(dZ, dx=dx, axis=0, initial=0)
    elif axis == 1:  # Integrate along y-axis
        pass
        #Z_int = cumtrapz(dZ, dx=dx, axis=1, initial=0)
    return Z_int


def find_hdf5_files(folder_path, extension=".hdf5"):
    # List to store paths of files with the specified extension
    hdf5_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file ends with the specified extension
            if file.endswith(extension):
                # Construct the full path and add to the list
                hdf5_files.append(os.path.join(root, file))

    return hdf5_files

##################################
# Data visualization and loading
##################################

def get_single_map(t_read_value, FG_12, FG_14, T_read, dem_R, pulse_direction = None, pulse_dir = 1, tp_dir = None):

    if pulse_direction is not None:
        condition = np.where((T_read == t_read_value) & (pulse_direction == pulse_dir))
    else:
        condition = np.where((T_read == t_read_value))

    dim = int(dem_R[condition].shape[0] / FG_12.shape[0])

    if not isinstance(tp_dir, np.ndarray):
        return dem_R[condition].reshape((FG_12.shape[0],dim)), FG_12[condition].reshape((FG_12.shape[0],dim)), FG_14[condition].reshape((FG_12.shape[0],dim))
    else:
        return dem_R[condition].reshape((FG_12.shape[0],dim)), FG_12[condition].reshape((FG_12.shape[0],dim)), \
            FG_14[condition].reshape((FG_12.shape[0],dim)), tp_dir[condition].reshape((FG_12.shape[0],dim))


def get_value_range(dem_R):
    hist, bins = np.histogram(dem_R.flatten(), bins=1000)
    bins = bins[1::]

    count_threshold = 10
    condition = np.where(hist > count_threshold)
    bins_temp = bins[condition]

    range_width = abs(np.min(bins_temp) - np.max(bins_temp))
    pad = 0.1
    min_bin = np.min(bins_temp) - pad * range_width
    max_bin = np.max(bins_temp) + pad * range_width

    return min_bin, max_bin

##################################
# Data preprocessing and analysis
##################################

def get_slopes(map):
    # Apply Sobel filter to detect edges
    sobel_h = filters.sobel_h(map)  # horizontal edges
    sobel_v = filters.sobel_v(map)  # vertical edges

    print(sobel_h)

    # Combine the edges to get the overall gradient magnitude
    edges = np.hypot(sobel_h, sobel_v)

    # Threshold the gradient to highlight the edges (peaks)
    threshold_value = filters.threshold_otsu(edges)
    binary_map = edges > threshold_value

    plt.figure()
    plt.imshow(binary_map)
    plt.colorbar()
    plt.show()

    # Find coordinates of the detected edges
    coords = np.column_stack(np.where(binary_map))

    # Perform linear regression on the detected edges
    if len(coords) > 1:
        # Horizontal slope (y = mx + c)
        h_slope, _ = np.polyfit(coords[:, 1], coords[:, 0], 1)
        # Vertical slope (x = my + c)
        v_slope, _ = np.polyfit(coords[:, 0], coords[:, 1], 1)
    else:
        h_slope = v_slope = None

    return h_slope, v_slope

def substract_linear_BG(dem_R, FG_12, FG_14, xl, xr, yt, yb, subtract=False):
    dem_R_temp = dem_R[:, ::-1].T
    #dem_R_temp = detrend(dem_R_temp)

    # Compute derivatives
    #x = np.arange(0, 201, 1)
    dx = 1#x[1] - x[0]
    #dZ_dx = differentiate_2d(dem_R_temp.T, dx)

    # Integrate back
    #integral = integrate_2d(dZ_dx, dx).T

    #integral = np.clip(integral, p_low, p_high)

    # region Plot TP
    min_bin, max_bin = get_value_range(dem_R_temp)

    fig, axs = plt.subplots(1, 2, layout='constrained')
    ax1, ax2 = axs[0], axs[1]
    im = ax1.imshow(dem_R_temp, vmin=min_bin, vmax=max_bin)

    # plt.xlabel(r'$FG_{12}$ (V)')
    # plt.xticks(ticks=[np.argmin(np.abs(FG_12[:,0] - tick_value)) for tick_value in [5.32, 5.33]],
    #                         labels=[5.32, 5.33])

    # plt.ylabel(r'$FG_{14}$ (V)')
    # plt.yticks(ticks=[np.argmin(np.abs(FG_14[0,::-1] - tick_value)) for tick_value in [5.21, 5.205]],
    #                         labels=[5.21, 5.205])
    # endregion

    # region Select region for linear fit
    ax1.plot([xl, xr], [yt, yt], color='red')
    ax1.plot([xl, xr], [yb, yb], color='red')

    ax1.plot([xl, xl], [yt, yb], color='red')
    ax1.plot([xr, xr], [yt, yb], color='red')
    # endregion

    # region Fit linear model
    dem_R_region = dem_R_temp[yt:yb, xl:xr]
    dem_R_region_ave = np.mean(dem_R_region, axis=1)
    popt, _ = curve_fit(linear_model, range(dem_R_region_ave.shape[0]), dem_R_region_ave)

    # plt.figure()
    # plt.plot(dem_R_region_ave)
    # plt.plot(range(dem_R_region_ave.shape[0]), linear_model(range(dem_R_region_ave.shape[0]), *popt))
    # endregion

    # region Subtract linear BG
    # BG = np.ones(shape=(dem_R_temp.shape[0],dem_R_temp.shape[1])) * np.arange(0, dem_R_temp.shape[0]).T * popt[0]
    BG = (np.arange(0, dem_R_temp.shape[0]) * popt[0] * np.ones(shape=(dem_R_temp.shape[1]))[:, np.newaxis]).T

    print(BG.shape, dem_R_temp.shape)

    if subtract:
        min_bin, max_bin = get_value_range(dem_R_temp - BG)

        ax2.set_title('Substracted')
        im = ax2.imshow(dem_R_temp - BG, vmin=min_bin, vmax=max_bin)
        # im = plt.imshow(BG)
    # endregion
    plt.close(fig)

    dem_R_sub = dem_R_temp - BG

    p_low, p_high = np.percentile(dem_R_sub, (2, 98))
    #dem_R_sub = np.clip(dem_R_sub, p_low, p_high)
    dem_R_sub = correct_median_diff(dem_R_sub.T).T
    #dem_R_corrected = correct_median_diff(dem_R_temp.T).T

    return dem_R_sub #integral

def define_resonances_tp(dem_R, FG_12, FG_14, file_dir, tread, pulse_dir,
                         h_res_slope=13.5 / 180,
                         h_res_offset_b=108,
                         h_res_offset_t=62,

                         v_res_slope=120 / 150,
                         v_res_offset_l=-17,
                         v_res_offset_r=63,

                         first_ny_diff = 55,
                         second_ny_diff = 68,
                         alpha_diff = 88
                         ):
    first_ny_offset = h_res_offset_b + first_ny_diff
    second_ny_offset = h_res_offset_b + second_ny_diff
    alpha_offset = h_res_offset_b + alpha_diff

    min_bin, max_bin = get_value_range(dem_R)

    dem_R = (dem_R - min_bin) / (max_bin - min_bin)

    fig = plt.figure(figsize=(10, 10))
    im = plt.imshow(dem_R, vmin=0, vmax=1)

    # region Plot outline of the TP
    plt.plot([0, FG_12.shape[0] - 1], [h_res_offset_b, h_res_offset_b + h_res_slope * (FG_12.shape[0] - 1)],
             linewidth=1., color="white", linestyle='dotted')
    plt.plot([0, FG_12.shape[0] - 1], [h_res_offset_t, h_res_offset_t + h_res_slope * (FG_12.shape[0] - 1)],
             linewidth=1., color="white", linestyle='dotted')

    # # -----------------------------------------------------------------------
    # # Convert parameters of vertical resonances into parameters of functions of x
    # # rather than y:
    v_res_slope_of_x = 1 / v_res_slope
    v_res_offset_l_of_x = -(v_res_offset_l / v_res_slope)
    v_res_offset_r_of_x = -(v_res_offset_r / v_res_slope)

    plt.plot([v_res_offset_l, v_res_offset_l + v_res_slope * (FG_12.shape[1] - 1)],
             [linear_model(v_res_offset_l, v_res_slope_of_x, v_res_offset_l_of_x),
              linear_model(v_res_offset_l + v_res_slope * (FG_12.shape[1] - 1), v_res_slope_of_x,
                           v_res_offset_l_of_x)], linewidth=1., color="white", linestyle='dotted')
    plt.plot([v_res_offset_r, v_res_offset_r + v_res_slope * (FG_12.shape[1] - 1)],
             [linear_model(v_res_offset_r, v_res_slope_of_x, v_res_offset_r_of_x),
              linear_model(v_res_offset_r + v_res_slope * (FG_12.shape[1] - 1), v_res_slope_of_x,
                           v_res_offset_r_of_x)], linewidth=1., color="white", linestyle='dotted')

    # -----------------------------------------------------------------------
    # Plot crossing-points of resonances:
    # lower hori - left verti
    point_lh_lv = {'x': (v_res_offset_l_of_x - h_res_offset_b) / (h_res_slope - v_res_slope_of_x),
                   'y': linear_model((v_res_offset_l_of_x - h_res_offset_b) / (h_res_slope - v_res_slope_of_x),
                                     h_res_slope, h_res_offset_b)}
    plt.plot([point_lh_lv['x']], [point_lh_lv['y']], marker='o', color='white')

    # lower hori - right verti
    point_lh_rv = {'x': (v_res_offset_r_of_x - h_res_offset_b) / (h_res_slope - v_res_slope_of_x),
                   'y': linear_model((v_res_offset_r_of_x - h_res_offset_b) / (h_res_slope - v_res_slope_of_x),
                                     h_res_slope, h_res_offset_b)}
    plt.plot([point_lh_rv['x']],
             [point_lh_rv['y']], marker='o', color='red')

    # upper hori - left verti
    point_uh_lv = {'x': (v_res_offset_l_of_x - h_res_offset_t) / (h_res_slope - v_res_slope_of_x),
                   'y': linear_model((v_res_offset_l_of_x - h_res_offset_t) / (h_res_slope - v_res_slope_of_x),
                                     h_res_slope, h_res_offset_t)}
    plt.plot([point_uh_lv['x']],
             [point_uh_lv['y']], marker='o', color='blue')

    # upper hori - right verti
    point_uh_rv = {'x': (v_res_offset_r_of_x - h_res_offset_t) / (h_res_slope - v_res_slope_of_x),
                   'y': linear_model((v_res_offset_r_of_x - h_res_offset_t) / (h_res_slope - v_res_slope_of_x),
                                     h_res_slope, h_res_offset_t)}
    plt.plot([point_uh_rv['x']],
             [point_uh_rv['y']], marker='o', color='black')

    # zero - detuning line
    plt.plot([point_lh_lv['x'], point_uh_rv['x']], [point_lh_lv['y'], point_uh_rv['y']], linewidth=1.5,
             color="white", linestyle='dotted')

    #slope_zero = point_uh_rv['y']+point_lh_lv['y'] / point_uh_rv['x']-point_lh_lv['x']

    # outline ny-lines and alpha line
    if pulse_dir == -1:
        popt, pcov = curve_fit(linear_model, xdata=[point_lh_lv['x'], point_uh_rv['x']],
                               ydata=[point_lh_lv['y'], point_uh_rv['y']])
        slope_zero = popt[0]

        plt.plot([0, FG_12.shape[0] - 1], [first_ny_offset, first_ny_offset + slope_zero * (FG_12.shape[0] - 1)],
                 linewidth=1., color="magenta", linestyle='dotted')

        plt.plot([0, FG_12.shape[0] - 1], [second_ny_offset, second_ny_offset + slope_zero * (FG_12.shape[0] - 1)],
                 linewidth=1., color="magenta", linestyle='dotted')

        plt.plot([0, FG_12.shape[0] - 1], [alpha_offset, alpha_offset + slope_zero * (FG_12.shape[0] - 1)],
                 linewidth=1., color="magenta", linestyle='dotted')

        point_first_ny_l = {'x': (h_res_offset_b - first_ny_offset) / (slope_zero - h_res_slope),
                            'y': linear_model((h_res_offset_b - first_ny_offset) / (slope_zero - h_res_slope),
                                              slope_zero, first_ny_offset)}
        plt.plot([point_first_ny_l['x']], [point_first_ny_l['y']], marker='o', color='magenta')

        point_first_ny_r = {'x': (v_res_offset_r_of_x - first_ny_offset) / (slope_zero - v_res_slope_of_x),
                            'y': linear_model((v_res_offset_r_of_x - first_ny_offset) / (slope_zero - v_res_slope_of_x),
                                              slope_zero, first_ny_offset)}
        plt.plot([point_first_ny_r['x']], [point_first_ny_r['y']], marker='o', color='magenta')

        point_second_ny_l = {'x': (h_res_offset_b - second_ny_offset) / (slope_zero - h_res_slope),
                            'y': linear_model((h_res_offset_b - second_ny_offset) / (slope_zero - h_res_slope),
                                              slope_zero, second_ny_offset)}
        plt.plot([point_second_ny_l['x']], [point_second_ny_l['y']], marker='o', color='magenta')

        point_second_ny_r = {'x': (v_res_offset_r_of_x - second_ny_offset) / (slope_zero - v_res_slope_of_x),
                            'y': linear_model((v_res_offset_r_of_x - second_ny_offset) / (slope_zero - v_res_slope_of_x),
                                              slope_zero, second_ny_offset)}
        plt.plot([point_second_ny_r['x']], [point_second_ny_r['y']], marker='o', color='magenta')

        point_alpha_l = {'x': (h_res_offset_b - alpha_offset) / (slope_zero - h_res_slope),
                            'y': linear_model((h_res_offset_b - alpha_offset) / (slope_zero - h_res_slope),
                                              slope_zero, alpha_offset)}
        plt.plot([point_alpha_l['x']], [point_alpha_l['y']], marker='o', color='magenta')

        point_alpha_r = {'x': (v_res_offset_r_of_x - alpha_offset) / (slope_zero - v_res_slope_of_x),
                            'y': linear_model((v_res_offset_r_of_x - alpha_offset) / (slope_zero - v_res_slope_of_x),
                                              slope_zero, alpha_offset)}
        plt.plot([point_alpha_r['x']], [point_alpha_r['y']], marker='o', color='magenta')

        dem_R_ny_flatten = np.empty(shape=0)
        for y_idx in range(math.ceil(point_uh_rv['y']), math.floor(point_lh_lv['y'])):
            if y_idx >= 0:
                x_temp_start = point_lh_lv['x'] + (point_uh_rv['x'] - point_lh_lv['x']) / (
                        point_uh_rv['y'] - point_lh_lv['y']) * (y_idx - point_lh_lv['y'])
                x_temp_start = max([x_temp_start, 0])
                x_temp_start = int(min([x_temp_start, dem_R.shape[1] - 1]))

                x_temp_end = point_uh_rv['x'] + (point_lh_rv['x'] - point_uh_rv['x']) / (
                        point_lh_rv['y'] - point_uh_rv['y']) * (y_idx - point_uh_rv['y'])
                x_temp_end = max([x_temp_end, 0])
                x_temp_end = int(min([x_temp_end, dem_R.shape[1] - 1]))
                x_temp_end = int(min([(y_idx - first_ny_offset) / slope_zero, x_temp_end]))
                x_temp_end = int(max([x_temp_end, point_first_ny_l['x']]))

                # plt.plot([x_temp_start], [y_idx], color='white', marker='.')
                plt.plot([x_temp_end], [y_idx], color='magenta', marker='.')

                dem_R_ny_flatten = np.concatenate((dem_R_ny_flatten, dem_R[y_idx, x_temp_start:x_temp_end]))

        for y_idx in range(math.ceil(point_lh_lv['y']), math.floor(point_first_ny_l['y'])):
            if y_idx >= 0:
                x_temp_start = point_lh_lv['x'] + (point_lh_rv['x'] - point_lh_lv['x']) / (
                        point_lh_rv['y'] - point_lh_lv['y']) * (y_idx - point_lh_lv['y'])
                x_temp_start = max([x_temp_start, 0])
                x_temp_start = int(min([x_temp_start, dem_R.shape[1] - 1]))

                x_temp_end = point_uh_rv['x'] + (point_lh_rv['x'] - point_uh_rv['x']) / (
                        point_lh_rv['y'] - point_uh_rv['y']) * (y_idx - point_uh_rv['y'])
                x_temp_end = max([x_temp_end, 0])
                x_temp_end = int(min([x_temp_end, dem_R.shape[1] - 1]))
                x_temp_end = int(min([(y_idx - first_ny_offset) / slope_zero, x_temp_end]))
                x_temp_end = int(max([x_temp_end, point_first_ny_l['x']]))

                y_idx = min(dem_R.shape[0] - 1, y_idx)

                plt.plot([x_temp_start], [y_idx], color='white', marker='.')
                plt.plot([x_temp_end], [y_idx], color='magenta', marker='.')

                dem_R_ny_flatten = np.concatenate((dem_R_ny_flatten, dem_R[y_idx, x_temp_start:x_temp_end]))

        dem_R_alpha_flatten = np.empty(shape=0)
        for y_idx in range(math.ceil(point_second_ny_r['y']), math.floor(point_second_ny_l['y'])):
            if y_idx >= 0:
                x_temp_start = point_second_ny_r['x'] + (point_second_ny_l['x'] - point_second_ny_r['x']) / (
                        point_second_ny_l['y'] - point_second_ny_r['y']) * (y_idx - point_second_ny_r['y'])
                x_temp_start = max([x_temp_start, 0])
                x_temp_start = int(min([x_temp_start, dem_R.shape[1] - 1]))
                #x_temp_start = int(max([(y_idx - second_ny_offset) / slope_zero, x_temp_start]))

                x_temp_end = point_uh_rv['x'] + (point_lh_rv['x'] - point_uh_rv['x']) / (
                        point_lh_rv['y'] - point_uh_rv['y']) * (y_idx - point_uh_rv['y'])
                x_temp_end = max([x_temp_end, 0])
                x_temp_end = int(min([x_temp_end, dem_R.shape[1] - 1]))
                x_temp_end = int(min([(y_idx - alpha_offset) / slope_zero, x_temp_end]))
                x_temp_end = int(max([x_temp_end, point_alpha_l['x']]))

                plt.plot([x_temp_start], [y_idx], color='white', marker='.')
                plt.plot([x_temp_end], [y_idx], color='magenta', marker='.')

                dem_R_alpha_flatten = np.concatenate((dem_R_alpha_flatten, dem_R[y_idx, x_temp_start:x_temp_end]))

        for y_idx in range(math.ceil(point_second_ny_l['y']), math.floor(point_alpha_l['y'])):
            if y_idx >= 0:
                x_temp_start = point_second_ny_l['x'] + (point_alpha_l['x'] - point_second_ny_l['x']) / (
                        point_alpha_l['y'] - point_second_ny_l['y']) * (y_idx - point_second_ny_l['y'])
                x_temp_start = max([x_temp_start, 0])
                x_temp_start = int(min([x_temp_start, dem_R.shape[1] - 1]))
                #x_temp_start = int(max([(y_idx - second_ny_offset) / slope_zero, x_temp_start]))

                x_temp_end = point_uh_rv['x'] + (point_lh_rv['x'] - point_uh_rv['x']) / (
                        point_lh_rv['y'] - point_uh_rv['y']) * (y_idx - point_uh_rv['y'])
                x_temp_end = max([x_temp_end, 0])
                x_temp_end = int(min([x_temp_end, dem_R.shape[1] - 1]))
                x_temp_end = int(min([(y_idx - alpha_offset) / slope_zero, x_temp_end]))
                x_temp_end = int(max([x_temp_end, point_alpha_l['x']]))

                y_idx = min(dem_R.shape[0] - 1, y_idx)

                # plt.plot([x_temp_start], [y_idx], color='white', marker='.')
                plt.plot([x_temp_end], [y_idx], color='magenta', marker='.')

                dem_R_alpha_flatten = np.concatenate((dem_R_alpha_flatten, dem_R[y_idx, x_temp_start:x_temp_end]))

    # endregion

    # region Extract intensities
    dem_R_blockade_flatten = np.empty(shape=0)
    for y_idx in range(math.ceil(point_uh_rv['y']), math.floor(point_lh_lv['y'])):
        if y_idx >= 0:
            x_temp_start = point_lh_lv['x'] + (point_uh_rv['x'] - point_lh_lv['x']) / (
                        point_uh_rv['y'] - point_lh_lv['y']) * (y_idx - point_lh_lv['y'])
            x_temp_start = max([x_temp_start, 0])
            x_temp_start = int(min([x_temp_start, dem_R.shape[1] - 1]))

            x_temp_end = point_uh_rv['x'] + (point_lh_rv['x'] - point_uh_rv['x']) / (
                        point_lh_rv['y'] - point_uh_rv['y']) * (y_idx - point_uh_rv['y'])
            x_temp_end = max([x_temp_end, 0])
            x_temp_end = int(min([x_temp_end, dem_R.shape[1] - 1]))

        # plt.plot([x_temp_start], [y_idx], color='white', marker='.')
        # plt.plot([x_temp_end], [y_idx], color='white', marker='.')

            dem_R_blockade_flatten = np.concatenate((dem_R_blockade_flatten, dem_R[y_idx, x_temp_start:x_temp_end]))

    for y_idx in range(math.ceil(point_lh_lv['y']), math.floor(point_lh_rv['y'])):
        if y_idx >= 0:
            x_temp_start = point_lh_lv['x'] + (point_lh_rv['x'] - point_lh_lv['x']) / (
                        point_lh_rv['y'] - point_lh_lv['y']) * (y_idx - point_lh_lv['y'])
            x_temp_start = max([x_temp_start, 0])
            x_temp_start = int(min([x_temp_start, dem_R.shape[1] - 1]))

            x_temp_end = point_uh_rv['x'] + (point_lh_rv['x'] - point_uh_rv['x']) / (
                        point_lh_rv['y'] - point_uh_rv['y']) * (y_idx - point_uh_rv['y'])
            x_temp_end = max([x_temp_end, 0])
            x_temp_end = int(min([x_temp_end, dem_R.shape[1] - 1]))

            y_idx = min(dem_R.shape[0] - 1, y_idx)

        # plt.plot([x_temp_start], [y_idx], color='white', marker='.')
        # plt.plot([x_temp_end], [y_idx], color='white', marker='.')

            dem_R_blockade_flatten = np.concatenate((dem_R_blockade_flatten, dem_R[y_idx , x_temp_start:x_temp_end]))

    dem_R_non_blockade_flatten = np.empty(shape=0)
    for y_idx in range(math.ceil(point_uh_lv['y']), math.ceil(point_uh_rv['y'])):
        if y_idx >= 0:
            x_temp_start = point_uh_lv['x'] + (point_lh_lv['x'] - point_uh_lv['x']) / (
                        point_lh_lv['y'] - point_uh_lv['y']) * (y_idx - point_uh_lv['y'])
            x_temp_start = max([x_temp_start, 0])
            x_temp_start = int(min([x_temp_start, dem_R.shape[1] - 1]))

            x_temp_end = point_uh_lv['x'] + (point_uh_rv['x'] - point_uh_lv['x']) / (
                        point_uh_rv['y'] - point_uh_lv['y']) * (y_idx - point_uh_lv['y'])
            x_temp_end = max([x_temp_end, 0])
            x_temp_end = int(min([x_temp_end, dem_R.shape[1] - 1]))

            plt.plot([x_temp_start], [y_idx], color='red', marker='.')
            plt.plot([x_temp_end], [y_idx], color='blue', marker='.')

            dem_R_non_blockade_flatten = np.concatenate(
                (dem_R_non_blockade_flatten, dem_R[y_idx, x_temp_start:x_temp_end]))

    for y_idx in range(math.ceil(point_uh_rv['y']), math.floor(point_lh_lv['y'])):
        if y_idx >= 0:
            x_temp_start = point_uh_lv['x'] + (point_lh_lv['x'] - point_uh_lv['x']) / (
                        point_lh_lv['y'] - point_uh_lv['y']) * (y_idx - point_uh_lv['y'])
            x_temp_start = max([x_temp_start, 0])
            x_temp_start = int(min([x_temp_start, dem_R.shape[1] - 1]))

            x_temp_end = point_lh_lv['x'] + (point_uh_rv['x'] - point_lh_lv['x']) / (
                        point_uh_rv['y'] - point_lh_lv['y']) * (y_idx - point_lh_lv['y'])
            x_temp_end = max([x_temp_end, 0])
            x_temp_end = int(min([x_temp_end, dem_R.shape[1] - 1]))

            plt.plot([x_temp_start], [y_idx], color='red', marker='.')
            plt.plot([x_temp_end], [y_idx], color='white', marker='.')

            dem_R_non_blockade_flatten = np.concatenate(
                (dem_R_non_blockade_flatten, dem_R[y_idx, x_temp_start:x_temp_end]))
    """
    contours = find_contours(dem_R)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    """

    plt.savefig(os.path.join(file_dir, f'{pulse_dir}_{np.round(tread)}_map.png'))
    plt.close(fig)

    # endregion
    if pulse_dir == -1:
        ## region before ny
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(dem_R_ny_flatten, bins=30)
        axs[1].hist(dem_R_non_blockade_flatten, bins=30)
        plt.savefig(os.path.join(file_dir, f'{pulse_dir}_{np.round(tread)}_ny_hist.png'))
        plt.close(fig)

        tread_mus = tread * 125 * 1e-6
        print(f'Read-out time: {tread_mus}')
        print('Blockade: ', np.round(np.mean(dem_R_ny_flatten), 3), " +- ",
              np.round(np.std(dem_R_ny_flatten), 3))
        print('Non-Blockade: ', np.round(np.mean(dem_R_non_blockade_flatten), 3), " +- ",
              np.round(np.std(dem_R_non_blockade_flatten), 3))

        R_ny = np.mean(dem_R_ny_flatten) / np.mean(dem_R_non_blockade_flatten)
        uncertainty_R_ny = np.sqrt(R_ny ** 2 * ((np.std(dem_R_ny_flatten) / np.mean(dem_R_ny_flatten)) ** 2 + (
                np.std(dem_R_non_blockade_flatten) / np.mean(dem_R_non_blockade_flatten)) ** 2))

        print('Ratio: ', np.round(R_ny, 3), " +- ",
              np.round(uncertainty_R_ny, 3))

        # region between ny and alpha
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(dem_R_alpha_flatten, bins=30)
        axs[1].hist(dem_R_non_blockade_flatten, bins=30)
        plt.savefig(os.path.join(file_dir, f'{pulse_dir}_{np.round(tread)}_alpha_hist.png'))
        plt.close(fig)

        print(f'Read-out time: {tread}')
        print('Blockade: ', np.round(np.mean(dem_R_alpha_flatten), 3), " +- ",
              np.round(np.std(dem_R_alpha_flatten), 3))
        print('Non-Blockade: ', np.round(np.mean(dem_R_non_blockade_flatten), 3), " +- ",
              np.round(np.std(dem_R_non_blockade_flatten), 3))

        R_alpha = np.mean(dem_R_alpha_flatten) / np.mean(dem_R_non_blockade_flatten)
        uncertainty_R_alpha = np.sqrt(R_alpha ** 2 * ((np.std(dem_R_alpha_flatten) / np.mean(dem_R_alpha_flatten)) ** 2 + (
                np.std(dem_R_non_blockade_flatten) / np.mean(dem_R_non_blockade_flatten)) ** 2))

        print('Ratio: ', np.round(R_alpha, 3), " +- ",
              np.round(uncertainty_R_alpha, 3))

        return R_ny, uncertainty_R_ny, R_alpha, uncertainty_R_alpha
    else:
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(dem_R_blockade_flatten, bins=30)
        axs[1].hist(dem_R_non_blockade_flatten, bins=30)
        plt.savefig(os.path.join(file_dir, f'{pulse_dir}_{np.round(tread)}_hist.png'))
        plt.close(fig)

        print(f'Read-out time: {tread}')
        print('Blockade: ', np.round(np.mean(dem_R_blockade_flatten), 3), " +- ",
              np.round(np.std(dem_R_blockade_flatten), 3))
        print('Non-Blockade: ', np.round(np.mean(dem_R_non_blockade_flatten), 3), " +- ",
              np.round(np.std(dem_R_non_blockade_flatten), 3))

        R = np.mean(dem_R_blockade_flatten) / np.mean(dem_R_non_blockade_flatten)
        uncertainty_R = np.sqrt(R ** 2 * ((np.std(dem_R_blockade_flatten) / np.mean(dem_R_blockade_flatten)) ** 2 + (
                    np.std(dem_R_non_blockade_flatten) / np.mean(dem_R_non_blockade_flatten)) ** 2))

        print('Ratio: ', np.round(R, 3), " +- ",
              np.round(uncertainty_R, 3))

        return R, uncertainty_R

