import os
import numpy as np
import pylab as pl

import hdf5_helper as helper
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import constants
from utils import find_hdf5_files, linear_model
from HDF5Data import HDF5Data
from sklearn.preprocessing import normalize
from scipy.signal import detrend, find_peaks, find_peaks_cwt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from Data_analysis_and_transforms import correct_median_diff

mpl.rcParams['font.size'] = 18

mu_b = constants.physical_constants['Bohr magneton in eV/T'][0]*1e6


def load_data(files, dir):
    FG14 = []
    Bx = []
    DemodR = []

    for file in files:
        hdf5data = HDF5Data(readpath=file)
        hdf5data.set_arrays()
        hdf5data.set_array_tags()
        FG14.append(hdf5data.arrays[0])
        Bx.append(hdf5data.arrays[1])
        DemodR.append(hdf5data.arrays[3])

    return FG14, Bx, DemodR



def stitch_maps(arr1, arr2, vals, FG14):
    first = np.argmin(np.abs(FG14-vals[0]))
    second = np.argmin(np.abs(FG14-vals[1]))

    position = first - second
    arr1_height, arr1_width = arr1.shape
    arr2_height, arr2_width = arr2.shape

    # Create a new array to hold the result, initialized with zeros
    result = np.full((arr1_height, arr1_width+arr2_width), np.mean(arr2))

    # Determine the range for stitching arr1 and arr2 based on the shift (position)
    if position >= 0:
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

def lorentzian(x, A, x0, gamma):
    return A * gamma**2 / ((x-x0)**2 + gamma**2)

def calc_peak_distance_tuple(arr1, arr2, y_threshold=0):
    distance = []
    for i, x in arr1:
        for j, y in arr2:
            if i == j and i > y_threshold:
                diff = np.abs(x-y)
                distance.append((i, diff))
    return distance

def g_pos(x, g, b, a):
    return 0.5*np.sqrt((g*mu_b*x)**2+b**2)

def g_neg(x, g, b, a):
    return a - 0.5*np.sqrt((g*mu_b*x)**2+b**2)

B_perp = 0.4

def del_E(x, gs, delta):
    return np.sqrt((delta) ** 2 + (gs * mu_b * x) ** 2) #-gv*mu_b*B_perp+

def del_E2(x, gs, delta):
    #return -0.5*gv*mu_b*B_perp + 0.5*np.sqrt((delta - gs * mu_b * B_perp)**2 + (gs * mu_b * x) ** 2)
    #return np.sqrt((delta + 2 * mu_b * x) ** 2 + (2 * mu_b * x) ** 2)
    return np.sqrt((delta + gs * mu_b * B_perp) ** 2 + (gs * mu_b * x) ** 2)

def g_lin(x, g, b):
    return 0.5*g*mu_b*x + b


def subtract_background_per_trace(map_data, FG, axis=0, percentile_clip=(2, 98)):
    """
    Subtract a linear background and normalize each trace individually.

    map_data: 2D array
    axis: 0 (columns) or 1 (rows) along which to process traces
    percentile_clip: tuple of percentiles to clip outliers
    norm_type: 'std' (standard deviation) or 'max' (maximum value) normalization
    """

    if axis == 1:
        map_data = map_data.T  # always operate along axis=0 internally

    n_points, n_traces = map_data.shape
    x = np.arange(n_points)
    corrected = np.empty_like(map_data)

    slope_alt = 0

    for i in range(n_traces):
        y = map_data[:, i]

        # Clip outliers
        lower, upper = np.percentile(y, percentile_clip)
        y_corrected = np.clip(y, lower, upper)

        # Estimate the derivative (smoothly)
        #print(y[-11:-1])
        derivative = np.gradient(y_corrected[-40:-5])
        valid = derivative[
            (derivative < np.percentile(derivative, 60)) &
            (derivative > np.percentile(derivative, 5)) #&
            #(derivative < 1e-6)
            ]
        #print(derivative)
        #slope = np.median(valid)
        #print(slope)
        popt, _ = curve_fit(linear_model, FG[-30:-3], y_corrected[-30:-3])
        slope = popt[0]
        #print(slope)

        if np.abs(slope-slope_alt) > 1e-5 and slope_alt != 0:
            slope = slope_alt

        # Build background
        #background = slope * (x - np.min(x))
        background = slope * (FG - np.min(FG))

        slope_alt = slope

        # Subtract background
        y_corrected = y - background

        min_bin = np.min(y_corrected)
        max_bin = np.max(y_corrected)

        y_corrected = (y_corrected - min_bin) / (max_bin - min_bin)

        corrected[:, i] = y_corrected
        #print(y_corrected)

    if axis == 1:
        corrected = corrected.T

    return corrected

def linecut_400mT():
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "linecut_400mT")
    file_names = find_hdf5_files(file_dir)

    FG14, Bx, DemodR = load_data(file_names, current_dir)
    #DemodR[0] = np.flip(DemodR[0], axis=1)

    DemodR[0] = subtract_background_per_trace(DemodR[0], FG14[0][:, 0], axis=0)

    map = DemodR[0]

    peaks_found = []
    diff = []
    Bpara_peak = []

    for i, row in enumerate(map.T):
        if i%10 == 0:
            peaks, properties = find_peaks((1-row), width=10, height=0.1, prominence=0.05)
            if len(peaks) == 2:
                diff.append(np.abs(FG14[0][peaks[0], i] - FG14[0][peaks[1], i]))
                Bpara_peak.append(Bx[0][peaks[0], i])
                for peak in peaks:
                    if FG14[0][peak, i] < 5.18687:
                        peaks_found.append((Bx[0][peak, i], FG14[0][peak, i]))

    dFG = FG14[0][1, 0] - FG14[0][0, 0]
    print(dFG)
    err = [600]*len(diff)
    leverarmFG12 = 0.08488 #eV/V
    leverarmFG14 = leverarmFG12*(5.196/5.157)
    a = -0.951128
    delE = (np.array(diff))*leverarmFG14*np.sqrt(1+(a)**2)*1e6 #mueV
    err = (np.array(err))*leverarmFG14 * np.sqrt(1 + (a) ** 2)
    #print(delE)

    popt, pcov = curve_fit(del_E, Bpara_peak[6:-1], delE[6:-1])#, p0=[2, 100], sigma=err[6:])
    print(f'g-Factor: {popt[0]}; delta: {popt[1]}')
    sigma = np.sqrt(np.diag(pcov))
    print(pcov)
    print(sigma)

    Bx_full = np.linspace(0, 2.3, 1000)
    fig = plt.figure(figsize=(12, 6))
    im = plt.pcolormesh(Bx[0], FG14[0], map, cmap='viridis_r')
    for i, peak in enumerate(peaks_found):
        plt.scatter(*peak)
    #plt.ylim(np.max(FG14[0]), np.min(FG14[0]))
    #plt.xlim(0, 2)
    plt.ylabel('$FG_{14}$')
    plt.xlabel('$B_{\parallel}(T)$')
    cbar = fig.colorbar(im, location='top', shrink=0.33, anchor=(1, 0))
    cbar.set_label('$R_{dem} (a.u.)$', loc='left')
    plt.show()

    test = [0.0017]

    plt.figure(figsize=(12, 6))
    for i, peak in enumerate(delE[6:]):
        plt.scatter(Bpara_peak[i+6], peak)
    #plt.ylim(0, 0.002)
    plt.plot(Bx_full, del_E(Bx_full, *popt))
    #plt.xlim(0, 2.3)
    plt.show()

    np.save(os.path.join(file_dir, 'linecut_400mT_map.npy'), map)
    np.save(os.path.join(file_dir, 'linecut_400mT_Bpara.npy'), Bx[0])
    np.save(os.path.join(file_dir, 'linecut_400mT_FG14.npy'), FG14[0])

    np.save(os.path.join(file_dir, 'linecut_400mT_delE.npy'), delE[6:])
    np.save(os.path.join(file_dir, 'linecut_400mT_Bpara_peaks.npy'), Bpara_peak[6:])
    np.save(os.path.join(file_dir, 'linecut_400mT_popt.npy'), popt)
    np.save(os.path.join(file_dir, 'linecut_400mT_err.npy'), err[6:])

    return delE



def linecut_500mT():
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "in_plane_linecuts")
    file_names = find_hdf5_files(file_dir)

    FG14, Bx, DemodR = load_data(file_names, current_dir)
    DemodR[1] = np.flip(DemodR[1], axis=1)

    # substract background
    #DemodR[1] = correct_median_diff(DemodR[1])
    #DemodR[0] = correct_median_diff(DemodR[0])
    #DemodR[2] = correct_median_diff(DemodR[2][:, :-1])

    DemodR[1] = np.array([trace - np.mean(trace) for trace in DemodR[1].T]).T
    DemodR[0] = np.array([trace - np.mean(trace) for trace in DemodR[0].T]).T
    DemodR[2] = np.array([trace - np.mean(trace) for trace in DemodR[2][:, :-1].T]).T

    #DemodR[0] = subtract_background_per_trace(DemodR[0], axis=0)
    #DemodR[1] = subtract_background_per_trace(DemodR[1], axis=0)
    #DemodR[2] = subtract_background_per_trace(DemodR[2][:, :-1], axis=0)

    for i, map in enumerate(DemodR):
        DemodR[i] = gaussian_filter1d(map, 1, axis=0)

    FG14 = [map[:, 0] for map in FG14]

    # stitch maps to one
    start_second = np.argmin(np.abs(np.flip(Bx[1])-0.64))
    end_second = np.argmin(np.abs(np.flip(Bx[1])-0.734))
    end_third = np.argmin(np.abs(np.flip(Bx[1])-0.907))
    second_jump_map = DemodR[1][:, start_second:end_second]
    third_jump_map = DemodR[1][:, end_second:end_third]
    fourth_jump_map = DemodR[1][:, end_third:]
    map = stitch_maps(DemodR[0], DemodR[2], (5.18988, 5.18988), FG14)
    map = stitch_maps(map, second_jump_map, (5.1899, 5.19052), FG14)
    map = stitch_maps(map, third_jump_map, (5.18995, 5.18992), FG14)
    map = stitch_maps(map, fourth_jump_map, (5.18994, 5.18979), FG14)

    map = map[55:-10]
    FG14[0] = FG14[0][55:-10]

    Bx_full = np.linspace(0, 1.5, map.shape[1])
    fig = plt.figure(figsize=(12, 6))
    im = plt.pcolormesh(Bx_full, FG14[0],  map, cmap='viridis_r', vmin=0.05, vmax=0.95)
    plt.ylim(5.1928, 5.189)
    plt.ylabel('$FG_{14}$')
    plt.xlabel('$B_{ \parallel}(T)$')
    cbar = fig.colorbar(im, location='top', shrink=0.33, anchor=(1,0))
    cbar.set_label('$R_{dem} (a.u.)$', loc='left')
    plt.show()

    #p_low, p_high = np.percentile(map, (0.01, 99.99))
    #map = np.clip(map, -2, 2)

    # find peaks
    Bx_full = np.linspace(0, 1.5, map.shape[1])
    ny_peaks = []
    shoulder1_peaks = []
    shoulder2_peaks = []
    upper_peaks = []
    lower_peaks = []

    for i, row in enumerate(map.T):
        peaks, properties = find_peaks(-row, height=0.4e-5, width=22)
        for peak in peaks:
            if peak > 185 and peak < 215:
                ny_peaks.append((i, peak))
        peaks, properties = find_peaks(-row, height=-0.3e-5, width=8)
        for peak in peaks:
            if peak > 160 and peak < 190:
                shoulder1_peaks.append((i, peak))
        peaks, properties = find_peaks(-row, height=-0.3e-5, width=8)
        for peak in peaks:
            if peak > 215 and peak < 240:
                shoulder2_peaks.append((i, peak))
        peaks, properties = find_peaks(-row, height=-1e-5, width=13)
        for peak in peaks:
            if FG14[0][peak] > f(Bx_full[i], 0.00035, 5.1915) and FG14[0][peak] < f(Bx_full[i], 0.00035, 5.192):
                lower_peaks.append((i, peak))
        peaks, properties = find_peaks(-row, height=-0.75e-5, width=12)
        for peak in peaks:
            if FG14[0][peak] > f(Bx_full[i], -0.00035, 5.19055) and FG14[0][peak] < f(Bx_full[i], -0.00035, 5.1908):
                #print(f(Bx_full[i], -0.00035, 5.19095), FG14[0][peak], peak)
                upper_peaks.append((i, peak))

    # plotting
    ny_rows, ny_cols = zip(*ny_peaks)
    upper_rows, upper_cols = zip(*upper_peaks)
    lower_rows, lower_cols = zip(*lower_peaks)
    shoulder1_rows, shoulder1_cols = zip(*shoulder1_peaks)
    shoulder2_rows, shoulder2_cols = zip(*shoulder2_peaks)

    # calculate distances
    ny_upper_dist = calc_peak_distance_tuple(ny_peaks, upper_peaks, y_threshold=20)
    ny_lower_dist = calc_peak_distance_tuple(ny_peaks, lower_peaks, y_threshold=20)
    ny_shoulder1_dist = calc_peak_distance_tuple(ny_peaks, shoulder1_peaks, y_threshold=20)
    ny_shoulder2_dist = calc_peak_distance_tuple(ny_peaks, shoulder2_peaks, y_threshold=20)

    dFG = FG14[0][1] - FG14[0][0]

    ny_upper_rows, ny_upper_cols = zip(*ny_upper_dist)
    ny_lower_rows, ny_lower_cols = zip(*ny_lower_dist)
    ny_shoulder1_rows, ny_shoulder1_cols = zip(*ny_shoulder1_dist)
    ny_shoulder2_rows, ny_shoulder2_cols = zip(*ny_shoulder2_dist)

    leverarmFG12 = 0.08488 #eV/V
    leverarmFG14 = leverarmFG12*(5.196/5.157)
    a = -1.73301
    ny_lower_cols = (np.array(ny_lower_cols)*dFG)*leverarmFG14*np.sqrt(1+(a)**2)*10**5 #10mueV
    ny_upper_cols = (np.array(ny_upper_cols)*dFG)*leverarmFG14*np.sqrt(1+(a)**2)*10**5 #10mueV
    ny_shoulder1_cols = (np.array(ny_shoulder1_cols) * dFG) * leverarmFG14 * np.sqrt(1 + (a) ** 2) * 10 ** 5  # 10mueV
    ny_shoulder2_cols = (np.array(ny_shoulder2_cols) * dFG) * leverarmFG14 * np.sqrt(1 + (a) ** 2) * 10 ** 5  # 10mueV

    # fitting
    popt_upper, pcov_upper = curve_fit(del_E, Bx_full[list(ny_upper_rows)], ny_upper_cols)
    popt_lower, pcov_lower = curve_fit(del_E, Bx_full[list(ny_lower_rows)], ny_lower_cols)

    print(f'g-Factor upper: {popt_upper[0]}; b upper: {popt_upper[1]}') #*10**-5
    print(f'g-Factor lower: {popt_lower[0]}; b lower: {popt_lower[1]}')

    print(popt_upper)

    plt.figure(figsize=(12, 8))
    for i, trace in enumerate(map.T):
        if i<100:
            plt.plot(-trace, alpha=0.5)
            x = np.linspace(185, 225, 40)
            #popt_lorentzian, pcov_lorentzian = curve_fit(lorentzian, x, -trace[185:225])
            #plt.plot(x, lorentzian(x, *popt_lorentzian))


    plt.scatter(list(lower_cols), -map[list(lower_cols), list(lower_rows)], facecolors='none',
                edgecolors='magenta')
    plt.scatter(list(ny_cols), -map[list(ny_cols), list(ny_rows)], facecolors='none',
                edgecolors='red')
    plt.scatter(list(shoulder1_cols), -map[list(shoulder1_cols), list(shoulder1_rows)], facecolors='none',
                edgecolors='yellow')
    plt.scatter(list(shoulder2_cols), -map[list(shoulder2_cols), list(shoulder2_rows)], facecolors='none',
                edgecolors='yellow')

    plt.axvline(215)
    plt.axvline(245)

    plt.ylim(-2*10**-5, 2*10**-5)


    fig = plt.figure(figsize=(12, 6))
    im = plt.pcolormesh(Bx_full, FG14[0],  map, cmap='viridis')#, vmin=-7.5e-6, vmax=7.5e-6)
    #plt.scatter(Bx_full[list(ny_rows)], FG14[0][list(ny_cols)], facecolors='none', edgecolors='red', alpha=0.5)
    #plt.scatter(Bx_full[list(upper_rows)], FG14[0][list(upper_cols)], facecolors='none', edgecolors='orange', alpha=0.5)
    #plt.scatter(Bx_full[list(lower_rows)], FG14[0][list(lower_cols)], facecolors='none', edgecolors='magenta', alpha=0.5)
    #plt.scatter(Bx_full[list(shoulder1_rows)], FG14[0][list(shoulder1_cols)], facecolors='none', edgecolors='yellow',
    #            alpha=0.5)
    #plt.scatter(Bx_full[list(shoulder2_rows)], FG14[0][list(shoulder2_cols)], facecolors='none', edgecolors='yellow',
    #            alpha=0.5)
    #plt.axhline(FG14[0][175], color='red')
    #plt.axhline(FG14[0][225], color='red')
    #plt.plot(Bx_full, f(Bx_full, 0.00035, 5.1914), color='orange')
    #plt.plot(Bx_full, f(Bx_full, 0.00035, 5.1921), color='orange')
    #plt.plot(Bx_full, f(Bx_full, -0.00035, 5.1905), color='orange')
    #plt.plot(Bx_full, f(Bx_full, -0.00035, 5.1911), color='orange')
    plt.ylim(5.1928, 5.189)
    plt.ylabel('$FG_{14}$')
    plt.xlabel('$B_{ \parallel}(T)$')
    cbar = fig.colorbar(im, location='top', shrink=0.33, anchor=(1,0))
    cbar.set_label('$R_{dem} (a.u.)$', loc='left')
    plt.show()


    plt.figure(figsize=(12, 8))
    plt.scatter(Bx_full[list(ny_upper_rows)[::3]], ny_upper_cols[::3], color='orangered', marker='.')
    plt.plot(Bx_full, del_E2(Bx_full, *popt_upper), color='orangered', linestyle='--')
    plt.ylim(5, 12)
    bbox = dict(boxstyle='round', fc='none', ec='black')
    plt.text(1.0, 6.5,
             r'$g_{S}$: ' + '{:.2}'.format(popt_upper[0]) + '\n'  +  '$\Delta$: ' + '{} $\mu eV$'.format(int(popt_upper[1]*10)),
             bbox=bbox)
    plt.xlabel('$B_{ \parallel}(T)$')
    plt.ylabel('$\Delta E$ $(10\mu eV)$')

    plt.figure(figsize=(12, 8))
    plt.scatter(Bx_full[list(ny_lower_rows)[::3]], ny_lower_cols[::3], color='mediumblue', marker='.')
    plt.plot(Bx_full, del_E2(Bx_full, *popt_lower), color='mediumblue', linestyle='--')
    plt.ylim(5, 12)
    bbox = dict(boxstyle='round', fc='none', ec='black')
    plt.text(1.0, 6.5,
             r'$g_{S}$: ' + '{:.2}'.format(popt_lower[0]) + '\n'  +  '$\Delta$: ' + '{} $\mu eV$'.format(int(popt_lower[1]*10)),
             bbox=bbox)
    plt.xlabel('$B_{ \parallel}(T)$')
    plt.ylabel('$\Delta E$ $(10\mu eV)$')

    plt.figure(figsize=(12, 8))
    plt.scatter(Bx_full[list(ny_shoulder1_rows)[1::3]], ny_shoulder1_cols[1::3],
                facecolors='none', edgecolors='mediumblue')
    plt.scatter(Bx_full[list(ny_shoulder2_rows)[::3]], ny_shoulder2_cols[::3],
                facecolors='none', edgecolors='orangered')
    #plt.ylim(5, 12)
    plt.xlabel('$B_{ \parallel}(T)$')
    plt.ylabel('$\Delta E$ $(10\mu eV)$')

    """
    plt.figure(figsize=(12, 8))
    for y in map.T[:20]:
        plt.plot(FG14[0][55:-10], y, c='red', alpha=0.3)
    """
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f'linecut.svg'))

    return popt_lower, np.sqrt(np.diag(pcov_lower))

B_perp = 1

def linecut_1T():
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "linecuts_1T")
    file_names = find_hdf5_files(file_dir)

    FG14, Bx, DemodR = load_data(file_names, current_dir)
    #DemodR[0] = np.flip(DemodR[0], axis=1)
    #DemodR[0] = np.flip(DemodR[0], axis=0)

    # substract background
    #DemodR[0] = correct_median_diff(DemodR[0])
    #DemodR[0] = np.array([trace - np.mean(trace) for trace in DemodR[0]])
    DemodR[0] = subtract_background_per_trace(DemodR[0], FG14[0][:, 0], axis=0)
    #DemodR[0] = np.array([trace - np.mean(trace) for trace in DemodR[0].T]).T
    #DemodR[0] = np.array([trace - np.mean(trace) for trace in DemodR[0].T]).T
    #DemodR[2] = np.array([trace - np.mean(trace) for trace in DemodR[2][:, :-1].T]).T

    #for i, map in enumerate(DemodR):
    #p_low, p_high = np.percentile(DemodR[0], (2, 98))
    #DemodR[0] = np.clip(DemodR[0], p_low, p_high)
    #DemodR[0] = gaussian_filter1d(DemodR[0], 1, axis=0)
    #DemodR[0] = normalize(DemodR[0])

    Bx_stitch = Bx[0][0, :]
    #print(Bx_stitch)
    FG_stitch = FG14[0][:, 0]
    #print(FG_stitch)

    # stitch maps to one
    start_second = np.argmin(np.abs(Bx_stitch-0.7845))
    start_third = np.argmin(np.abs(Bx_stitch-1.109))
    start_fourth = np.argmin(np.abs(Bx_stitch-1.188))
    start_fifth = np.argmin(np.abs(Bx_stitch-1.219))
    start_sixth = np.argmin(np.abs(Bx_stitch-1.239))
    start_seventh = np.argmin(np.abs(Bx_stitch-1.247))
    first_jump_map = DemodR[0][:, :start_second]
    second_jump_map = DemodR[0][:, start_second:start_third]
    third_jump_map = DemodR[0][:, start_third:start_fourth]
    fourth_jump_map = DemodR[0][:, start_fourth:start_fifth]
    fifth_jump_map = DemodR[0][:, start_fifth:start_sixth]
    sixth_jump_map = DemodR[0][:, start_sixth:start_seventh]
    seventh_jump_map = DemodR[0][:, start_seventh:]
    map = stitch_maps(first_jump_map, second_jump_map, (5.1880, 5.1879), FG_stitch)
    #map = stitch_maps(map, second_jump_map, (5.1888, 5.1878), FG14)
    map = stitch_maps(map, third_jump_map, (5.1880, 5.1878), FG_stitch)
    map = stitch_maps(map, fourth_jump_map, (5.1880, 5.1879), FG_stitch)
    map = stitch_maps(map, fifth_jump_map, (5.1880, 5.1878), FG_stitch)
    map = stitch_maps(map, sixth_jump_map, (5.1880, 5.1879), FG_stitch)
    map = stitch_maps(map, seventh_jump_map, (5.1880, 5.1878), FG_stitch)

    #map = map[:-10]
    #FG14 = FG14[0][:-10]

    map = DemodR[0]

    peaks_found = []
    diff = []
    Bpara_peak = []

    for i, row in enumerate(map.T):
        if i % 8 == 0:
            peaks, properties = find_peaks((1 - row), width=10, height=0.1, prominence=0.01)
            if len(peaks) == 2:
                diff.append(np.abs(FG14[0][peaks[0], i] - FG14[0][peaks[1], i]))
                Bpara_peak.append(Bx[0][peaks[0], i])
                for peak in peaks:
                    if FG14[0][peak, i] < 5.189:
                        peaks_found.append((Bx[0][peak, i], FG14[0][peak, i]))

    dFG = FG14[0][1, 0] - FG14[0][0, 0]
    print(dFG)
    err = [600] * len(diff)
    leverarmFG12 = 0.08488  # eV/V
    leverarmFG14 = leverarmFG12 * (5.196 / 5.157)
    a = -0.951128
    delE = (np.array(diff)) * leverarmFG14 * np.sqrt(1 + (a) ** 2) * 1e6  # mueV
    mask = delE<170
    delE = delE[mask]
    Bpara_peak = np.array(Bpara_peak)
    Bpara_peak = Bpara_peak[mask]
    err = (np.array(err)) * leverarmFG14 * np.sqrt(1 + (a) ** 2)
    # print(delE)

    popt, pcov = curve_fit(del_E, Bpara_peak[:], delE[:], p0=[2, 100])#, sigma=err[6:])
    print(f'g-Factor: {popt[0]}; delta: {popt[1]}')
    sigma = np.sqrt(np.diag(pcov))
    print(pcov)
    print(sigma)

    # find peaks
    Bx_full = np.linspace(0, 2, map.shape[1])
    fig = plt.figure(figsize=(12, 6))
    im = plt.pcolormesh(Bx[0], FG14[0], map, cmap='viridis_r')
    for i, peak in enumerate(peaks_found):
        plt.scatter(*peak)
    #plt.ylim(5.1916, 5.186)
    plt.ylabel('$FG_{14}$')
    plt.xlabel('$B_{ \parallel}(T)$')
    cbar = fig.colorbar(im, location='top', shrink=0.33, anchor=(1,0))
    cbar.set_label('$R_{dem} (a.u.)$', loc='left')

    plt.figure(figsize=(12, 6))
    for i, peak in enumerate(delE[:]):
        plt.scatter(Bpara_peak[i], peak)
    #plt.ylim(0, 0.002)
    plt.plot(Bx_full, del_E(Bx_full, *popt))
    #plt.xlim(0, 2.3)
    plt.show()

    np.save(os.path.join(file_dir, 'linecut_1T_map.npy'), map)
    np.save(os.path.join(file_dir, 'linecut_1T_Bpara.npy'), Bx[0])
    np.save(os.path.join(file_dir, 'linecut_1T_FG14.npy'), FG14[0])

    np.save(os.path.join(file_dir, 'linecut_1T_delE.npy'), delE)
    np.save(os.path.join(file_dir, 'linecut_1T_Bpara_peaks.npy'), Bpara_peak)
    np.save(os.path.join(file_dir, 'linecut_1T_popt.npy'), popt)
    np.save(os.path.join(file_dir, 'linecut_1T_err.npy'), err)



    # ny_peaks = []
    # upper_peaks = []
    # lower_peaks = []
    # for i, row in enumerate(map.T):
    #     peaks, properties = find_peaks(-row, height=2e-5, width=16)
    #     #peaks = find_peaks_cwt(-row, 10)
    #     for peak in peaks:
    #         if peak > 175 and peak < 225:
    #             ny_peaks.append((i, peak))
    #     #peaks, properties = find_peaks(-row, height=10e-5, width=20)
    #     '''
    #     for peak in peaks:
    #         if FG14[peak] > f(Bx_full[i], 0.00035, 5.19155) and FG14[peak] < f(Bx_full[i], 0.00035, 5.192):
    #             lower_peaks.append((i, peak))
    #     '''
    #     peaks, properties = find_peaks(-row, height=-5.5e-5, width=11)
    #     #peaks = find_peaks_cwt(row, 5)
    #     for peak in peaks:
    #         if FG14[peak] > f(Bx_full[i], 0.00035, 5.1902) and FG14[peak] < f(Bx_full[i], 0.00035, 5.191):
    #             #print(f(Bx_full[i], -0.00035, 5.19095), FG14[0][peak], peak)
    #             lower_peaks.append((i, peak))
    #
    # # plotting
    # ny_rows, ny_cols = zip(*ny_peaks)
    # #upper_rows, upper_cols = zip(*upper_peaks)
    # lower_rows, lower_cols = zip(*lower_peaks)
    #
    # # calculate distances
    # #ny_upper_dist = calc_peak_distance_tuple(ny_peaks, upper_peaks, y_threshold=100)
    # ny_lower_dist = calc_peak_distance_tuple(ny_peaks, lower_peaks, y_threshold=100)
    #
    # dFG = FG14[1] - FG14[0]
    #
    # #ny_upper_rows, ny_upper_cols = zip(*ny_upper_dist)
    # ny_lower_rows, ny_lower_cols = zip(*ny_lower_dist)
    #
    # leverarmFG12 = 0.08488 #eV/V
    # leverarmFG14 = leverarmFG12*(5.196/5.157)
    # a = -1.95881
    # ny_lower_cols = (np.array(ny_lower_cols)*dFG)*leverarmFG14*np.sqrt(1+(a)**2)*10**5 #10mueV
    # #ny_upper_cols = (np.array(ny_upper_cols)*dFG)*leverarmFG14*np.sqrt(1+(1/a)**2)*10**5 #10mueV
    #
    # # fitting
    # #popt_upper, pcov_upper = curve_fit(g, Bx_full[list(ny_upper_rows)], ny_upper_cols)
    # popt_lower, pcov_lower = curve_fit(del_E2, Bx_full[list(ny_lower_rows)], ny_lower_cols)
    #
    # print(np.sqrt(np.diag(pcov_lower)))
    #
    # #print(f'g-Factor upper: {popt_upper[0]}; b upper: {popt_upper[1]}') #*10**-5
    # print(f'g-Factor lower: {popt_lower[0]}; b lower: {popt_lower[1]}')
    #
    # plt.figure(figsize=(12, 8))
    # for i, trace in enumerate(map.T):
    #     plt.plot(-trace, color='gray', alpha=0.5)
    #     x = np.linspace(185, 225, 40)
    #     #popt_lorentzian, pcov_lorentzian = curve_fit(lorentzian, x, -trace[185:225])
    #     #plt.plot(x, lorentzian(x, *popt_lorentzian))
    #
    #
    # plt.scatter(list(lower_cols), -map[list(lower_cols), list(lower_rows)], facecolors='none', edgecolors='magenta')
    # plt.scatter(list(ny_cols), -map[list(ny_cols), list(ny_rows)], facecolors='none', edgecolors='red')
    # #plt.scatter(list(lower_cols), -map[list(lower_cols), list(lower_rows)], facecolors='none', edgecolors='magenta')
    #
    # plt.axvline(175)
    # plt.axvline(225)
    #
    #
    # fig = plt.figure(figsize=(12, 6))
    # im = plt.pcolormesh(Bx_full, FG14,  map, cmap='viridis_r', vmin=-4.5*10**-5, vmax=4.5*10**-5)
    # #plt.scatter(Bx_full[list(ny_rows)], FG14[list(ny_cols)], facecolors='none', edgecolors='red', alpha=0.5)
    # #plt.scatter(Bx_full[list(upper_rows)], FG14[0][list(upper_cols)], facecolors='none', edgecolors='orange', alpha=0.5)
    # #plt.scatter(Bx_full[list(lower_rows)], FG14[list(lower_cols)], facecolors='none', edgecolors='magenta', alpha=0.5)
    # #plt.axhline(FG14[175], color='red')
    # #plt.axhline(FG14[225], color='red')
    # #plt.plot(Bx_full, f(Bx_full, 0.00035, 5.19155), color='orange')
    # #plt.plot(Bx_full, f(Bx_full, 0.00035, 5.192), color='orange')
    # #plt.plot(Bx_full, f(Bx_full, 0.00035, 5.1903), color='orange')
    # #plt.plot(Bx_full, f(Bx_full, 0.00035, 5.1908), color='orange')
    # plt.ylim(5.1916, 5.186)
    # plt.ylabel('$FG_{14}$')
    # plt.xlabel('$B_{ \parallel}(T)$')
    # cbar = fig.colorbar(im, location='top', shrink=0.33, anchor=(1,0))
    # cbar.set_label('$R_{dem} (a.u.)$', loc='left')
    #
    # '''
    # plt.figure(figsize=(12, 8))
    # plt.scatter(Bx_full[list(ny_upper_rows)[::3]], ny_upper_cols[::3], color='orangered', marker='.')
    # plt.plot(Bx_full, g(Bx_full, *popt_upper), color='orangered', linestyle='--')
    # plt.ylim(5.5, 11)
    # bbox = dict(boxstyle='round', fc='none', ec='black')
    # plt.text(1.0, 6.5, f'g-factor: {(popt_upper[0]).round(2)} \n     $\Delta$     : {int(popt_upper[1]*10)} $\mu eV$', bbox=bbox)
    # plt.xlabel('$B_{ \parallel}(T)$')
    # plt.ylabel('$\Delta E$ $(10\mu eV)$')
    # '''
    #
    # plt.figure(figsize=(12, 8))
    # plt.scatter(Bx_full[list(ny_lower_rows)[::3]], ny_lower_cols[::3], color='mediumblue', marker='.')
    # plt.plot(Bx_full, del_E2(Bx_full, *popt_lower), color='mediumblue', linestyle='--')
    # plt.ylim(7.5, 14)
    # bbox = dict(boxstyle='round', fc='none', ec='black')
    # plt.text(1.25, 8.5,
    #          r'$g_{S}$: ' + '{:.2}'.format(popt_lower[0]) + '\n'  +  '$\Delta$: ' + '{} $\mu eV$'.format(int(popt_lower[1]*10)),
    #          bbox=bbox)
    # plt.xlabel('$B_{ \parallel}(T)$')
    # plt.ylabel('$\Delta E$ $(10\mu eV)$')
    #
    # """
    # plt.figure(figsize=(12, 8))
    # for y in map.T[:20]:
    #     plt.plot(FG14[0][55:-10], y, c='red', alpha=0.3)
    # """
    #
    # plt.show()

    #return popt_lower, np.sqrt(np.diag(pcov_lower))

def linecut_0T():
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "linecuts_0T")
    file_names = find_hdf5_files(file_dir)

    FG14, Bx, DemodR = load_data(file_names, current_dir)
    #DemodR[0] = np.flip(DemodR[0], axis=1)
    #DemodR[1] = np.flip(DemodR[1], axis=1)

    for i, map in enumerate(DemodR):
        DemodR[i] = np.array([trace - np.mean(trace) for trace in map.T]).T
        #DemodR[i] = np.array([detrend(trace) for trace in map])
        #DemodR[i] = gaussian_filter1d(map, 1, axis=0)

    end_first = np.argmin(np.abs(Bx[0]-1.0))
    end_second = np.argmin(np.abs(Bx[1]-1.3))
    first_map = DemodR[0][:, :end_first]
    second_map = DemodR[1][:, :end_second]

    map = stitch_maps(first_map, second_map, (5.18344, 5.18344), FG14)

    Bx_full = np.linspace(-0.25, 1.3, map.shape[1])

    edge = []

    for i, row in enumerate(map.T):
        peaks, properties = find_peaks(row)#, height=1e-5, width=16)
        edge.append(peaks)

    edge_rows, edge_cols = zip(*edge)

    plt.figure(figsize=(12, 8))
    for i, trace in enumerate(map.T):
        plt.plot(trace, color='gray', alpha=0.5)
    #plt.scatter(list(edge_cols), -map[list(edge_cols), list(edge_rows)], facecolors='none', edgecolors='red')

    plt.figure(figsize=(12, 6))
    im = plt.pcolormesh(Bx_full, FG14[0],  map, cmap='seismic')#, vmin=-7.5e-4, vmax=7.5e-4)
    plt.show()


def main():
    #linecut_0T()
    #popt_500mT, pcov_500mT = linecut_500mT()
    linecut_1T()
    #linecut_400mT()

    '''
    x = np.array([0.5, 1])
    y = np.array([popt_500mT[1], popt_1T[1]])
    y_err = np.array([pcov_500mT[1], pcov_1T[1]])

    print(y, y_err)

    def delta(x, delta_SO, g_s):
        return delta_SO - g_s*mu_b*x

    popt, pcov = curve_fit(delta, xdata=x, ydata=y, sigma=y_err)

    B_perp = np.linspace(0, 1.1, 100)

    plt.figure(figsize=(12, 8))
    plt.errorbar(x, y, yerr=y_err, linestyle='none', marker='o', markersize=8, mfc='w')
    plt.plot(B_perp, delta(B_perp, *popt), color='gray', linestyle='--')
    plt.xlim(0, 1.1)
    bbox = dict(boxstyle='round', fc='none', ec='black')
    plt.text(0.7, 12.5,
             r'$g_{S}$: ' + '{:.2f}'.format(popt[1]) + '\n'  +  '$\Delta_{SO}$: ' + '{} $\mu eV$'.format(int(popt[0]*10)),
             bbox=bbox)
    plt.xlabel(r'$B_{\bot}$')
    plt.ylabel(r'$\Delta_{\nu}$ $(10\mu eV)$')
    plt.show()
    '''


if __name__ == "__main__":
    main()
