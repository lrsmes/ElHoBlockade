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

def y_tuple(tread, tini, fac, FG14, diff, off, pulse_dir):
    FG14_temp = np.flip(FG14[0])
    step = np.abs(FG14_temp[1]-FG14_temp[0])
    FG14_val = off+pulse_dir*fac*((tread-tini)/(tread+tini))
    find_val = np.abs(FG14_temp-FG14_val)
    lower = np.abs(FG14_temp-FG14_val).argmin()
    if lower == 0:
        lower = np.round(-(np.min(find_val) // step))
    upper = lower + diff
    if tread == 20729.865928436448:
        print(FG14_temp[70], FG14_temp[150])
    return (lower, upper)

# Calculate the offset on the FG14 axis for a give tread time
def get_offset(tread, tini, fac, FG14, y_cut, pulse_dir):
    FG14_temp = np.flip(FG14[0])
    FG14_val = FG14_temp[y_cut]
    FG14_off = FG14_val - pulse_dir*fac*((tread-tini)/(tread+tini))
    return FG14_off
def load_data(file_name, group, dataset, channels, all_maps_pos, all_maps_neg):
    channel_data = helper.read_file(file_name, group, dataset, channels, information=False)

    FG12_raw = channel_data['FG_12']
    FG14_raw = channel_data['FG_14']

    T_read = channel_data['Puls_im_Dreieck - Tread']
    T_read_unique = np.unique(T_read)

    pulse_direction = channel_data['triple_pulse_direction']

    demR_raw = channel_data['UHFLI - Demod1R']

    maps_pos = all_maps_pos
    maps_neg = all_maps_neg

    for tread in T_read_unique:
        demR, FG12, FG14 = get_single_map(tread, FG12_raw, FG14_raw, T_read, demR_raw, pulse_direction, 1)
        demR_neg, FG12_neg, FG14_neg = get_single_map(tread, FG12_raw, FG14_raw, T_read, demR_raw, pulse_direction, -1)
        maps_pos.append((demR, FG12, FG14, tread))
        maps_neg.append((demR_neg, FG12_neg, FG14_neg, tread))
        print(tread)

    return maps_pos, maps_neg

def load_data_wo_dir(file_name, group, dataset, channels, all_maps):
    channel_data = helper.read_file(file_name, group, dataset, channels, information=False)

    FG12_raw = channel_data['FG_12']
    FG14_raw = channel_data['FG_14']

    T_read = channel_data['Puls_im_Dreieck - Tread']
    T_read_unique = np.unique(T_read)

    #pulse_direction = channel_data['triple_pulse_direction']

    demR_raw = channel_data['UHFLI - Demod1R']

    maps = all_maps
    #maps_neg = all_maps_neg

    for tread in T_read_unique:
        demR, FG12, FG14 = get_single_map(tread, FG12_raw, FG14_raw, T_read, demR_raw)
        #demR_neg, FG12_neg, FG14_neg = get_single_map(tread, FG12_raw, FG14_raw, T_read, demR_raw, pulse_direction, -1)
        maps.append((demR, FG12, FG14, tread))
        #maps_neg.append((demR_neg, FG12_neg, FG14_neg, tread))
        print(tread)

    return maps

def process_data(map, off_y, off_x, file_dir, pulse_dir, neg=True, h_slope=0.42, v_slope = 0.2, diff= [0, 0, 0]):
    dem_R, FG_12, FG_14, T_read = map
    end_x = dem_R.shape[0]
    end_y = dem_R.shape[1]
    if neg:
        dem_R = substract_linear_BG(dem_R, FG_12, FG_14, xl=end_x - (end_x//3), xr=end_x - 1, yt=end_y - (end_y//4), yb=end_y - 1, #end_y - (end_y//4)
                                    subtract=True)
    else:
        dem_R = substract_linear_BG(dem_R, FG_12, FG_14, xl=1, xr=(end_x//3), yt=1,
                                    yb=(end_y//4), #(end_y//4)
                                    subtract=True)

    current_dir = os.getcwd()

    if pulse_dir == 1:
        R, R_err = define_resonances_tp(dem_R, FG_12, FG_14, file_dir, T_read, pulse_dir,
                                             h_res_slope=h_slope,
                                             h_res_offset_t=off_y[0],
                                             h_res_offset_b=off_y[1],

                                             v_res_slope=v_slope,
                                             v_res_offset_l=off_x[0],
                                             v_res_offset_r=off_x[1],
                                             )
        return R, R_err
    else:
        R_ny, R_err_ny, R_alpha, R_error_alpha = define_resonances_tp(dem_R, FG_12, FG_14, file_dir, T_read, pulse_dir,
                                        h_res_slope=h_slope,
                                        h_res_offset_t=off_y[0],
                                        h_res_offset_b=off_y[1],

                                        v_res_slope=v_slope,
                                        v_res_offset_l=off_x[0],
                                        v_res_offset_r=off_x[1],

                                        first_ny_diff=diff[0],
                                        second_ny_diff=diff[1],
                                        alpha_diff=diff[2]
                                        )
        return R_ny, R_err_ny, R_alpha, R_error_alpha

def regime_tl_larger_ti_1T():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "1T Regime tL_larger_ti")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_pos = []
    ratios_err_pos = []
    ratios_neg = []
    ratios_err_neg = []
    all_maps_pos = []
    all_maps_neg = []

    for file in file_names:
        # Load the data in the given file
        all_maps_pos, all_maps_neg = load_data(file, group, dataset, channels, all_maps_pos, all_maps_neg)


    # 0
    r, err = process_data(all_maps_neg[0], (18, 52), (20, 50), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[0], (12, 46), (20, 50), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    # 1
    r, err = process_data(all_maps_neg[1], (-16, 18), (25, 55), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[1], (45, 80), (15, 45), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    # 2
    r, err = process_data(all_maps_neg[2], (-10, 25), (25, 55), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[2], (38, 72), (15, 45), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    # 3
    r, err = process_data(all_maps_neg[3], (-5, 30), (25, 55), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[3], (36, 70), (15, 48), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    # 4
    r, err = process_data(all_maps_neg[4], (-2, 32), (25, 55), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[4], (32, 66), (15, 48), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    # 5
    r, err = process_data(all_maps_neg[5], (0, 36), (25, 50), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[5], (30, 64), (20, 45), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)
    
    # 6
    r, err = process_data(all_maps_neg[6], (4, 40), (20, 55), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[6], (26, 60), (20, 50), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)
    
    # 7
    r, err = process_data(all_maps_neg[7], (8, 43), (20, 53), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[7], (24, 58), (20, 50), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)
    
    # 8
    r, err = process_data(all_maps_neg[8], (10, 44), (20, 50), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[8], (22, 56), (20, 50), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)
    
    # 9
    r, err = process_data(all_maps_neg[9], (10, 45), (20, 50), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[9], (20, 54), (20, 50), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)
    
    # 10
    r, err = process_data(all_maps_neg[10], (35, 105), (47, 97), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[10], (20, 88), (40, 105), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    # 11
    r, err = process_data(all_maps_neg[11], (-35, 35), (50, 115), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[11], (85, 160), (35, 100), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    # 12
    r, err = process_data(all_maps_neg[12], (-25, 45), (50, 115), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[12], (75, 150), (35, 100), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)
 
    # 13
    r, err = process_data(all_maps_neg[13], (-18, 52), (50, 115), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[13], (70, 140), (40, 100), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)       

    # 14
    r, err = process_data(all_maps_neg[14], (-10, 60), (50, 110), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[14], (65, 135), (40, 100), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    # 15
    r, err = process_data(all_maps_neg[15], (-5, 65), (50, 110), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[15], (60, 130), (40, 100), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    # 16
    r, err = process_data(all_maps_neg[16], (0, 65), (50, 110), file_dir, -1)
    ratios_neg.append(r)
    ratios_err_neg.append(err)
    r, err = process_data(all_maps_pos[16], (55, 125), (40, 100), file_dir, 1, False)
    ratios_pos.append(r)
    ratios_err_pos.append(err)

    ######################
    #Fitting and Plotting
    ######################

    plt.figure()
    t_read_s = []
    for elem in all_maps_pos:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s)*125*1e-6
    print(t_read_s)
    plt.plot(t_read_s, ratios_pos,
             linestyle='None', marker='.')

    #plt.errorbar(t_read_s,
    #              ratios_pos,
    #              ratios_err_pos,
    #              linestyle='None', marker='.')

    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_pos,
                           xdata=t_read_s,
                           sigma=ratios_err_pos,
                           p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    print(popt, "\n", np.sqrt(np.diag(pcov)))

    plt.plot(t, exponential_model(t, *popt), label=r'$\tau$' + ': {:.2}$\mu$s'.format(popt[0]))
    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.savefig('blockade_int_ratio.svg', dpi=600)
    plt.savefig(os.path.join(file_dir, f'exp_fit_pos.png'))

    plt.figure()
    plt.plot(t_read_s, ratios_neg,
             linestyle='None', marker='.')

    #plt.errorbar(t_read_s,
    #              ratios_neg,
    #              ratios_err_neg,
    #              linestyle='None', marker='.')

    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_neg,
                           xdata=t_read_s,
                           sigma=ratios_err_neg,
                           p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    print(popt, "\n", np.sqrt(np.diag(pcov)))

    plt.plot(t, exponential_model(t, *popt), label=r'$\tau$' + ': {:.2}$\mu$s'.format(popt[0]))
    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.savefig('blockade_int_ratio.svg', dpi=600)
    plt.savefig(os.path.join(file_dir, f'exp_fit_neg.png'))

def regime_ti_larger_tl_1T():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "1T Regime ti_larger_tl")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_transport = []
    ratios_err_transport = []
    ratios_blockade = []
    ratios_err_blockade = []
    all_maps_blockade = []
    all_maps_transport = []

    for file in file_names:
        # Load the data in the given file
        if "blockade" in file:
            all_maps_blockade = load_data_wo_dir(file, group, dataset, channels, all_maps_blockade)
        elif "transport" in file:
            all_maps_transport = load_data_wo_dir(file, group, dataset, channels, all_maps_transport)


    ##########
    #blockade
    ##########

    # 499, 0.625
    r, err = process_data(all_maps_blockade[0], (13, 70), (22, 62), file_dir, 1, True,0.5, 0.12)
    ratios_blockade.append(r)
    ratios_err_blockade.append(err)

    #792, 0.1
    r, err = process_data(all_maps_blockade[1], (8, 65), (22, 62), file_dir, 1, True, 0.5, 0.12)
    ratios_blockade.append(r)
    ratios_err_blockade.append(err)

    #1257, 0.15
    r, err = process_data(all_maps_blockade[2], (3, 58), (22, 62), file_dir, 1, True, 0.5, 0.12)
    ratios_blockade.append(r)
    ratios_err_blockade.append(err)

    #1994, 0.25
    r, err = process_data(all_maps_blockade[3], (-7, 50), (22, 62), file_dir, 1, True, 0.5, 0.12)
    ratios_blockade.append(r)
    ratios_err_blockade.append(err)

    #3162, 0.4
    r, err = process_data(all_maps_blockade[4], (-17, 40), (22, 62), file_dir, 1, True, 0.5, 0.12)
    ratios_blockade.append(r)
    ratios_err_blockade.append(err)

    #5014, 0.62
    r, err = process_data(all_maps_blockade[5], (-27, 30), (22, 62), file_dir, 1, True, 0.5, 0.12)
    ratios_blockade.append(r)
    ratios_err_blockade.append(err)

    #7952, 1.0
    r, err = process_data(all_maps_blockade[6], (-33, 25), (22, 62), file_dir, 1, True, 0.5, 0.12)
    ratios_blockade.append(r)
    ratios_err_blockade.append(err)

    #12611, 1.57
    r, err = process_data(all_maps_blockade[7], (-42, 15), (22, 62), file_dir, 1, True, 0.5, 0.12)
    ratios_blockade.append(r)
    ratios_err_blockade.append(err)

    #20000, 2.5
    r, err = process_data(all_maps_blockade[8], (-47, 10), (22, 62), file_dir, 1, True, 0.5, 0.12)
    ratios_blockade.append(r)
    ratios_err_blockade.append(err)

    ##########
    # transport
    ##########

    # 499
    r, err = process_data(all_maps_transport[0], (-27, 50), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 573
    r, err = process_data(all_maps_transport[1], (-17, 65), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 657
    r, err = process_data(all_maps_transport[2], (-15, 70), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 753
    r, err = process_data(all_maps_transport[3], (-12, 75), (22, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 863
    r, err = process_data(all_maps_transport[4], (-12, 72), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 990
    r, err = process_data(all_maps_transport[5], (-8, 72), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 1134
    r, err = process_data(all_maps_transport[6], (0, 75), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 1301
    r, err = process_data(all_maps_transport[7], (3, 77), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 1491
    r, err = process_data(all_maps_transport[8], (5, 83), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 1709
    r, err = process_data(all_maps_transport[9], (10, 87), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 1960
    r, err = process_data(all_maps_transport[10], (15, 93), (30, 87), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 2247
    r, err = process_data(all_maps_transport[11], (20, 95), (30, 85), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 2576
    r, err = process_data(all_maps_transport[12], (24, 98), (30, 85), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 2953
    r, err = process_data(all_maps_transport[13], (28, 110), (30, 85), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 3385
    r, err = process_data(all_maps_transport[14], (33, 114), (30, 85), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 3881
    r, err = process_data(all_maps_transport[15], (38, 118), (30, 85), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 4449
    r, err = process_data(all_maps_transport[16], (42, 122), (25, 83), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 5101
    r, err = process_data(all_maps_transport[17], (45, 127), (25, 83), file_dir, -1, True, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 5848
    r, err = process_data(all_maps_transport[18], (50, 130), (25, 83), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 6704
    r, err = process_data(all_maps_transport[19], (53, 133), (22, 81), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 7685
    r, err = process_data(all_maps_transport[20], (55, 135), (25, 81), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 8810
    r, err = process_data(all_maps_transport[21], (60, 140), (25, 81), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 10100
    r, err = process_data(all_maps_transport[22], (63, 143), (25, 81), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 11579
    r, err = process_data(all_maps_transport[23], (66, 146), (25, 81), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 13274
    r, err = process_data(all_maps_transport[24], (70, 150), (25, 78), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 15218
    r, err = process_data(all_maps_transport[25], (75, 150), (25, 78), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 17445
    r, err = process_data(all_maps_transport[26], (75, 150), (25, 78), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)

    # 20000
    r, err = process_data(all_maps_transport[27], (78, 155), (25, 78), file_dir, -1, False, 0.5, 0.12)
    ratios_transport.append(r)
    ratios_err_transport.append(err)


    ######################
    # Fitting and Plotting
    ######################

    plt.figure()
    t_read_s = []
    for elem in all_maps_transport:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    plt.plot(t_read_s, ratios_transport,
             linestyle='None', marker='.')

    # plt.errorbar(t_read_s,
    #              ratios_pos,
    #              ratios_err_pos,
    #              linestyle='None', marker='.')

    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_transport,
                           xdata=t_read_s,
                           sigma=ratios_err_transport,
                           p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    print(popt, "\n", np.sqrt(np.diag(pcov)))

    plt.plot(t, exponential_model(t, *popt), label=r'$\tau$' + ': {:.2}$\mu$s'.format(popt[0]))
    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.savefig(os.path.join(file_dir, f'exp_fit_transport.png'))

    plt.figure()
    t_read_s = []
    for elem in all_maps_blockade:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    plt.plot(t_read_s[1:-1], ratios_blockade[1:-1],
             linestyle='None', marker='.')

    # plt.errorbar(t_read_s,
    #              ratios_neg,
    #              ratios_err_neg,
    #              linestyle='None', marker='.')

    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_blockade[1:-1],
                           xdata=t_read_s[1:-1],
                           sigma=ratios_err_blockade[1:-1],
                           p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s[1:-1]), 100)
    print(popt, "\n", np.sqrt(np.diag(pcov)))

    plt.plot(t, exponential_model(t, *popt), label=r'$\tau$' + ': {:.2}$\mu$s'.format(popt[0]))
    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.savefig(os.path.join(file_dir, f'exp_fit_blockade.png'))

def regime_ti_larger_tl_450mT():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "450mT_Regime_tL_larger_ti+")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_transport_ny = []
    ratios_err_transport_ny = []
    ratios_transport_alpha = []
    ratios_err_transport_alpha = []
    ratios_blockade = []
    ratios_err_blockade = []
    all_maps_blockade = []
    all_maps_transport = []


    for file in file_names:
        # Load the data in the given file
        if "blockade" in file:
            all_maps_blockade = load_data_wo_dir(file, group, dataset, channels, all_maps_blockade)
        elif "transport" in file:
            all_maps_transport = load_data_wo_dir(file, group, dataset, channels, all_maps_transport)

    all_maps_blockade = all_maps_blockade[:-21] + all_maps_blockade[-7:]

    ###########
    # Blockade
    ###########
    print('###################### blockade #########################')
    off = get_offset(2597, all_maps_blockade[11][-2], 55, 1)
    for i, map in enumerate(all_maps_blockade):
        if i<21:
            r, err = process_data(map, y_tuple(map[-1], map[-2], 38, off, 1), (65, 110), #+10
                                  file_dir, 1, True, 0.35, 0.18)
        else:
            r, err = process_data(map, y_tuple(map[-1], map[-2], 38, off, 1), (50, 90), #+10 4.8117
                                  file_dir, 1, True, 0.35, 0.18)
        ratios_blockade.append(r)
        ratios_err_blockade.append(err)


    #############
    # Transport
    #############
    print('###################### transport #########################')
    print(len(all_maps_transport))
    off_transport = get_offset(20730, all_maps_transport[11][-2], 80, -1)
    for i, map in enumerate(all_maps_transport[1:]):
        if i<14:
            r_ny, err_ny, r_alpha, error_alpha = process_data(map, y_tuple(map[-1], map[-2], 60, off_transport, -1), (35, 110), #+10 4.821
                                  file_dir, -1, False, 0.35, 0.18, [62, 77, 95])
        else:
            r_ny, err_ny, r_alpha, error_alpha = process_data(map, y_tuple(map[-1], map[-2], 60, off_transport, -1), (35, 110), #+10
                                  file_dir, -1, False, 0.35, 0.18, [58, 75, 90])
        ratios_transport_ny.append(r_ny)
        ratios_err_transport_ny.append(err_ny)
        ratios_transport_alpha.append(r_alpha)
        ratios_err_transport_alpha.append(error_alpha)

    ######################
    # Fitting and Plotting
    ######################

    # blockade
    plt.figure(figsize=(20, 12))
    ls = 'dashed'
    t_read_s = []
    for elem in all_maps_blockade:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    #plt.plot(t_read_s, ratios_blockade[:-3],
    #         linestyle='None', marker='.')

    plt.errorbar(t_read_s, ratios_blockade,
                 ratios_err_blockade,
                 linestyle='None', marker='.',
                 color='mediumvioletred', elinewidth=0.5)


    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_blockade[:-3],
                           xdata=t_read_s[:-3],
                           sigma=ratios_err_blockade[:-3],
                           p0=[1, 0.5, 0.5])


    #popt, pcov = fit_with_derivative(exponential_model, t_read_s, ratios_blockade, p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    print(popt, "\n", pcov)

    plt.plot(t, exponential_model(t, *popt),
             label=r'$\tau_{blockade}$' + ': {:.2f} $\pm$ {:.2f}$\mu$s'.format(popt[0], pcov[0]),
             color='mediumvioletred', linestyle=ls, alpha=0.8)
    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    #plt.savefig(os.path.join(file_dir, f'exp_fit_blockade.png'))

    # transport
    #plt.figure()
    t_read_s = []
    for elem in all_maps_transport[1:]:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    #plt.plot(t_read_s, ratios_transport_ny,
    #         linestyle='None', marker='.')
    #plt.plot(t_read_s, ratios_transport_alpha,
    #         linestyle='None', marker='.')

    plt.errorbar(t_read_s,
                 ratios_transport_ny,
                 ratios_err_transport_ny,
                 linestyle='None', marker='.',
                 color='slateblue', elinewidth=0.5)

    plt.errorbar(t_read_s,
                 ratios_transport_alpha,
                 ratios_err_transport_alpha,
                 linestyle='None', marker='.',
                 color='mediumblue', elinewidth=0.5)


    popt_ny, pcov_ny = curve_fit(exponential_model,
                           ydata=ratios_transport_ny,
                           xdata=t_read_s,
                           sigma=ratios_err_transport_ny,
                           p0=[1, 0.5, 0.5])

    #popt_ny, pcov_ny = fit_with_derivative(exponential_model, t_read_s, ratios_transport_ny, p0=[1, 0.5, 0.5])


    popt_alpha, pcov_alpha = curve_fit(exponential_model,
                           ydata=ratios_transport_alpha,
                           xdata=t_read_s,
                           sigma=ratios_err_transport_alpha,
                           p0=[1, 0.5, 0.5])

    #popt_alpha, pcov_alpha = fit_with_derivative(exponential_model, t_read_s, ratios_transport_alpha, p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    print(popt_ny, "\n", pcov_ny) #np.sqrt(np.diag(pcov_ny)))

    plt.plot(t, exponential_model(t, *popt_ny),
             label=r'$\tau_{first}$' + ': {:.2f} $\pm$ {:.2f}$\mu$s'.format(popt_ny[0], pcov_ny[0]),
             color='slateblue', linestyle=ls, alpha=0.8)
    plt.plot(t, exponential_model(t, *popt_alpha),
             label=r'$\tau_{second}$' + ': {:.2f} $\pm$ {:.2f}$\mu$s'.format(popt_alpha[0], pcov_alpha[0]),
             color='mediumblue', linestyle=ls, alpha=0.8)
    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    #plt.savefig(os.path.join(file_dir, f'exp_fit_transport.png'))

    plt.savefig(os.path.join(file_dir, f'exp_fit_all.png'))

def regime_ti_larger_tl_200mT():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "200mT_Regime_tL_larger_ti")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_transport = []
    ratios_err_transport = []
    ratios_blockade = []
    ratios_err_blockade = []
    all_maps_blockade = []
    all_maps_transport = []

    for file in file_names:
        # Load the data in the given file
        if "blockade" in file:
            all_maps_blockade = load_data_wo_dir(file, group, dataset, channels, all_maps_blockade)
        elif "transport" in file:
            all_maps_transport = load_data_wo_dir(file, group, dataset, channels, all_maps_transport)

    ###########
    # Blockade
    ###########
    for map in all_maps_blockade:
        # 500, 0.625
        r, err = process_data(map, (13, 70), (22, 62), file_dir, 1, True, 0.5, 0.12)
        ratios_blockade.append(r)
        ratios_err_blockade.append(err)

    #############
    # Transport
    #############
    for map in all_maps_transport[6:]:
        # 500, 0.625
        r, err = process_data(map, (20, 30), (22, 62), file_dir, -1, False, 0.5, 0.12)
        ratios_blockade.append(r)
        ratios_err_blockade.append(err)

def both_dir_500mT():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "500mT_both_dir")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_transport = []
    ratios_err_transport = []
    ratios_blockade = []
    ratios_err_blockade = []
    all_maps_blockade = []
    all_maps_transport = []

    for file in file_names:
        # Load the data in the given file
        if 'non_blockade' in file:
            all_maps_transport = load_data_wo_dir(file, group, dataset, channels, all_maps_transport)
        else:
            all_maps_pos, all_maps_neg = load_data(file, group, dataset, channels, all_maps_blockade, all_maps_transport)

    print(all_maps_blockade)
    ###########
    # Blockade
    ###########
    print('###################### blockade #########################')
    off = get_offset(2597, all_maps_blockade[0][0], 0, 1)
    for i, map in enumerate(all_maps_blockade):
        r, err = process_data(map, y_tuple(map[-1], map[-2], 20, off, 1), (50, 50), #+10
                              file_dir, 1, True, 0.35, 0.18)
        ratios_blockade.append(r)
        ratios_err_blockade.append(err)

def both_dir_400mT():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "400mT_both_dir")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_transport = []
    ratios_err_transport = []
    ratios_blockade = []
    ratios_err_blockade = []
    all_maps_blockade = []
    all_maps_transport = []
    single_maps_blockade = []
    single_maps_transport = []

    for file in file_names:
        # Load the data in the given file
        if 'both_dir_2' in file:
            all_maps_blockade, all_maps_transport = load_data(file, group, dataset, channels, all_maps_blockade,
                                                              all_maps_transport)
        else:
            all_maps_transport = load_data_wo_dir(file, group, dataset, channels, all_maps_transport)

    #all_maps_transport = all_maps_transport[5:11] + all_maps_transport[13:]
    #for map in all_maps_transport:
    #    print(map[3])

    ###########
    # Blockade
    ###########
    print('###################### Blockade #########################')
    for map in all_maps_blockade[1:]:
        print(map[3])
        map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3], 1, 1, file_dir)
        single_maps_blockade.append(map_obj)

    single_maps_blockade[0].set_centers((5.1775, 5.267), (5.186, 5.260))
    single_maps_blockade[1].set_centers((5.1775, 5.270), (5.185, 5.260))
    single_maps_blockade[2].set_centers((5.1765, 5.270), (5.185, 5.260))
    single_maps_blockade[3].set_centers((5.1765, 5.270), (5.185, 5.260))
    single_maps_blockade[4].set_centers((5.1765, 5.270), (5.185, 5.260))
    single_maps_blockade[5].set_centers((5.1765, 5.270), (5.185, 5.260))
    single_maps_blockade[6].set_centers((5.1765, 5.270), (5.185, 5.260))
    single_maps_blockade[7].set_centers((5.1765, 5.270), (5.185, 5.260))
    single_maps_blockade[8].set_centers((5.1765, 5.270), (5.185, 5.260))
    single_maps_blockade[9].set_centers((5.175, 5.270), (5.185, 5.260))
    single_maps_blockade[10].set_centers((5.1775, 5.275), (5.185, 5.260))
    single_maps_blockade[11].set_centers((5.1775, 5.275), (5.185, 5.260))
    single_maps_blockade[12].set_centers((5.1775, 5.280), (5.185, 5.260))
    single_maps_blockade[13].set_centers((5.1775, 5.280), (5.185, 5.260))
    single_maps_blockade[14].set_centers((5.1775, 5.280), (5.185, 5.260))
    single_maps_blockade[15].set_centers((5.1775, 5.280), (5.185, 5.260))
    single_maps_blockade[16].set_centers((5.1775, 5.280), (5.185, 5.260))
    single_maps_blockade[17].set_centers((5.1775, 5.280), (5.185, 5.260))
    single_maps_blockade[18].set_centers((5.1775, 5.280), (5.185, 5.260))
    single_maps_blockade[19].set_centers((5.1775, 5.280), (5.185, 5.260))

    for map_obj in single_maps_blockade:
        map_obj.subtract_background()
        #map_obj.detect_lines()

    # 2950
    single_maps_blockade[0].add_vertical_line(-10, 5.1788)
    single_maps_blockade[0].add_vertical_line(-10, 5.1833)
    single_maps_blockade[0].add_horizontal_line(-0.85, 5.2657)
    single_maps_blockade[0].add_horizontal_line(-0.85, 5.2738)

    # 4900
    single_maps_blockade[1].add_vertical_line(-10.0, 5.1781)
    single_maps_blockade[1].add_vertical_line(-10.0, 5.1828)
    single_maps_blockade[1].add_horizontal_line(-0.85, 5.2661)
    single_maps_blockade[1].add_horizontal_line(-0.85, 5.2748)

    # 6850
    single_maps_blockade[2].add_vertical_line(-10.0, 5.1778)
    single_maps_blockade[2].add_vertical_line(-10.0, 5.1826)
    single_maps_blockade[2].add_horizontal_line(-0.85, 5.2671)
    single_maps_blockade[2].add_horizontal_line(-0.85, 5.2755)

    # 8800
    single_maps_blockade[3].add_vertical_line(-10.0, 5.1777)
    single_maps_blockade[3].add_vertical_line(-10.0, 5.1824)
    single_maps_blockade[3].add_horizontal_line(-0.85, 5.2675)
    single_maps_blockade[3].add_horizontal_line(-0.85, 5.2759)

    # 10750
    single_maps_blockade[4].add_vertical_line(-10.0, 5.1774)
    single_maps_blockade[4].add_vertical_line(-10.0, 5.1821)
    single_maps_blockade[4].add_horizontal_line(-0.85, 5.2677)
    single_maps_blockade[4].add_horizontal_line(-0.85, 5.2758)

    # 12700
    single_maps_blockade[5].add_vertical_line(-10.0, 5.1774)
    single_maps_blockade[5].add_vertical_line(-10.0, 5.1821)
    single_maps_blockade[5].add_horizontal_line(-0.85, 5.2678)
    single_maps_blockade[5].add_horizontal_line(-0.85, 5.2761)

    # 14650
    single_maps_blockade[6].add_vertical_line(-10.0, 5.1772)
    single_maps_blockade[6].add_vertical_line(-10.0, 5.182)
    single_maps_blockade[6].add_horizontal_line(-0.85, 5.2679)
    single_maps_blockade[6].add_horizontal_line(-0.85, 5.2762)

    # 16600
    single_maps_blockade[7].add_vertical_line(-10.0, 5.1771)
    single_maps_blockade[7].add_vertical_line(-10.0, 5.182)
    single_maps_blockade[7].add_horizontal_line(-0.85, 5.268)
    single_maps_blockade[7].add_horizontal_line(-0.85, 5.2765)

    # 18550
    single_maps_blockade[8].add_vertical_line(-10.0, 5.1772)
    single_maps_blockade[8].add_vertical_line(-10.0, 5.1819)
    single_maps_blockade[8].add_horizontal_line(-0.85, 5.2681)
    single_maps_blockade[8].add_horizontal_line(-0.85, 5.2765)

    # 20500
    single_maps_blockade[9].add_vertical_line(-10.0, 5.1771)
    single_maps_blockade[9].add_vertical_line(-10.0, 5.1819)
    single_maps_blockade[9].add_horizontal_line(-0.85, 5.268)
    single_maps_blockade[9].add_horizontal_line(-0.85, 5.2765)

    # 22450
    single_maps_blockade[10].add_vertical_line(-10.0, 5.1771)
    single_maps_blockade[10].add_vertical_line(-10.0, 5.1818)
    single_maps_blockade[10].add_horizontal_line(-0.85, 5.268)
    single_maps_blockade[10].add_horizontal_line(-0.85, 5.2765)

    # 24400
    single_maps_blockade[11].add_vertical_line(-10.0, 5.1771)
    single_maps_blockade[11].add_vertical_line(-10.0, 5.1819)
    single_maps_blockade[11].add_horizontal_line(-0.85, 5.2682)
    single_maps_blockade[11].add_horizontal_line(-0.85, 5.2768)

    # 26350
    single_maps_blockade[12].add_vertical_line(-10.0, 5.177)
    single_maps_blockade[12].add_vertical_line(-10.0, 5.1818)
    single_maps_blockade[12].add_horizontal_line(-0.85, 5.2682)
    single_maps_blockade[12].add_horizontal_line(-0.85, 5.2768)

    # 28300
    single_maps_blockade[13].add_vertical_line(-10.0, 5.1771)
    single_maps_blockade[13].add_vertical_line(-10.0, 5.1818)
    single_maps_blockade[13].add_horizontal_line(-0.85, 5.2683)
    single_maps_blockade[13].add_horizontal_line(-0.85, 5.2771)

    # 30250
    single_maps_blockade[14].add_vertical_line(-10.0, 5.1769)
    single_maps_blockade[14].add_vertical_line(-10.0, 5.1819)
    single_maps_blockade[14].add_horizontal_line(-0.85, 5.2684)
    single_maps_blockade[14].add_horizontal_line(-0.85, 5.277)

    # 32200
    single_maps_blockade[15].add_vertical_line(-10.0, 5.1769)
    single_maps_blockade[15].add_vertical_line(-10.0, 5.1818)
    single_maps_blockade[15].add_horizontal_line(-0.85, 5.2684)
    single_maps_blockade[15].add_horizontal_line(-0.85, 5.277)


    # 34150
    single_maps_blockade[16].add_vertical_line(-10.0, 5.1769)
    single_maps_blockade[16].add_vertical_line(-10.0, 5.1818)
    single_maps_blockade[16].add_horizontal_line(-0.85, 5.2684)
    single_maps_blockade[16].add_horizontal_line(-0.85, 5.277)


    # 36100
    single_maps_blockade[17].add_vertical_line(-10.0, 5.177)
    single_maps_blockade[17].add_vertical_line(-10.0, 5.1817)
    single_maps_blockade[17].add_horizontal_line(-0.85, 5.2685)
    single_maps_blockade[17].add_horizontal_line(-0.85, 5.2769)

    # 38050
    single_maps_blockade[18].add_vertical_line(-10.0, 5.177)
    single_maps_blockade[18].add_vertical_line(-10.0, 5.1818)
    single_maps_blockade[18].add_horizontal_line(-0.85, 5.2687)
    single_maps_blockade[18].add_horizontal_line(-0.85, 5.2772)

    # 40000
    single_maps_blockade[19].add_vertical_line(-10.0, 5.1769)
    single_maps_blockade[19].add_vertical_line(-10.0, 5.1818)
    single_maps_blockade[19].add_horizontal_line(-0.85, 5.2687)
    single_maps_blockade[19].add_horizontal_line(-0.85, 5.2772)




    for map_obj in single_maps_blockade:
        #print(map_obj.get_vertical_lines())
        map_obj.add_triangle()
        map_obj.plot_map()
        map_obj.save_map()
        ratio, sigma_ratio = map_obj.get_ratio()
        ratios_blockade.append(ratio)
        ratios_err_blockade.append(sigma_ratio)

    #############
    # Transport
    #############
    print('###################### Transport #########################')


    for map in all_maps_transport[1:22]:
        print(map[3])
        map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3], -1, 2, file_dir)
        single_maps_transport.append(map_obj)

    #single_maps_transport[0].set_comp_fac(0.004)
    #single_maps_transport[1].set_comp_fac(0.006)
    #single_maps_transport[2].set_comp_fac(0.007)
    #single_maps_transport[3].set_comp_fac(0.008)
    #single_maps_transport[4].set_comp_fac(0.009)
    #single_maps_transport[5].set_comp_fac(0.0093)
    #single_maps_transport[6].set_comp_fac(0.0095)
    #single_maps_transport[7].set_comp_fac(0.01)
    #single_maps_transport[8].set_comp_fac(0.01)
    #single_maps_transport[9].set_comp_fac(0.01)

    single_maps_transport[0].set_centers((5.177, 5.2675), (5.172, 5.2800))
    single_maps_transport[1].set_centers((5.178, 5.267), (5.174, 5.2800))
    single_maps_transport[2].set_centers((5.183, 5.2675), (5.174, 5.2800))
    single_maps_transport[3].set_centers((5.183, 5.2675), (5.174, 5.2800))
    single_maps_transport[4].set_centers((5.183, 5.2675), (5.174, 5.2800))
    single_maps_transport[5].set_centers((5.183, 5.2675), (5.174, 5.2800))
    single_maps_transport[6].set_centers((5.183, 5.2675), (5.174, 5.2800))
    single_maps_transport[7].set_centers((5.183, 5.2675), (5.174, 5.2800))
    single_maps_transport[8].set_centers((5.183, 5.2675), (5.174, 5.2800))
    single_maps_transport[9].set_centers((5.183, 5.2675), (5.174, 5.2800))
    single_maps_transport[10].set_centers((5.1775, 5.265), (5.176, 5.2800))
    single_maps_transport[11].set_centers((5.18, 5.263), (5.175, 5.2800))
    single_maps_transport[12].set_centers((5.18, 5.263), (5.175, 5.2800))
    single_maps_transport[13].set_centers((5.182, 5.261), (5.175, 5.2800))
    single_maps_transport[14].set_centers((5.182, 5.261), (5.175, 5.2800))
    single_maps_transport[15].set_centers((5.182, 5.261), (5.1775, 5.275))
    single_maps_transport[16].set_centers((5.1825, 5.262), (5.1775, 5.275))
    single_maps_transport[17].set_centers((5.185, 5.265), (5.1775, 5.2775))
    single_maps_transport[18].set_centers((5.185, 5.265), (5.175, 5.2775))
    single_maps_transport[19].set_centers((5.185, 5.263), (5.1775, 5.2775))
    single_maps_transport[20].set_centers((5.185, 5.263), (5.1775, 5.2775))

    for map_obj in single_maps_transport:
        map_obj.subtract_background()
        #map_obj.detect_lines()

    # 880
    single_maps_transport[0].add_vertical_line(-11, 5.1728)
    single_maps_transport[0].add_vertical_line(-11, 5.1773)
    single_maps_transport[0].add_horizontal_line(-0.85, 5.2771)
    single_maps_transport[0].add_horizontal_line(-0.85, 5.2855)

    # 1560
    single_maps_transport[1].add_vertical_line(-11, 5.1737)
    single_maps_transport[1].add_vertical_line(-11, 5.1782)
    single_maps_transport[1].add_horizontal_line(-0.85, 5.2758)
    single_maps_transport[1].add_horizontal_line(-0.85, 5.2841)

    # 2240
    single_maps_transport[2].add_vertical_line(-11, 5.1745)
    single_maps_transport[2].add_vertical_line(-11, 5.1787)
    single_maps_transport[2].add_horizontal_line(-0.85, 5.275)
    single_maps_transport[2].add_horizontal_line(-0.85, 5.2833)

    # 2920
    single_maps_transport[3].add_vertical_line(-11, 5.1745)
    single_maps_transport[3].add_vertical_line(-11, 5.1791)
    single_maps_transport[3].add_horizontal_line(-0.85, 5.2749)
    single_maps_transport[3].add_horizontal_line(-0.85, 5.283)

    # 3600
    single_maps_transport[4].add_vertical_line(-11, 5.175)
    single_maps_transport[4].add_vertical_line(-11, 5.1795)
    single_maps_transport[4].add_horizontal_line(-0.85, 5.2742)
    single_maps_transport[4].add_horizontal_line(-0.85, 5.2821)

    # 4280
    single_maps_transport[5].add_vertical_line(-11, 5.1751)
    single_maps_transport[5].add_vertical_line(-11, 5.1797)
    single_maps_transport[5].add_horizontal_line(-0.85, 5.2738)
    single_maps_transport[5].add_horizontal_line(-0.85, 5.2821)

    # 4960
    single_maps_transport[6].add_vertical_line(-11, 5.1753)
    single_maps_transport[6].add_vertical_line(-11, 5.1798)
    single_maps_transport[6].add_horizontal_line(-0.85, 5.2737)
    single_maps_transport[6].add_horizontal_line(-0.85, 5.2816)

    # 5640
    single_maps_transport[7].add_vertical_line(-11, 5.1754)
    single_maps_transport[7].add_vertical_line(-11, 5.1799)
    single_maps_transport[7].add_horizontal_line(-0.85, 5.2735)
    single_maps_transport[7].add_horizontal_line(-0.85, 5.2815)

    # 6320
    single_maps_transport[8].add_vertical_line(-11, 5.1756)
    single_maps_transport[8].add_vertical_line(-11, 5.1801)
    single_maps_transport[8].add_horizontal_line(-0.85, 5.2734)
    single_maps_transport[8].add_horizontal_line(-0.85, 5.2814)

    # 7000
    single_maps_transport[9].add_vertical_line(-11, 5.1756)
    single_maps_transport[9].add_vertical_line(-11, 5.1803)
    single_maps_transport[9].add_horizontal_line(-0.85, 5.2733)
    single_maps_transport[9].add_horizontal_line(-0.85, 5.2814)

    # 1000
    single_maps_transport[10].add_vertical_line(-11, 5.1733)
    single_maps_transport[10].add_vertical_line(-11, 5.1777)
    single_maps_transport[10].add_horizontal_line(-0.85, 5.2749)
    single_maps_transport[10].add_horizontal_line(-0.85, 5.2829)

    # 2950
    single_maps_transport[11].add_vertical_line(-11, 5.1749)
    single_maps_transport[11].add_vertical_line(-11, 5.1792)
    single_maps_transport[11].add_horizontal_line(-0.85, 5.2725)
    single_maps_transport[11].add_horizontal_line(-0.85, 5.2804)

    # 4900
    single_maps_transport[12].add_vertical_line(-11, 5.1756)
    single_maps_transport[12].add_vertical_line(-11, 5.1799)
    single_maps_transport[12].add_horizontal_line(-0.85, 5.2717)
    single_maps_transport[12].add_horizontal_line(-0.85, 5.2796)

    # 6850
    single_maps_transport[13].add_vertical_line(-11, 5.1758)
    single_maps_transport[13].add_vertical_line(-11, 5.1802)
    single_maps_transport[13].add_horizontal_line(-0.85, 5.2711)
    single_maps_transport[13].add_horizontal_line(-0.85, 5.2791)

    # 8800
    single_maps_transport[14].add_vertical_line(-11, 5.176)
    single_maps_transport[14].add_vertical_line(-11, 5.1804)
    single_maps_transport[14].add_horizontal_line(-0.85, 5.2706)
    single_maps_transport[14].add_horizontal_line(-0.85, 5.2787)

    # 10750
    single_maps_transport[15].add_vertical_line(-11, 5.1762)
    single_maps_transport[15].add_vertical_line(-11, 5.1804)
    single_maps_transport[15].add_horizontal_line(-0.85, 5.2704)
    single_maps_transport[15].add_horizontal_line(-0.85, 5.2784)

    # 12700
    single_maps_transport[16].add_vertical_line(-11, 5.1763)
    single_maps_transport[16].add_vertical_line(-11, 5.1806)
    single_maps_transport[16].add_horizontal_line(-0.85, 5.2704)
    single_maps_transport[16].add_horizontal_line(-0.85, 5.2783)

    # 14650
    single_maps_transport[17].add_vertical_line(-11, 5.1774)
    single_maps_transport[17].add_vertical_line(-11, 5.1819)
    single_maps_transport[17].add_horizontal_line(-0.85, 5.2755)
    single_maps_transport[17].add_horizontal_line(-0.85, 5.2836)

    # 16600
    single_maps_transport[18].add_vertical_line(-11, 5.1774)
    single_maps_transport[18].add_vertical_line(-11, 5.1819)
    single_maps_transport[18].add_horizontal_line(-0.85, 5.2753)
    single_maps_transport[18].add_horizontal_line(-0.85, 5.2834)

    # 18550
    single_maps_transport[19].add_vertical_line(-11, 5.1774)
    single_maps_transport[19].add_vertical_line(-11, 5.1819)
    single_maps_transport[19].add_horizontal_line(-0.85, 5.2752)
    single_maps_transport[19].add_horizontal_line(-0.85, 5.2832)

    # 20500
    single_maps_transport[20].add_vertical_line(-11, 5.1776)
    single_maps_transport[20].add_vertical_line(-11, 5.1819)
    single_maps_transport[20].add_horizontal_line(-0.85, 5.2751)
    single_maps_transport[20].add_horizontal_line(-0.85, 5.2834)



    for map_obj in single_maps_transport:
        map_obj.add_triangle()
        map_obj.add_region(-0.0012)
        map_obj.add_region(-0.002)
        map_obj.add_region(-0.0025)
        map_obj.plot_map()
        map_obj.save_map()
        ratio_transport, ratio_err_transport = map_obj.get_ratio()
        ratios_transport.append(ratio_transport)
        ratios_err_transport.append(ratio_err_transport)

    ########################
    # Plotting and fitting
    ########################

    # Blockade
    plt.figure(figsize=(20, 12))
    ls = 'dashed'
    t_read_s = []
    for elem in all_maps_blockade[1:]:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    plt.scatter(t_read_s, ratios_blockade,
                facecolors='none', edgecolors='orangered')
    #plt.errorbar(t_read_s, ratios_blockade,
    #             ratios_err_blockade,
    #             linestyle='None', marker='.',
    #             color='mediumvioletred', elinewidth=0.5)
    np.save(os.path.join(file_dir, 'tread_blockade.npy'), t_read_s)
    np.save(os.path.join(file_dir, 'ratios_blockade.npy'), ratios_blockade)
    np.save(os.path.join(file_dir, 'ratios_err_blockade.npy'), ratios_err_blockade)


    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_blockade,
                           xdata=t_read_s,
                           sigma=ratios_err_blockade,
                           p0=[1, 0.5, 0.5])


    #popt, pcov = fit_with_derivative(exponential_model, t_read_s, ratios_blockade, p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    #print(popt, "\n", pcov)
    sigma = np.sqrt(np.diag(pcov[0]))

    plt.plot(t, exponential_model(t, *popt),
             label=rf'$\tau_b$ : {popt[0]} $\pm$ {sigma}$\mu$s',
             color='mediumblue', linestyle=ls, alpha=0.8)
    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f'exp_fit_blockade.png'))

    # Transport
    plt.figure(figsize=(20, 12))
    ls = 'dashed'
    t_read_s = []
    for elem in all_maps_transport[1:22]:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    ratios_transport = np.array(ratios_transport)
    ratios_err_transport = np.array(ratios_err_transport)
    print(ratios_transport[:, 0])

    np.save(os.path.join(file_dir, 'tread_transport.npy'), t_read_s)
    np.save(os.path.join(file_dir, 'ratios_transport.npy'), ratios_transport)
    np.save(os.path.join(file_dir, 'ratios_err_transport.npy'), ratios_err_transport)

    plt.scatter(t_read_s, ratios_transport[:, 2],
                 facecolors='none', edgecolors='orangered')

    plt.scatter(t_read_s, ratios_transport[:, 1],
                facecolors='none', edgecolors='mediumvioletred')

    popt1, pcov1 = curve_fit(exponential_model,
                           ydata=ratios_transport[:, 1],
                           xdata=t_read_s,
                           sigma=ratios_err_transport[:, 1],
                           p0=[1, 0.5, 0.5])

    sigma1 = np.sqrt(np.diag(pcov1))

    plt.plot(t, exponential_model(t, *popt1),
             label=rf'$\tau_b$ : {popt1[0]} $\pm$ {sigma1}$\mu$s',
             color='gray', linestyle=ls, alpha=0.8)

    popt2, pcov2 = curve_fit(exponential_model,
                           ydata=ratios_transport[:, 2],
                           xdata=t_read_s[:],
                           sigma=ratios_err_transport[:, 2],
                           p0=[1, 0.5, 0.5])

    sigma2 = np.sqrt(np.diag(pcov2))

    plt.plot(t, exponential_model(t, *popt2),
             label=rf'$\tau_b$ : {popt2[0]} $\pm$ {sigma2}$\mu$s',
             color='gray', linestyle=ls, alpha=0.8)


    plt.legend()
    plt.ylim(0, 1)
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f'exp_fit_transport.png'))


def both_dir_0T():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "0T_both_dir")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_transport = []
    ratios_err_transport = []
    ratios_blockade = []
    ratios_err_blockade = []
    all_maps_blockade = []
    all_maps_transport = []
    single_maps_blockade = []
    single_maps_transport = []

    for file in file_names:
        # Load the data in the given file
        if 'transport' in file:
            all_maps_transport = load_data_wo_dir(file, group, dataset, channels, all_maps_transport)
        elif 'blockade' in file:
            all_maps_blockade = load_data_wo_dir(file, group, dataset, channels, all_maps_blockade)

    ###########
    # Blockade
    ###########
    print('###################### Blockade #########################')
    for map in all_maps_blockade[1:]:
        print(map[3])
        map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3], 1, 1, file_dir)
        single_maps_blockade.append(map_obj)

    single_maps_blockade[0].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[1].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[2].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[3].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[4].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[5].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[6].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[7].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[8].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[9].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[10].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[11].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[12].set_centers((5.1825, 5.285), (5.1850, 5.2725))
    single_maps_blockade[13].set_centers((5.1845, 5.283), (5.1850, 5.2725))
    single_maps_blockade[14].set_centers((5.1835, 5.284), (5.1850, 5.2725))
    single_maps_blockade[15].set_centers((5.1825, 5.284), (5.1850, 5.2725))

    for map_obj in single_maps_blockade:
        map_obj.subtract_background()
        #map_obj.detect_lines()

    # 1857
    single_maps_blockade[0].add_horizontal_line(-0.85, 5.2795)
    single_maps_blockade[0].add_horizontal_line(-0.85, 5.2875)
    single_maps_blockade[0].add_vertical_line(-10.0, 5.1791)
    single_maps_blockade[0].add_vertical_line(-10.0, 5.1838)

    # 2535
    single_maps_blockade[1].add_horizontal_line(-0.85, 5.2799)
    single_maps_blockade[1].add_horizontal_line(-0.85, 5.2878)
    single_maps_blockade[1].add_vertical_line(-10.0, 5.179)
    single_maps_blockade[1].add_vertical_line(-10.0, 5.1837)

    # 3214
    single_maps_blockade[2].add_horizontal_line(-0.85, 5.2801)
    single_maps_blockade[2].add_horizontal_line(-0.85, 5.2879)
    single_maps_blockade[2].add_vertical_line(-10.0, 5.1789)
    single_maps_blockade[2].add_vertical_line(-10.0, 5.1836)

    # 3892
    single_maps_blockade[3].add_horizontal_line(-0.85, 5.2802)
    single_maps_blockade[3].add_horizontal_line(-0.85, 5.2879)
    single_maps_blockade[3].add_vertical_line(-10.0, 5.1789)
    single_maps_blockade[3].add_vertical_line(-10.0, 5.1835)

    # 4571
    single_maps_blockade[4].add_horizontal_line(-0.85, 5.2803)
    single_maps_blockade[4].add_horizontal_line(-0.85, 5.288)
    single_maps_blockade[4].add_vertical_line(-10.0, 5.1787)
    single_maps_blockade[4].add_vertical_line(-10.0, 5.1834)

    # 5250
    single_maps_blockade[5].add_horizontal_line(-0.85, 5.2804)
    single_maps_blockade[5].add_horizontal_line(-0.85, 5.2879)
    single_maps_blockade[5].add_vertical_line(-10.0, 5.1786)
    single_maps_blockade[5].add_vertical_line(-10.0, 5.1832)

    # 5928
    single_maps_blockade[6].add_horizontal_line(-0.85, 5.2808)
    single_maps_blockade[6].add_horizontal_line(-0.85, 5.2878)
    single_maps_blockade[6].add_vertical_line(-10.0, 5.1787)
    single_maps_blockade[6].add_vertical_line(-10.0, 5.183)

    # 6607
    single_maps_blockade[7].add_horizontal_line(-0.85, 5.2806)
    single_maps_blockade[7].add_horizontal_line(-0.85, 5.288)
    single_maps_blockade[7].add_vertical_line(-10.0, 5.1786)
    single_maps_blockade[7].add_vertical_line(-10.0, 5.1831)

    # 7285
    single_maps_blockade[8].add_horizontal_line(-0.85, 5.2807)
    single_maps_blockade[8].add_horizontal_line(-0.85, 5.288)
    single_maps_blockade[8].add_vertical_line(-10.0, 5.1786)
    single_maps_blockade[8].add_vertical_line(-10.0, 5.1831)

    # 7964
    single_maps_blockade[9].add_horizontal_line(-0.85, 5.2807)
    single_maps_blockade[9].add_horizontal_line(-0.85, 5.288)
    single_maps_blockade[9].add_vertical_line(-10.0, 5.1787)
    single_maps_blockade[9].add_vertical_line(-10.0, 5.18296)

    # 8642
    single_maps_blockade[10].add_horizontal_line(-0.85, 5.28076)
    single_maps_blockade[10].add_horizontal_line(-0.85, 5.288)
    single_maps_blockade[10].add_vertical_line(-10.0, 5.1788)
    single_maps_blockade[10].add_vertical_line(-10.0, 5.1832)

    # 9321
    single_maps_blockade[11].add_horizontal_line(-0.85, 5.28058)
    single_maps_blockade[11].add_horizontal_line(-0.85, 5.288)
    single_maps_blockade[11].add_vertical_line(-10.0, 5.17873)
    single_maps_blockade[11].add_vertical_line(-10.0, 5.1832)

    # 10000
    single_maps_blockade[12].add_horizontal_line(-0.85, 5.2804)
    single_maps_blockade[12].add_horizontal_line(-0.85, 5.2881)
    single_maps_blockade[12].add_vertical_line(-10.0, 5.1786)
    single_maps_blockade[12].add_vertical_line(-10.0, 5.1831)

    # 500
    single_maps_blockade[13].add_horizontal_line(-0.85, 5.286)
    single_maps_blockade[13].add_horizontal_line(-0.85, 5.2776)
    single_maps_blockade[13].add_vertical_line(-11.0, 5.1804)
    single_maps_blockade[13].add_vertical_line(-11.0, 5.1853)

    # 1250
    single_maps_blockade[14].add_horizontal_line(-0.85, 5.2872)
    single_maps_blockade[14].add_horizontal_line(-0.85, 5.2792)
    single_maps_blockade[14].add_vertical_line(-11.0, 5.1794)
    single_maps_blockade[14].add_vertical_line(-11.0, 5.1842)

    # 2000
    single_maps_blockade[15].add_horizontal_line(-0.85, 5.2876)
    single_maps_blockade[15].add_horizontal_line(-0.85, 5.2798)
    single_maps_blockade[15].add_vertical_line(-11.0, 5.1792)
    single_maps_blockade[15].add_vertical_line(-11.0, 5.1838)


    for map_obj in single_maps_blockade:
        #print(map_obj.get_vertical_lines())
        map_obj.add_triangle()
        map_obj.plot_map()
        #map_obj.save_map()
        ratio, sigma_ratio = map_obj.get_ratio()
        ratios_blockade.append(ratio)
        ratios_err_blockade.append(sigma_ratio)

    #############
    # Transport
    #############
    print('###################### Transport #########################')
    all_maps_transport = all_maps_transport[:10]+all_maps_transport[11:]
    for map in all_maps_transport:
        print(map[3])
        map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3], -1, 3, file_dir)
        map_obj.subtract_background()
        #map_obj.detect_lines()
        single_maps_transport.append(map_obj)

    single_maps_transport[0].set_centers((5.178, 5.28), (5.1750, 5.2930))
    single_maps_transport[1].set_centers((5.178, 5.28), (5.1750, 5.2930))
    single_maps_transport[2].set_centers((5.178, 5.28), (5.1750, 5.2930))
    single_maps_transport[3].set_centers((5.178, 5.28), (5.1750, 5.2930))
    single_maps_transport[4].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[5].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[6].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[7].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[8].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[9].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[10].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[11].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[12].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[13].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[14].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[15].set_centers((5.180, 5.2775), (5.1750, 5.2930))
    single_maps_transport[16].set_centers((5.180, 5.2775), (5.1750, 5.2930))



    for map_obj in single_maps_transport:
        map_obj.subtract_background()
        #map_obj.detect_lines()

    # 100
    single_maps_transport[0].add_horizontal_line(-0.85, 5.2934)
    single_maps_transport[0].add_horizontal_line(-0.85, 5.302)
    single_maps_transport[0].add_vertical_line(-10, 5.1739)
    single_maps_transport[0].add_vertical_line(-10, 5.1788)

    # 130
    single_maps_transport[1].add_horizontal_line(-0.85, 5.2929)
    single_maps_transport[1].add_horizontal_line(-0.85, 5.3012)
    single_maps_transport[1].add_vertical_line(-10, 5.1743)
    single_maps_transport[1].add_vertical_line(-10, 5.1792)

    # 160
    single_maps_transport[2].add_horizontal_line(-0.85, 5.2921)
    single_maps_transport[2].add_horizontal_line(-0.85, 5.3008)
    single_maps_transport[2].add_vertical_line(-10, 5.1744)
    single_maps_transport[2].add_vertical_line(-10, 5.1795)

    # 190
    single_maps_transport[3].add_horizontal_line(-0.85, 5.2919)
    single_maps_transport[3].add_horizontal_line(-0.85, 5.3005)
    single_maps_transport[3].add_vertical_line(-10, 5.1747)
    single_maps_transport[3].add_vertical_line(-10, 5.1796)

    # 220
    single_maps_transport[4].add_horizontal_line(-0.85, 5.2915)
    single_maps_transport[4].add_horizontal_line(-0.85, 5.3)
    single_maps_transport[4].add_vertical_line(-10, 5.1748)
    single_maps_transport[4].add_vertical_line(-10, 5.1799)

    # 250
    single_maps_transport[5].add_horizontal_line(-0.85, 5.2908)
    single_maps_transport[5].add_horizontal_line(-0.85, 5.2996)
    single_maps_transport[5].add_vertical_line(-10, 5.175)
    single_maps_transport[5].add_vertical_line(-10, 5.1801)

    # 280
    single_maps_transport[6].add_horizontal_line(-0.85, 5.2909)
    single_maps_transport[6].add_horizontal_line(-0.85, 5.2995)
    single_maps_transport[6].add_vertical_line(-10, 5.1752)
    single_maps_transport[6].add_vertical_line(-10, 5.1803)

    # 310
    single_maps_transport[7].add_horizontal_line(-0.85, 5.2908)
    single_maps_transport[7].add_horizontal_line(-0.85, 5.2994)
    single_maps_transport[7].add_vertical_line(-10, 5.1753)
    single_maps_transport[7].add_vertical_line(-10, 5.1805)

    # 340
    single_maps_transport[8].add_horizontal_line(-0.85, 5.2903)
    single_maps_transport[8].add_horizontal_line(-0.85, 5.2989)
    single_maps_transport[8].add_vertical_line(-10, 5.1753)
    single_maps_transport[8].add_vertical_line(-10, 5.1804)

    # 370
    single_maps_transport[9].add_horizontal_line(-0.85, 5.2903)
    single_maps_transport[9].add_horizontal_line(-0.85, 5.2987)
    single_maps_transport[9].add_vertical_line(-10, 5.1754)
    single_maps_transport[9].add_vertical_line(-10, 5.1804)

    # 400
    single_maps_transport[10].add_horizontal_line(-0.85, 5.29)
    single_maps_transport[10].add_horizontal_line(-0.85, 5.2985)
    single_maps_transport[10].add_vertical_line(-10, 5.1756)
    single_maps_transport[10].add_vertical_line(-10, 5.1805)

    # 500
    single_maps_transport[11].add_horizontal_line(-0.85, 5.2894)
    single_maps_transport[11].add_horizontal_line(-0.85, 5.2978)
    single_maps_transport[11].add_vertical_line(-10, 5.1757)
    single_maps_transport[11].add_vertical_line(-10, 5.1808)

    # 600
    single_maps_transport[12].add_horizontal_line(-0.85, 5.2892)
    single_maps_transport[12].add_horizontal_line(-0.85, 5.2978)
    single_maps_transport[12].add_vertical_line(-10, 5.1760)
    single_maps_transport[12].add_vertical_line(-10, 5.1812)

    # 700
    single_maps_transport[13].add_horizontal_line(-0.85, 5.289)
    single_maps_transport[13].add_horizontal_line(-0.85, 5.2972)
    single_maps_transport[13].add_vertical_line(-10, 5.1761)
    single_maps_transport[13].add_vertical_line(-10, 5.1812)

    # 800
    single_maps_transport[14].add_horizontal_line(-0.85, 5.2887)
    single_maps_transport[14].add_horizontal_line(-0.85, 5.2971)
    single_maps_transport[14].add_vertical_line(-10, 5.1763)
    single_maps_transport[14].add_vertical_line(-10, 5.1813)

    # 900
    single_maps_transport[15].add_horizontal_line(-0.85, 5.2886)
    single_maps_transport[15].add_horizontal_line(-0.85, 5.2968)
    single_maps_transport[15].add_vertical_line(-10, 5.1764)
    single_maps_transport[15].add_vertical_line(-10, 5.1815)

    # 1000
    single_maps_transport[16].add_horizontal_line(-0.85, 5.2884)
    single_maps_transport[16].add_horizontal_line(-0.85, 5.2968)
    single_maps_transport[16].add_vertical_line(-10, 5.1767)
    single_maps_transport[16].add_vertical_line(-10, 5.1816)

    for map_obj in single_maps_transport:
        map_obj.add_triangle()
        map_obj.add_region(-0.005)
        map_obj.plot_map()
        #map_obj.save_map()
        ratio_transport, ratio_err_transport = map_obj.get_ratio()
        ratios_transport.append(ratio_transport)
        ratios_err_transport.append(ratio_err_transport)

    ########################
    # Plotting and fitting
    ########################

    # Blockade
    plt.figure(figsize=(20, 12))
    ls = 'dashed'
    t_read_s = []
    for elem in all_maps_blockade[1:]:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    t_read_s = np.array(t_read_s)
    ratios_blockade = np.array(ratios_blockade)
    ratios_err_blockade = np.array(ratios_err_blockade)
    np.save(os.path.join(file_dir, 'tread_blockade.npy'), t_read_s)
    np.save(os.path.join(file_dir, 'ratios_blockade.npy'), ratios_blockade)
    np.save(os.path.join(file_dir, 'ratios_err_blockade.npy'), ratios_err_blockade)
    plt.scatter(t_read_s, ratios_blockade,
                facecolors='crimson', edgecolors='black', s=80)
    # plt.errorbar(t_read_s, ratios_blockade,
    #             ratios_err_blockade,
    #             linestyle='None', marker='.',
    #             color='mediumvioletred', elinewidth=0.5)

    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_blockade,
                           xdata=t_read_s,
                           sigma=ratios_err_blockade,
                           p0=[1, 0.5, 0.5])

    # popt, pcov = fit_with_derivative(exponential_model, t_read_s, ratios_blockade, p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    # print(popt, "\n", pcov)
    sigma = np.sqrt(np.diag(pcov)[0])

    plt.plot(t, exponential_model(t, *popt),
             label=rf'$\tau_b$ : {popt[0]} $\pm$ {sigma}$\mu$s',
             color='gray', linestyle=ls)
    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f'exp_fit_blockade.png'))

    # Transport

    plt.figure(figsize=(20, 12))
    ls = 'dashed'
    t_read_s = []
    for elem in all_maps_transport:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    t_read_s = np.array(t_read_s)
    ratios_transport = np.array(ratios_transport)
    ratios_err_transport = np.array(ratios_err_transport)
    np.save(os.path.join(file_dir, 'tread_transport.npy'), t_read_s)
    np.save(os.path.join(file_dir, 'ratios_transport.npy'), ratios_transport)
    np.save(os.path.join(file_dir, 'ratios_err_transport.npy'), ratios_err_transport)
    plt.scatter(t_read_s, ratios_transport,
                facecolors='orangered', edgecolors='black', s=80)

    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_transport,
                           xdata=t_read_s,
                           sigma=ratios_err_transport,
                           p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    sigma = np.sqrt(np.diag(pcov)[0])

    plt.plot(t, exponential_model(t, *popt),
             label=rf'$\tau_b$ : {popt[0]} $\pm$ {sigma}$\mu$s',
             color='gray', linestyle=ls)
    plt.legend()
    #plt.ylim(0, 1)
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f'exp_fit_transport.png'))


def both_dir_0T_leon():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "0T_both_dir_leon")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_transport = []
    ratios_err_transport = []
    ratios_blockade = []
    ratios_err_blockade = []
    all_maps_blockade = []
    all_maps_transport = []
    single_maps_blockade = []
    single_maps_transport = []

    for file in file_names:
        # Load the data in the given file
        if '427' in file:
            #all_maps_transport, _ = load_data(file, group, dataset, channels, all_maps_transport, all_maps_blockade)
            all_maps_transport = load_data_wo_dir(file, group, dataset, channels, all_maps_transport)
        elif 'blockade_2' in file:#or 'blockade_3' in file:
            all_maps_blockade = load_data_wo_dir(file, group, dataset, channels, all_maps_blockade)

    ###########
    # Blockade
    ###########
    print('###################### Blockade #########################')
    for map in all_maps_blockade:
        print(map[3])
        map_obj = SingleMap(map[2], map[1], map[0], 1500, map[3], -1, 1, file_dir)
        single_maps_blockade.append(map_obj)

    single_maps_blockade[0].set_centers((5.338, 5.21), (5.33, 5.22))
    single_maps_blockade[1].set_centers((5.345, 5.205), (5.325, 5.22))
    single_maps_blockade[2].set_centers((5.345, 5.205), (5.325, 5.22))
    single_maps_blockade[3].set_centers((5.345, 5.2), (5.325, 5.22))


    for map_obj in single_maps_blockade:
        map_obj.subtract_background()
        #map_obj.detect_lines()

    # 1000
    single_maps_blockade[0].add_horizontal_line(-0.09, 5.2128)
    single_maps_blockade[0].add_horizontal_line(-0.09, 5.218)
    single_maps_blockade[0].add_vertical_line(-1.1, 5.328)
    single_maps_blockade[0].add_vertical_line(-1.1, 5.3185)

    # 3419
    single_maps_blockade[1].add_horizontal_line(-0.09, 5.2108)
    single_maps_blockade[1].add_horizontal_line(-0.09, 5.216)
    single_maps_blockade[1].add_vertical_line(-1.1, 5.328)
    single_maps_blockade[1].add_vertical_line(-1.1, 5.3185)

    # 11696
    single_maps_blockade[2].add_horizontal_line(-0.09, 5.2095)
    single_maps_blockade[2].add_horizontal_line(-0.09, 5.2146)
    single_maps_blockade[2].add_vertical_line(-1.1, 5.328)
    single_maps_blockade[2].add_vertical_line(-1.1, 5.3187)

    # 40000
    single_maps_blockade[3].add_horizontal_line(-0.09, 5.209)
    single_maps_blockade[3].add_horizontal_line(-0.09, 5.2139)
    single_maps_blockade[3].add_vertical_line(-1.1, 5.328)
    single_maps_blockade[3].add_vertical_line(-1.1, 5.319)



    for map_obj in single_maps_blockade:
        #print(map_obj.get_vertical_lines())
        map_obj.add_triangle()
        map_obj.plot_map()
        map_obj.save_map()
        ratio, sigma_ratio = map_obj.get_ratio()
        ratios_blockade.append(ratio)
        ratios_err_blockade.append(sigma_ratio)

    #############
    # Transport
    #############
    print('###################### Transport #########################')
    all_maps_transport = all_maps_transport[3:10]
    for map in all_maps_transport:
        print(map[3])
        map_obj = SingleMap(map[2], map[1], map[0], 1500, map[3], 1, 3, file_dir)
        # map_obj.detect_lines()
        single_maps_transport.append(map_obj)

    single_maps_transport[0].set_centers((5.34, 5.2065), (5.3475, 5.1975))
    single_maps_transport[1].set_centers((5.34, 5.2065), (5.3475, 5.1975))
    single_maps_transport[2].set_centers((5.34, 5.2075), (5.3475, 5.1975))
    single_maps_transport[3].set_centers((5.34, 5.2075), (5.3475, 5.1975))
    single_maps_transport[4].set_centers((5.34, 5.2075), (5.3475, 5.1975))
    single_maps_transport[5].set_centers((5.34, 5.208), (5.3475, 5.2))
    single_maps_transport[6].set_centers((5.34, 5.208), (5.3475, 5.1975))

    for map_obj in single_maps_transport:
        map_obj.subtract_background()
        # map_obj.detect_lines()


    for map_obj in single_maps_transport:
        #map_obj.add_triangle()
        #map_obj.add_region(-0.005)
        map_obj.plot_map()
        map_obj.save_map()
        #ratio_transport, ratio_err_transport = map_obj.get_ratio()
        #ratios_transport.append(ratio_transport)
        #ratios_err_transport.append(ratio_err_transport)

    ########################
    # Plotting and fitting
    ########################

    # Blockade
    plt.figure(figsize=(20, 12))
    ls = 'dashed'
    t_read_s = []
    for elem in all_maps_blockade:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    t_read_s = np.array(t_read_s)
    ratios_blockade = np.array(ratios_blockade)
    ratios_err_blockade = np.array(ratios_err_blockade)
    np.save(os.path.join(file_dir, 'tread_blockade.npy'), t_read_s)
    np.save(os.path.join(file_dir, 'ratios_blockade.npy'), ratios_blockade)
    np.save(os.path.join(file_dir, 'ratios_err_blockade.npy'), ratios_err_blockade)
    plt.scatter(t_read_s, ratios_blockade,
                facecolors='crimson', edgecolors='black', s=80)
    # plt.errorbar(t_read_s, ratios_blockade,
    #             ratios_err_blockade,
    #             linestyle='None', marker='.',
    #             color='mediumvioletred', elinewidth=0.5)

    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_blockade,
                           xdata=t_read_s,
                           sigma=ratios_err_blockade,
                           p0=[1, 0.5, 0.5])

    # popt, pcov = fit_with_derivative(exponential_model, t_read_s, ratios_blockade, p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    # print(popt, "\n", pcov)
    sigma = np.sqrt(np.diag(pcov)[0])

    plt.plot(t, exponential_model(t, *popt),
             label=rf'$\tau_b$ : {popt[0]} $\pm$ {sigma}$\mu$s',
             color='gray', linestyle=ls)
    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f'exp_fit_blockade.png'))

    # Transport

    # plt.figure(figsize=(20, 12))
    # ls = 'dashed'
    # t_read_s = []
    # for elem in all_maps_transport:
    #     t_read_s.append(elem[3])
    # t_read_s = np.array(t_read_s) * 125 * 1e-6
    # print(t_read_s)
    # t_read_s = np.array(t_read_s)
    # ratios_transport = np.array(ratios_transport)
    # ratios_err_transport = np.array(ratios_err_transport)
    # np.save(os.path.join(file_dir, 'tread_transport.npy'), t_read_s)
    # np.save(os.path.join(file_dir, 'ratios_transport.npy'), ratios_transport)
    # np.save(os.path.join(file_dir, 'ratios_err_transport.npy'), ratios_err_transport)
    # plt.scatter(t_read_s, ratios_transport,
    #             facecolors='orangered', edgecolors='black', s=80)
    #
    # popt, pcov = curve_fit(exponential_model,
    #                        ydata=ratios_transport,
    #                        xdata=t_read_s,
    #                        sigma=ratios_err_transport,
    #                        p0=[1, 0.5, 0.5])
    #
    # t = np.linspace(0, max(t_read_s), 100)
    # sigma = np.sqrt(np.diag(pcov)[0])
    #
    # plt.plot(t, exponential_model(t, *popt),
    #          label=rf'$\tau_b$ : {popt[0]} $\pm$ {sigma}$\mu$s',
    #          color='gray', linestyle=ls)
    # plt.legend()
    # #plt.ylim(0, 1)
    # plt.ylabel('Intensity ratio')
    # plt.xlabel(r'$T_{read}$ ($\mu$s)')
    # plt.tight_layout()
    # plt.savefig(os.path.join(file_dir, f'exp_fit_transport.png'))

def both_dir_1T():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "1T_both_dir")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_transport = []
    ratios_err_transport = []
    ratios_blockade = []
    ratios_err_blockade = []
    all_maps_blockade = []
    all_maps_transport = []
    single_maps_blockade = []
    single_maps_transport = []

    for file in file_names:
        # Load the data in the given file
        all_maps_blockade, all_maps_transport = load_data(file, group, dataset, channels,
                                                          all_maps_blockade, all_maps_transport)

    ###########
    # Blockade
    ###########
    print('###################### Blockade #########################')
    for map in all_maps_blockade[1:21]:
        print(map[3])
        map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3], 1, 1, file_dir)
        single_maps_blockade.append(map_obj)

    single_maps_blockade[0].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[1].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[2].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[3].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[4].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[5].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[6].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[7].set_centers((5.176, 5.265), (5.1800, 5.2500))
    single_maps_blockade[8].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[9].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[10].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[11].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[12].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[13].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[14].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[15].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[16].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[17].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[18].set_centers((5.176, 5.266), (5.1800, 5.2500))
    single_maps_blockade[19].set_centers((5.176, 5.266), (5.1800, 5.2500))

    for map_obj in single_maps_blockade:
        map_obj.subtract_background()
        #map_obj.detect_lines()

    # 2475
    single_maps_blockade[0].add_horizontal_line(-0.85, 5.2618)
    single_maps_blockade[0].add_horizontal_line(-0.85, 5.2701)
    single_maps_blockade[0].add_vertical_line(-11, 5.1727)
    single_maps_blockade[0].add_vertical_line(-11, 5.1777)

    # 4450
    single_maps_blockade[1].add_horizontal_line(-0.85, 5.2621)
    single_maps_blockade[1].add_horizontal_line(-0.85, 5.2706)
    single_maps_blockade[1].add_vertical_line(-11, 5.1725)
    single_maps_blockade[1].add_vertical_line(-11, 5.1774)

    # 6425
    single_maps_blockade[2].add_horizontal_line(-0.85, 5.2623)
    single_maps_blockade[2].add_horizontal_line(-0.85, 5.2707)
    single_maps_blockade[2].add_vertical_line(-11, 5.1724)
    single_maps_blockade[2].add_vertical_line(-11, 5.1773)

    # 8400
    single_maps_blockade[3].add_horizontal_line(-0.85, 5.2624)
    single_maps_blockade[3].add_horizontal_line(-0.85, 5.271)
    single_maps_blockade[3].add_vertical_line(-11, 5.1724)
    single_maps_blockade[3].add_vertical_line(-11, 5.1773)

    # 10375
    single_maps_blockade[4].add_horizontal_line(-0.85, 5.2625)
    single_maps_blockade[4].add_horizontal_line(-0.85, 5.2712)
    single_maps_blockade[4].add_vertical_line(-11, 5.1724)
    single_maps_blockade[4].add_vertical_line(-11, 5.1772)

    # 12350
    single_maps_blockade[5].add_horizontal_line(-0.85, 5.2626)
    single_maps_blockade[5].add_horizontal_line(-0.85, 5.2713)
    single_maps_blockade[5].add_vertical_line(-11, 5.1723)
    single_maps_blockade[5].add_vertical_line(-11, 5.1772)

    # 14325
    single_maps_blockade[6].add_horizontal_line(-0.85, 5.2623)
    single_maps_blockade[6].add_horizontal_line(-0.85, 5.2712)
    single_maps_blockade[6].add_vertical_line(-11, 5.1723)
    single_maps_blockade[6].add_vertical_line(-11, 5.1772)

    # 16300
    single_maps_blockade[7].add_horizontal_line(-0.85, 5.2623)
    single_maps_blockade[7].add_horizontal_line(-0.85, 5.2712)
    single_maps_blockade[7].add_vertical_line(-11, 5.1723)
    single_maps_blockade[7].add_vertical_line(-11, 5.1771)

    # 18275
    single_maps_blockade[8].add_horizontal_line(-0.85, 5.2623)
    single_maps_blockade[8].add_horizontal_line(-0.85, 5.2712)
    single_maps_blockade[8].add_vertical_line(-11, 5.1722)
    single_maps_blockade[8].add_vertical_line(-11, 5.1771)

    # 20250
    single_maps_blockade[9].add_horizontal_line(-0.85, 5.2626)
    single_maps_blockade[9].add_horizontal_line(-0.85, 5.2712)
    single_maps_blockade[9].add_vertical_line(-11, 5.1723)
    single_maps_blockade[9].add_vertical_line(-11, 5.1771)

    # 22225
    single_maps_blockade[10].add_horizontal_line(-0.85, 5.2625)
    single_maps_blockade[10].add_horizontal_line(-0.85, 5.2711)
    single_maps_blockade[10].add_vertical_line(-11, 5.1722)
    single_maps_blockade[10].add_vertical_line(-11, 5.1771)

    # 24200
    single_maps_blockade[11].add_horizontal_line(-0.85, 5.2625)
    single_maps_blockade[11].add_horizontal_line(-0.85, 5.2711)
    single_maps_blockade[11].add_vertical_line(-11, 5.1722)
    single_maps_blockade[11].add_vertical_line(-11, 5.17705)

    # 26175
    single_maps_blockade[12].add_horizontal_line(-0.85, 5.2625)
    single_maps_blockade[12].add_horizontal_line(-0.85, 5.2711)
    single_maps_blockade[12].add_vertical_line(-11, 5.1723)
    single_maps_blockade[12].add_vertical_line(-11, 5.1771)

    # 28150
    single_maps_blockade[13].add_horizontal_line(-0.85, 5.2624)
    single_maps_blockade[13].add_horizontal_line(-0.85, 5.2712)
    single_maps_blockade[13].add_vertical_line(-11, 5.1723)
    single_maps_blockade[13].add_vertical_line(-11, 5.1771)

    # 30125
    single_maps_blockade[14].add_horizontal_line(-0.85, 5.2625)
    single_maps_blockade[14].add_horizontal_line(-0.85, 5.2711)
    single_maps_blockade[14].add_vertical_line(-11, 5.1723)
    single_maps_blockade[14].add_vertical_line(-11, 5.1772)

    # 32100
    single_maps_blockade[15].add_horizontal_line(-0.85, 5.2626)
    single_maps_blockade[15].add_horizontal_line(-0.85, 5.2713)
    single_maps_blockade[15].add_vertical_line(-11, 5.1723)
    single_maps_blockade[15].add_vertical_line(-11, 5.1771)

    # 34075
    single_maps_blockade[16].add_horizontal_line(-0.85, 5.2625)
    single_maps_blockade[16].add_horizontal_line(-0.85, 5.2711)
    single_maps_blockade[16].add_vertical_line(-11, 5.1722)
    single_maps_blockade[16].add_vertical_line(-11, 5.177)

    # 36050
    single_maps_blockade[17].add_horizontal_line(-0.85, 5.2627)
    single_maps_blockade[17].add_horizontal_line(-0.85, 5.2712)
    single_maps_blockade[17].add_vertical_line(-11, 5.1723)
    single_maps_blockade[17].add_vertical_line(-11, 5.1771)

    # 38025
    single_maps_blockade[18].add_horizontal_line(-0.85, 5.2627)
    single_maps_blockade[18].add_horizontal_line(-0.85, 5.2713)
    single_maps_blockade[18].add_vertical_line(-11, 5.1722)
    single_maps_blockade[18].add_vertical_line(-11, 5.1771)

    # 40000
    single_maps_blockade[19].add_horizontal_line(-0.85, 5.2627)
    single_maps_blockade[19].add_horizontal_line(-0.85, 5.27115)
    single_maps_blockade[19].add_vertical_line(-11, 5.1723)
    single_maps_blockade[19].add_vertical_line(-11, 5.1771)

    for map_obj in single_maps_blockade:
        #print(map_obj.get_vertical_lines())
        map_obj.add_triangle(offset=0.0025)
        map_obj.plot_map()
        map_obj.save_map()
        ratio, sigma_ratio = map_obj.get_ratio()
        ratios_blockade.append(ratio)
        ratios_err_blockade.append(sigma_ratio)


    #############
    # Transport
    #############
    print('###################### Transport #########################')

    for map in all_maps_transport[1:]:
        print(map[3])
        map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3], -1, 2, file_dir)
        single_maps_transport.append(map_obj)

    single_maps_transport[0].set_centers((5.176, 5.2525), (5.1725, 5.2650))
    single_maps_transport[1].set_centers((5.176, 5.2525), (5.1725, 5.2650))
    single_maps_transport[2].set_centers((5.176, 5.2525), (5.1725, 5.2650))
    single_maps_transport[3].set_centers((5.176, 5.2525), (5.1725, 5.2650))
    single_maps_transport[4].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[5].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[5].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[6].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[7].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[8].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[9].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[10].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[11].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[12].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[13].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[14].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[15].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[16].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[17].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[18].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[19].set_centers((5.177, 5.2525), (5.1725, 5.2650))
    single_maps_transport[20].set_centers((5.1750, 5.2525), (5.17, 5.2680))
    single_maps_transport[21].set_centers((5.1750, 5.2525), (5.17, 5.2680))
    single_maps_transport[22].set_centers((5.1750, 5.2525), (5.17, 5.2680))
    single_maps_transport[23].set_centers((5.1750, 5.2525), (5.17, 5.2680))
    single_maps_transport[24].set_centers((5.1750, 5.2525), (5.17, 5.2680))
    single_maps_transport[25].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[26].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[27].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[28].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[29].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[30].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[31].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[32].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[33].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[34].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[35].set_centers((5.1760, 5.2525), (5.17, 5.2680))
    single_maps_transport[36].set_centers((5.1760, 5.2525), (5.17, 5.2680))

    for map_obj in single_maps_transport:
        map_obj.subtract_background()
       #map_obj.detect_lines()

    # 2475
    single_maps_transport[0].add_horizontal_line(-0.85, 5.2638)
    single_maps_transport[0].add_horizontal_line(-0.85, 5.2721)
    single_maps_transport[0].add_vertical_line(-11, 5.1716)
    single_maps_transport[0].add_vertical_line(-11, 5.1759)

    # 4450
    single_maps_transport[1].add_horizontal_line(-0.85, 5.2633)
    single_maps_transport[1].add_horizontal_line(-0.85, 5.2717)
    single_maps_transport[1].add_vertical_line(-11, 5.1719)
    single_maps_transport[1].add_vertical_line(-11, 5.1761)

    # 6425
    single_maps_transport[2].add_horizontal_line(-0.85, 5.2631)
    single_maps_transport[2].add_horizontal_line(-0.85, 5.2716)
    single_maps_transport[2].add_vertical_line(-11, 5.172)
    single_maps_transport[2].add_vertical_line(-11, 5.1762)

    # 8400
    single_maps_transport[3].add_horizontal_line(-0.85, 5.2631)
    single_maps_transport[3].add_horizontal_line(-0.85, 5.2715)
    single_maps_transport[3].add_vertical_line(-11, 5.172)
    single_maps_transport[3].add_vertical_line(-11, 5.1763)

    # 10375
    single_maps_transport[4].add_horizontal_line(-0.85, 5.2633)
    single_maps_transport[4].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[4].add_vertical_line(-11, 5.172)
    single_maps_transport[4].add_vertical_line(-11, 5.1763)

    # 12350
    single_maps_transport[5].add_horizontal_line(-0.85, 5.2631)
    single_maps_transport[5].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[5].add_vertical_line(-11, 5.1721)
    single_maps_transport[5].add_vertical_line(-11, 5.1763)

    # 14325
    single_maps_transport[6].add_horizontal_line(-0.85, 5.2632)
    single_maps_transport[6].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[6].add_vertical_line(-11, 5.1721)
    single_maps_transport[6].add_vertical_line(-11, 5.1764)

    # 16300
    single_maps_transport[7].add_horizontal_line(-0.85, 5.2632)
    single_maps_transport[7].add_horizontal_line(-0.85, 5.2712)
    single_maps_transport[7].add_vertical_line(-11, 5.1721)
    single_maps_transport[7].add_vertical_line(-11, 5.1764)

    # 18275
    single_maps_transport[8].add_horizontal_line(-0.85, 5.2633)
    single_maps_transport[8].add_horizontal_line(-0.85, 5.2712)
    single_maps_transport[8].add_vertical_line(-11, 5.1721)
    single_maps_transport[8].add_vertical_line(-11, 5.1764)

    # 20250
    single_maps_transport[9].add_horizontal_line(-0.85, 5.2633)
    single_maps_transport[9].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[9].add_vertical_line(-11, 5.1722)
    single_maps_transport[9].add_vertical_line(-11, 5.1765)

    # 22225
    single_maps_transport[10].add_horizontal_line(-0.85, 5.2632)
    single_maps_transport[10].add_horizontal_line(-0.85, 5.2712)
    single_maps_transport[10].add_vertical_line(-11, 5.1721)
    single_maps_transport[10].add_vertical_line(-11, 5.1765)

    # 24200
    single_maps_transport[11].add_horizontal_line(-0.85, 5.2633)
    single_maps_transport[11].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[11].add_vertical_line(-11, 5.1722)
    single_maps_transport[11].add_vertical_line(-11, 5.1764)

    # 26175
    single_maps_transport[12].add_horizontal_line(-0.85, 5.2632)
    single_maps_transport[12].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[12].add_vertical_line(-11, 5.1722)
    single_maps_transport[12].add_vertical_line(-11, 5.1764)

    # 28150
    single_maps_transport[13].add_horizontal_line(-0.85, 5.2631)
    single_maps_transport[13].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[13].add_vertical_line(-11, 5.1722)
    single_maps_transport[13].add_vertical_line(-11, 5.1765)

    # 30125
    single_maps_transport[14].add_horizontal_line(-0.85, 5.2632)
    single_maps_transport[14].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[14].add_vertical_line(-11, 5.1722)
    single_maps_transport[14].add_vertical_line(-11, 5.1764)

    # 32100
    single_maps_transport[15].add_horizontal_line(-0.85, 5.2633)
    single_maps_transport[15].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[15].add_vertical_line(-11, 5.1721)
    single_maps_transport[15].add_vertical_line(-11, 5.1764)

    # 34075
    single_maps_transport[16].add_horizontal_line(-0.85, 5.2631)
    single_maps_transport[16].add_horizontal_line(-0.85, 5.2712)
    single_maps_transport[16].add_vertical_line(-11, 5.1721)
    single_maps_transport[16].add_vertical_line(-11, 5.1764)

    # 36050
    single_maps_transport[17].add_horizontal_line(-0.85, 5.2632)
    single_maps_transport[17].add_horizontal_line(-0.85, 5.2712)
    single_maps_transport[17].add_vertical_line(-11, 5.1721)
    single_maps_transport[17].add_vertical_line(-11, 5.1764)

    # 38025
    single_maps_transport[18].add_horizontal_line(-0.85, 5.2631)
    single_maps_transport[18].add_horizontal_line(-0.85, 5.2713)
    single_maps_transport[18].add_vertical_line(-11, 5.1721)
    single_maps_transport[18].add_vertical_line(-11, 5.1764)

    # 40000
    single_maps_transport[19].add_horizontal_line(-0.85, 5.2631)
    single_maps_transport[19].add_horizontal_line(-0.85, 5.2712)
    single_maps_transport[19].add_vertical_line(-11, 5.1721)
    single_maps_transport[19].add_vertical_line(-11, 5.1764)

    # 300
    single_maps_transport[20].add_horizontal_line(-0.85, 5.2679)
    single_maps_transport[20].add_horizontal_line(-0.85, 5.276)
    single_maps_transport[20].add_vertical_line(-11, 5.1693)
    single_maps_transport[20].add_vertical_line(-11, 5.1738)

    # 406
    single_maps_transport[21].add_horizontal_line(-0.85, 5.2673)
    single_maps_transport[21].add_horizontal_line(-0.85, 5.2755)
    single_maps_transport[21].add_vertical_line(-11, 5.1698)
    single_maps_transport[21].add_vertical_line(-11, 5.1743)

    # 512
    single_maps_transport[22].add_horizontal_line(-0.85, 5.2669)
    single_maps_transport[22].add_horizontal_line(-0.85, 5.275)
    single_maps_transport[22].add_vertical_line(-11, 5.1701)
    single_maps_transport[22].add_vertical_line(-11, 5.1747)

    # 618
    single_maps_transport[23].add_horizontal_line(-0.85, 5.2664)
    single_maps_transport[23].add_horizontal_line(-0.85, 5.2746)
    single_maps_transport[23].add_vertical_line(-11, 5.1704)
    single_maps_transport[23].add_vertical_line(-11, 5.1749)

    # 725
    single_maps_transport[24].add_horizontal_line(-0.85, 5.2662)
    single_maps_transport[24].add_horizontal_line(-0.85, 5.2743)
    single_maps_transport[24].add_vertical_line(-11, 5.1706)
    single_maps_transport[24].add_vertical_line(-11, 5.175)

    # 831
    single_maps_transport[25].add_horizontal_line(-0.85, 5.2661)
    single_maps_transport[25].add_horizontal_line(-0.85, 5.2739)
    single_maps_transport[25].add_vertical_line(-11, 5.1709)
    single_maps_transport[25].add_vertical_line(-11, 5.175)

    # 937
    single_maps_transport[26].add_horizontal_line(-0.85, 5.2657)
    single_maps_transport[26].add_horizontal_line(-0.85, 5.2738)
    single_maps_transport[26].add_vertical_line(-11, 5.1711)
    single_maps_transport[26].add_vertical_line(-11, 5.1752)

    # 1043
    single_maps_transport[27].add_horizontal_line(-0.85, 5.2657)
    single_maps_transport[27].add_horizontal_line(-0.85, 5.2736)
    single_maps_transport[27].add_vertical_line(-11, 5.1711)
    single_maps_transport[27].add_vertical_line(-11, 5.1753)

    # 1150
    single_maps_transport[28].add_horizontal_line(-0.85, 5.2652)
    single_maps_transport[28].add_horizontal_line(-0.85, 5.2732)
    single_maps_transport[28].add_vertical_line(-11, 5.1711)
    single_maps_transport[28].add_vertical_line(-11, 5.1754)

    # 1256
    single_maps_transport[29].add_horizontal_line(-0.85, 5.2652)
    single_maps_transport[29].add_horizontal_line(-0.85, 5.2732)
    single_maps_transport[29].add_vertical_line(-11, 5.1712)
    single_maps_transport[29].add_vertical_line(-11, 5.1756)

    # 1362
    single_maps_transport[30].add_horizontal_line(-0.85, 5.2651)
    single_maps_transport[30].add_horizontal_line(-0.85, 5.273)
    single_maps_transport[30].add_vertical_line(-11, 5.1713)
    single_maps_transport[30].add_vertical_line(-11, 5.1756)

    # 1468
    single_maps_transport[31].add_horizontal_line(-0.85, 5.265)
    single_maps_transport[31].add_horizontal_line(-0.85, 5.273)
    single_maps_transport[31].add_vertical_line(-11, 5.1714)
    single_maps_transport[31].add_vertical_line(-11, 5.1756)

    # 1575
    single_maps_transport[32].add_horizontal_line(-0.85, 5.2648)
    single_maps_transport[32].add_horizontal_line(-0.85, 5.2728)
    single_maps_transport[32].add_vertical_line(-11, 5.1714)
    single_maps_transport[32].add_vertical_line(-11, 5.1756)

    # 1681
    single_maps_transport[33].add_horizontal_line(-0.85, 5.2647)
    single_maps_transport[33].add_horizontal_line(-0.85, 5.2727)
    single_maps_transport[33].add_vertical_line(-11, 5.1715)
    single_maps_transport[33].add_vertical_line(-11, 5.1758)

    # 1787
    single_maps_transport[34].add_horizontal_line(-0.85, 5.2647)
    single_maps_transport[34].add_horizontal_line(-0.85, 5.2726)
    single_maps_transport[34].add_vertical_line(-11, 5.1715)
    single_maps_transport[34].add_vertical_line(-11, 5.1757)

    # 1893
    single_maps_transport[35].add_horizontal_line(-0.85, 5.2646)
    single_maps_transport[35].add_horizontal_line(-0.85, 5.2725)
    single_maps_transport[35].add_vertical_line(-11, 5.1715)
    single_maps_transport[35].add_vertical_line(-11, 5.1758)

    # 2000
    single_maps_transport[36].add_horizontal_line(-0.85, 5.2646)
    single_maps_transport[36].add_horizontal_line(-0.85, 5.2727)
    single_maps_transport[36].add_vertical_line(-11, 5.1715)
    single_maps_transport[36].add_vertical_line(-11, 5.1759)

    for map_obj in single_maps_transport:
        map_obj.add_triangle()
        map_obj.add_region(-0.002)
        map_obj.add_region(-0.003)
        map_obj.plot_map()
        map_obj.save_map()
        ratio_transport, ratio_err_transport = map_obj.get_ratio()
        ratios_transport.append(ratio_transport)
        ratios_err_transport.append(ratio_err_transport)


    ########################
    # Plotting and fitting
    ########################

    # Blockade
    plt.figure(figsize=(20, 12))
    ls = 'dashed'
    t_read_s = []
    for elem in all_maps_blockade[1:21]:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)

    # plt.errorbar(t_read_s, ratios_blockade,
    #             ratios_err_blockade,
    #             linestyle='None', marker='.',
    #             color='mediumvioletred', elinewidth=0.5)
    np.save(os.path.join(file_dir, 'tread_blockade.npy'), t_read_s)
    np.save(os.path.join(file_dir,'ratios_blockade.npy'), ratios_blockade)
    np.save(os.path.join(file_dir,'ratios_err_blockade.npy'), ratios_err_blockade)


    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_blockade,
                           xdata=t_read_s,
                           sigma=ratios_err_blockade,
                           p0=[1, 0.5, 0.5])

    # popt, pcov = fit_with_derivative(exponential_model, t_read_s, ratios_blockade, p0=[1, 0.5, 0.5])

    t = np.linspace(0, max(t_read_s), 100)
    # print(popt, "\n", pcov)
    sigma = np.sqrt(np.diag(pcov)[0])

    plt.plot(t, exponential_model(t, *popt),
             label=rf'$\tau_b$ : {popt[0]} $\pm$ {sigma}$\mu$s',
             color='gray', linestyle=ls)

    plt.scatter(t_read_s, ratios_blockade,
                facecolors='crimson', edgecolors='black', s=80)

    #plt.scatter(t_read_s, ratios_blockade,
    #            facecolors='gray', edgecolors='black', s=80)

    plt.legend()
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f'exp_fit_blockade.png'))


    # Transport first region
    plt.figure(figsize=(20, 12))
    ls = 'dashed'
    t_read_s = []
    for elem in all_maps_transport[1:]:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    ratios_transport = np.array(ratios_transport)
    ratios_err_transport = np.array(ratios_err_transport)
    print(ratios_transport[:16, 1])

    #plt.scatter(t_read_s, ratios_transport[:, 1],
    #            facecolors='none', edgecolors='mediumvioletred', s=80)

    
    popt, pcov = curve_fit(exponential_model,
                           ydata=ratios_transport[:, 0],
                           xdata=t_read_s,
                           sigma=ratios_err_transport[:, 0],
                           p0=[1, 0.5, 0.5])

    sigma = np.sqrt(np.diag(pcov)[0])
    t = np.linspace(0, max(t_read_s), 100)

    plt.plot(t, exponential_model(t, *popt),
             label=rf'$\tau_b$ : {popt[0]} $\pm$ {sigma}$\mu$s',
             color='gray', linestyle=ls)

    plt.scatter(t_read_s, ratios_transport[:, 0],
                facecolors='orangered', edgecolors='black', s=80)

    #plt.scatter(t_read_s, ratios_transport[:, 0],
    #            facecolors='gray', edgecolors='black', s=80)
    """
    plt.legend()
    #plt.ylim(0, 1)
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f'exp_fit_transport.png'))
    """

    # Transport second region
    #plt.figure(figsize=(20, 12))
    #ls = 'dashed'
    t_read_s = []
    for elem in all_maps_transport[1:]:
        t_read_s.append(elem[3])
    t_read_s = np.array(t_read_s) * 125 * 1e-6
    print(t_read_s)
    ratios_transport = np.array(ratios_transport)
    ratios_err_transport = np.array(ratios_err_transport)
    print(ratios_transport[:, 1])

    np.save(os.path.join(file_dir, 'tread_transport.npy'), t_read_s)
    np.save(os.path.join(file_dir, 'ratios_transport.npy'), ratios_transport)
    np.save(os.path.join(file_dir, 'ratios_err_transport.npy'), ratios_err_transport)
    
    popt1, pcov1 = curve_fit(exponential_model,
                             ydata=ratios_transport[:, 1],
                             xdata=t_read_s,
                             sigma=ratios_err_transport[:, 1],
                             p0=[1, 0.5, 0.5])

    sigma1 = np.sqrt(np.diag(pcov1)[0])
    t = np.linspace(0, max(t_read_s), 100)

    plt.plot(t, exponential_model(t, *popt1),
             label=rf'$\tau_b$ : {popt1[0]} $\pm$ {sigma1}$\mu$s',
             color='gray', linestyle=ls)

    plt.scatter(t_read_s, ratios_transport[:, 1],
                facecolors='mediumvioletred', edgecolors='black', s=80)
    
    plt.legend()
    # plt.ylim(0, 1)
    plt.ylabel('Intensity ratio')
    plt.xlabel(r'$T_{read}$ ($\mu$s)')
    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, f'exp_fit_transport1.png'))

def maps_vs_pg13():
    # Get the current working directory
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, "pg13")
    file_names = find_hdf5_files(file_dir)

    group = "Data"
    dataset = "Data/Data"
    channels = "Data/Channel names"

    ratios_blockade = []
    ratios_err_blockade = []
    all_maps_blockade = []
    single_maps_blockade = []

    for file in file_names:
        # Load the data in the given file
        all_maps_blockade = load_data_wo_dir(file, group, dataset, channels,
                                                          all_maps_blockade)

    print('###################### Blockade #########################')
    for map in all_maps_blockade[1:21]:
        print(map[3])
        map_obj = SingleMap(map[1], map[2], map[0], 1500, map[3], 1, 1, file_dir)
        single_maps_blockade.append(map_obj)


def delta_x(tread, tini, comp_fac):
    return -0.05*comp_fac*((tread-tini)/(tread+tini))

def y(x, m, b):
    return m*x + b




def main():
    #regime_tl_larger_ti()
    #regime_ti_larger_tl_1T()
    #regime_ti_larger_tl_450mT()
    #regime_ti_larger_tl_200mT()
    #both_dir_500mT()
    #both_dir_400mT()
    #both_dir_0T()
    #both_dir_1T()
    #both_dir_0T_leon()
    maps_vs_pg13()

if __name__ == "__main__":
    main()
