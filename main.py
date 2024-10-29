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

def y_tuple(tread, FG14, diff, off, pulse_dir):
    FG14_temp = np.flip(FG14[0])
    step = np.abs(FG14_temp[1]-FG14_temp[0])
    FG14_val = off+pulse_dir*0.0055*((tread-5000)/(tread+5000))
    find_val = np.abs(FG14_temp-FG14_val)
    lower = np.abs(FG14_temp-FG14_val).argmin()
    if lower == 0:
        lower = np.round(-(np.min(find_val) // step))
    upper = lower + diff
    if tread == 20729.865928436448:
        print(FG14_temp[70], FG14_temp[150])
    return (lower, upper)

def get_offset(tread, FG14, y_cut, pulse_dir):
    FG14_temp = np.flip(FG14[0])
    FG14_val = FG14_temp[y_cut]
    FG14_off = FG14_val - pulse_dir*0.0055*((tread-5000)/(tread+5000))
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


def main():
    # regime_tl_larger_ti()
    regime_ti_larger_tl_1T()
    #regime_ti_larger_tl_450mT()
    # regime_ti_larger_tl_200mT()
    #both_dir_500mT()

if __name__ == "__main__":
    main()
