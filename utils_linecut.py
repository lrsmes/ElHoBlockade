import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import hdf5_helper as helper
from scipy import constants

'''
Functions used to perform analysis of detuning linecuts
'''


def lorentzian(E, A, g, E0, a, b):
    '''
    A function to describe a lorentzian function plus a linear shift in y to account for data where it is not centred on y=0.

    Arguments:
    E - energy (x-axis variable)
    A - peak amplitude
    g - peak width
    E0 - peak centre
    a - gradient of linear shift
    b - y intercept of linear shift

    Output:
    y - lorentzian function plus linear shift
    '''
    y = A * (g / 2 * np.pi) / (g ** 2 / 4 + (E - E0) ** 2) + a * E + b
    return y


def multiplelorentzian(E, *params):
    '''
    Function to fit multiple lorentzians (each with a linear shift in y), depending on the number of input parameters given.

    Arguments:
    E - energy (x-axis variable)
    params - parameters for the lorentzians. Need to give 5N parameters, where N is an integer and is the number of the lorentzians to be fitted.

    Output:
    y - sum of multiple lorentzians
    '''
    y = np.zeros_like(E)
    for i in range(0, len(params), 5):
        A = params[i]
        g = params[i + 1]
        E0 = params[i + 2]
        a = params[i + 3]
        b = params[i + 4]
        y = y + lorentzian(E, A, g, E0, a, b)
    return y


def fit2lorentzian(p0, fig, ax, E, R):
    '''
    Function to automating the fitting procedure for 2 peaks, plot the resulting data and fit and return the relevant parameters for further analysis.

    Arguments:
    p0 - initial guesses for the fit parameters
    fig, ax - objects for plotting of results
    E - energy (x-axis variable)
    R - demodulated amplitude (y-axis variable)

    Outputs:
    width1 - width of peak 1
    peak1 - position of peak 1
    width2 - width of peak 2
    peak2 - position of peak 2
    np.sqrt(cov[1][1]) - error on width 1 (from fitting error)
    np.sqrt(cov[2][2]) - error on peak 1 (from fitting error)
    np.sqrt(cov[6][6]) - error on width 2 (from fitting error)
    np.sqrt(cov[7][7]) - error on peak 2 (from fitting error)
    '''
    popt, cov = curve_fit(multiplelorentzian, E, R, p0=p0)
    ax.plot(E, R, color='red')
    ax.plot(E, multiplelorentzian(E, *popt), color='black')
    width1 = popt[1]
    peak1 = popt[2]
    width2 = popt[6]
    peak2 = popt[7]
    return width1, peak1, width2, peak2, np.sqrt(cov[1][1]), np.sqrt(cov[2][2]), np.sqrt(cov[6][6]), np.sqrt(cov[7][7])


def fitlorentzian(p0, fig, ax, E, R):
    '''
    Function to automating the fitting procedure for 1 peak, plot the resulting data and fit and return the relevant parameters for further analysis.

    Arguments:
    p0 - initial guesses for the fit parameters
    fig, ax - objects for plotting of results
    E - energy (x-axis variable)
    R - demodulated amplitude (y-axis variable)

    Outputs:
    width1 - width of peak
    np.sqrt(cov[1][1]) - error on width  (from fitting error)
    '''
    popt, cov = curve_fit(lorentzian, E, R, p0=p0)
    ax.plot(E, R, color='red')
    ax.plot(E, lorentzian(E, *popt), color='black')
    ax.plot(E, lorentzian(E, *p0), color='purple')
    width1 = popt[1]
    return width1, np.sqrt(cov[1][1])


class linewidthanalysis:
    '''
    Class to create an object in the analysis and perform important pre-processing steps, convert from finger gate voltage to
    energy and extract linecuts for further analysis.
    '''

    def __init__(self, filename, datashape, datalengthFG, datalength_y, leverarmFG, FG_channel_name,
                 param_channel_name):
        '''
        Initialising the class.

        Arguments -
        filename - filename for data to be analysed, string
        datashape - shape of data, (lengthofFG(integer), 3, lengthofparam(integer)). 3 is for storing FG voltage, varied parameter values
        and the demodulated amplitude
        datalengthFG - length of FG voltage data, integer
        datalength_y - length of data of varied parameter (y-axis variable), integer
        leverarmFG - lever arm of the relevant finger gate in eV/V, float
        FG_channel_name - name of the channel in Labber of the finger gate, string
        param_channel_name - name of the channel in Labber of the varied parameter, string
        '''
        self._filename = filename
        self._datashape = datashape
        self._datalength_y = datalength_y
        self._datalengthFG = datalengthFG
        self._leverarmFG = leverarmFG
        self._FG_channel_name = FG_channel_name
        self._param_channel_name = param_channel_name

    def read_data(self):
        '''
        Method to read in the relevant data from the hdf5 file and return the finger gate voltage, y-axis variable and 2D demodulated amplitude as arrays.

        Outputs:
        FG- array of finger gate voltage, 1D
        y - array of y axis values, 1D
        data[:,2,:] - demodulated amplitude, 2D array
        '''
        group = "Data"
        dataset = "Data/Data"
        channels = "Data/Channel names"
        channel_data = helper.read_file(self._filename, group, dataset, channels, information=False)

        FG = channel_data[self._FG_channel_name]
        dem_R = channel_data['UHFLI - Demod1R']
        param = channel_data[self._param_channel_name]

        data = np.ndarray(self._datashape)
        data[:, 0, :] = FG[:, :]
        data[:, 1, :] = param[:, :]
        data[:, 2, :] = dem_R[:, :]
        FG = data[:, 0, 0]
        y = data[0, 1, :]

        return FG, y, data[:, 2, :]

    def remove_average(self, R):

        '''
        Method for removing average from data.

        Arguments:
        R - input linecut of demodulated R, 1D array

        Output
        R - input R linecut minus average, 1D array
        '''
        R = R - np.average(R)
        return R

    def remove_linear_bg(self, constant, other_timescale, FG, y, R):
        '''
        Method to remove a linear background in the finger gate direction. Only used for measurements with varied tread or tini

        Arguments:
        constant - gradient of linear background
        other_timescale - other timescale (either tini or tread) not being measured here
        FG - array of finger gate voltages
        y- array of y values (varied parameters)
        R- demodulated amplitude linecut array

        Outputs:
        R - demodulated amplitude linecut with linear background subtracted, array
        '''
        R = -R / (y + other_timescale) - FG * constant
        return R

    def normalise_data(self, R):
        '''
        Method to normalise data in the finger gate voltage direction

        Arguments:
        Arguments:
        R - input linecut of demodulated R, 1D array

        Output
        R - normalised R linecut, 1D array
        '''
        R = R / np.linalg.norm(R, axis=0)
        return R

    def convert_to_energy_get_linecut(self, FG, R, a, FG_selected, linecut_no, selected):
        '''
        Method to convert from gate voltages to energy and extract a linecut. There are 2 different conversion equations
        for energy to voltage depending on whether the measurements were done versus FG14 or FG12.

        Arguments:
        FG - array of finger gate voltages
        R - 1D linecut array of demodulated R
        a - gradient of line in equation relating FG12 and FG14 for the detuning cut
        FG_selected - 12 or 14, selected FG for measurements
        linecut_no_selected - index of selected linecut, integer
        selected - index of starting point of x axis to start linecut (sometimes good to not do the whole thing as it's easier to fit the correct feature later on)
        Outputs:
        E - linecut x axis values, in energy in eV
        R - demodulated amplitude linecut
        '''
        if FG_selected == 14:
            E = FG * self._leverarmFG * np.sqrt(1 + (1 / a) ** 2)
        if FG_selected == 12:
            E = FG * self._leverarmFG * np.sqrt(1 + a ** 2)
        return E[selected:], R[selected:, linecut_no]


def get_results(p0_incomplete, plotnamespecific, E, R, j):
    '''
    Function to perform the fitting procedure for 2 peaks and return the relevant fit parameters with errors.

    Arguments:
    p0_incomplete - list of 10 floats, initial guesses for amplitudes, peak widths and peak centres. Leave a and b to be 0 as these are set later on in the code.
    plotnamespecific - string, name for savile pdf file in correct folder, example 'varyingtread_' for each linecut
    E - 1D array, energy values (origin)
    R - demodulated amplitude linecut
    j - number labelling plot/linecut number (used in a loop to loop through linecuts in the general case)

    Outputs:
    separation - separation between the 2 peaks
    separation_error - error in separation from fitting
    width 1 - width of peak 1
    width 1_error - error on width of peak 1
    width 2 - width of peak 2
    width 2_error - error on width of peak 2
    '''
    # Setting up initial guesses
    p0 = np.ndarray(10)
    ind = [0, 1, 2, 3, 5, 6, 7, 8]
    for k in ind:
        p0[k] = p0_incomplete[k]
    # Setting the intial guess for the constant shift in y,parameter called b, to be the mean value of R in the linecut
    p0[4] = np.mean(R)
    p0[9] = np.mean(R)
    # Fitting and plotting results
    fig, ax = plt.subplots()
    results = fit2lorentzian(p0, fig, ax, E, R)
    separation = results[3] - results[1]
    width1 = results[0]
    width2 = results[2]
    separation_error = np.sqrt(results[5] ** 2 + results[7] ** 2)
    width1_error = results[4]
    width2_error = results[6]
    plotname = '02.results' + plotnamespecific + f'{j}' + '.pdf'
    plt.savefig(plotname)
    return separation, separation_error, width1, width1_error, width2, width2_error,


def get_results_vs_param(ind, y_value, param_axis_label, plotname, y_all, separation_all, separation_all_error,
                         width1_all, width1_all_error, width2_all, width2_all_error):
    '''
    Function to take in all results from fit and only return and plot trends for selected linecuts which are fitted correctly (avoiding anomalies
    from only fitting 1 peak instead of 2, fitting to a smaller peak than the data describes due to lots of datapoints that form a peak (in the noise) etc.)

    Arguments:
    ind - indices of the linecuts to be included (manually checked from plots from earlier analysis)
    y_value, if 0 then the x-axis for measurements with varying tread or tini will be converted from number of points to seconds using
    the AWG sampling rate, if not varying tread or tini (eg. for varying the magnetic field), then can set to any other non-zero integer and will
    do nothing
    param_axis_label - axis label for varied parameter, string
    pltoname - name of pdf file to save plot to, string
    y_all - array of varied parameter values
    separation_all - separation between the 2 peaks
    separation_all_error - error in separation from fitting
    width 1_all - width of peak 1
    width 1_all_error - error on width of peak 1
    width 2_all - width of peak 2
    width 2_all_error - error on width of peak 2

    Outputs:
    Saves plot to a pdf file in 02.results
    '''
    # Choosing values from fits that were successful using selected indices and making array of the resulting fit parameters, for each parameter.
    y_all_final = []
    width1_all_final = []
    width2_all_final = []
    separation_all_final = []
    separation_all_error_final = []
    width1_all_error_final = []
    width2_all_error_final = []
    for i in ind:
        width1_all_final.append(width1_all[i])
        width2_all_final.append(width2_all[i])
        width1_all_error_final.append(width1_all_error[i])
        width2_all_error_final.append(width2_all_error[i])
        separation_all_error_final.append(separation_all_error[i])
        separation_all_final.append(separation_all[i])
        y_all_final.append(y_all[i])
    fig, ax = plt.subplots(ncols=3, figsize=(20, 10))
    y_all_final = np.array(y_all_final)
    width1_all_final = np.array(width1_all_final)
    width2_all_final = np.array(width2_all_final)
    width1_all_error_final = np.array(width1_all_error_final)
    width2_all_error_final = np.array(width2_all_error_final)
    separation_all_final = np.array(separation_all_final)
    separation_all_error_final = np.array(separation_all_error_final)
    # Convert from number of points to seconds when meausurements are vs tread or tini
    if y_value == 0:
        y_all_final = y_all_final / (8e9)  # to get time in s, divide by AWG sampling rate

    # Converting from energy in eV to frequency in Hz
    width1_all_final = (width1_all_final * constants.e) / constants.Planck
    width1_all_error_final = (width1_all_error_final * constants.e) / constants.Planck
    width2_all_final = (width2_all_final * constants.e) / constants.Planck
    width2_all_error_final = (width2_all_error_final * constants.e) / constants.Planck
    separation_all_final = (separation_all_final * constants.e) / constants.Planck
    separation_all_error_final = (separation_all_error_final * constants.e) / constants.Planck

    # Plotting separation and peak widths vs the varied parameter
    ax[0].errorbar(y_all_final, width1_all_final * 1e-9, yerr=width1_all_error_final * 1e-9, fmt='.', capsize=3)
    ax[0].set_xlabel(param_axis_label)
    ax[0].set_ylabel('Width of peak 1 (GHz)')
    ax[0].set_ylim(ymin=0)
    ax[1].errorbar(y_all_final, width2_all_final * 1e-9, yerr=width2_all_error_final * 1e-9, fmt='.', capsize=3)
    ax[1].set_xlabel(param_axis_label)
    ax[1].set_ylabel('Width of peak 2 (GHz)')
    ax[1].set_ylim(ymin=0)
    ax[2].errorbar(y_all_final, separation_all_final * 1e-9, yerr=separation_all_error_final * 1e-9, fmt='.', capsize=3)
    ax[2].set_xlabel(param_axis_label)
    ax[2].set_ylabel('Peak separation (GHz)')
    ax[2].set_ylim(ymin=0)
    plt.savefig(plotname)