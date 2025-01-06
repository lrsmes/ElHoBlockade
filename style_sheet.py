import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from collections.abc import Iterable  # Import Iterable
import numpy as np


# Assuming a standard text width for an A4 document in LaTeX to be approximately 6.3 inches
# Define fontsizes, linewidths, ticklenghts, that will be applied for all plots

TEXT_WIDTH_INCHES = 6.45
xfontsize = 12 #xaxis label fontsize
yfontsize = 12 # yaxis label fontsize
labelfontsize = 10 # ticklabel fontsize
textfontsize = 12 
line_width = 0.4
tick_width = 0.4
tick_length = 3.5

def width(fraction_of_textwidth=1.0, aspect_ratio = 0.75):
    figsize = (TEXT_WIDTH_INCHES * fraction_of_textwidth, TEXT_WIDTH_INCHES * fraction_of_textwidth * aspect_ratio)  
    fig = plt.figure(figsize=figsize)
    
    return fig

    
    
def create_subplots(num_rows, num_cols, textwidth_ratio=1.0, fig_ratio = 0.3, width_ratios=None, sharey=False):
    """
    Creates a figure with a grid of subplots.

    Parameters:
    num_rows : int
        Number of rows in the subplot grid.
    num_cols : int
        Number of columns in the subplot grid.
    textwidth_ratio : float, optional
        Ratio of the total text width that the figure should occupy.
    width_ratios : list, optional
        Relative widths of the columns. Default is None, which means equal widths.
    sharey : bool, optional
        Specifies whether y-axis will be shared among the subplots.
    
    Returns:
    fig : matplotlib.figure.Figure
        The created figure.
    axs : array of AxesSubplot
        Array of subplot axes.
    """
    figsize = (TEXT_WIDTH_INCHES * textwidth_ratio, TEXT_WIDTH_INCHES * textwidth_ratio * fig_ratio)
    fig, axs = plt.subplots(num_rows, num_cols,
                            figsize=figsize,
                            constrained_layout=True,
                            gridspec_kw={'width_ratios': width_ratios} if width_ratios else None,
                            sharey=sharey)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
    
    return fig, axs



def colorscale(is_2d, colors='viridis'):
    if is_2d:
        plt.set_cmap(colors)

def add_custom_colorbar(ax, im, location='right', orientation='vertical', aspect=20, fraction=0.05, pad=0.05, label=None):
    """
    Adds a customized colorbar to the given axes object.

    Parameters:
    ax : matplotlib.axes.Axes
        The axes object to which the colorbar will be added.
    im : ScalarMappable
        The Image, ContourSet, etc., to which the colorbar applies.
    location : str, optional
        The location of the colorbar ('right', 'left', 'top', 'bottom'). Default is 'right'.
    orientation : str, optional
        The orientation of the colorbar ('vertical' or 'horizontal'). Default is 'vertical'.
    aspect : float, optional
        The aspect ratio of the colorbar. Default is 20.
    fraction : float, optional
        The fraction of the axes that the colorbar should occupy. This is adjusted automatically based on orientation.
    pad : float, optional
        The space between the colorbar and the axes. Default is 0.05.
    label : str, optional
        The label text for the colorbar with LaTeX support.

    Returns:
    cbar : matplotlib.colorbar.Colorbar
        The created colorbar instance.
    """
    # Adjust fraction according to orientation
    cbar = ax.figure.colorbar(im, ax=ax, location=location, orientation=orientation,
                              aspect=aspect, fraction=fraction, pad=pad)
    if label:
        cbar.set_label(label, labelpad=10)

    return cbar


def lines(linewidth=0.7, ax_width=0.7, tick_width=0.4, tick_length=3.5, max_xticks=5, max_yticks=5, axes=None):
    """
    Configures line widths and tick properties for the plot.

    Parameters:
    linewidth : float
        Line width for plot lines.
    ax_width : float
        Line width for axes lines.
    tick_width : float
        Line width for major ticks.
    tick_length : float
        Length of major ticks.
    max_xticks : int
        Maximum number of major ticks on the x-axis.
    max_yticks : int
        Maximum number of major ticks on the y-axis.
    axes : matplotlib.axes.Axes or iterable of Axes, optional
        Specific axes object or iterable of axes objects to apply settings. If None, applies to current active axes.
    """
    if axes is None:
        axes = [plt.gca()]
    elif not isinstance(axes, Iterable):  # Check if axes is not an iterable
        axes = [axes]

    for ax in axes:
        # Set line width for plot lines and axes lines
        ax.spines['top'].set_linewidth(ax_width)
        ax.spines['bottom'].set_linewidth(ax_width)
        ax.spines['left'].set_linewidth(ax_width)
        ax.spines['right'].set_linewidth(ax_width)

        # Set properties for major ticks
        ax.tick_params(axis='both', which='major', width=tick_width, length=tick_length, direction='in')
        ax.tick_params(axis='x', which='both', top=True)  # Enable top ticks if not enabled
        ax.tick_params(axis='y', which='both', right=True)  # Enable right ticks if not enabled

        # Set the locator for major ticks
        ax.xaxis.set_major_locator(MaxNLocator(max_xticks))
        ax.yaxis.set_major_locator(MaxNLocator(max_yticks))

        ax.set_axisbelow(False)  # Ensure grid and minor ticks are below other plot elements
        print("This function has been called")

    
def def_fontsizes():
    return xfontsize, yfontsize, labelfontsize

def fonts(textfontsize = 12, labelfontsize = 10):
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "cmr10"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rc('axes', titlesize=textfontsize, labelsize=textfontsize)
    plt.rc('xtick', labelsize=labelfontsize)
    plt.rc('ytick', labelsize=labelfontsize)
    plt.rc('legend', fontsize=labelfontsize)


def savefig(filename, dpi=600, format='pdf', fig=None, bbox_inches='tight'):
    """
    Saves the specified figure with high resolution and in the given format.

    Parameters:
    filename : str
        Name and path where the file should be saved.
    dpi : int, optional
        Dots per inch for the figure. Default is 600 for high resolution.
    format : str, optional
        File format to save the figure in (e.g., 'pdf', 'png'). Default is 'pdf'.
    fig : matplotlib.figure.Figure, optional
        The specific figure to save. If None, saves the current figure.
    bbox_inches : str, optional
        Bounding box in inches: 'tight' means try to figure out the tight bbox of the figure.
    """
    if fig is not None:
        fig.savefig(filename, dpi=dpi, format=format, bbox_inches=bbox_inches)
    else:
        plt.savefig(filename, dpi=dpi, format=format, bbox_inches=bbox_inches)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# colormap using rwth colors
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    

def read_colors_from_gsl(filename):
    colors = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()  # Splits on whitespace
            if len(parts) >= 3:  # Check if there are at least three parts for RGB
                try:
                    # Normalize and convert to float RGB values expected by Matplotlib
                    rgb = [int(parts[i])/255.0 for i in range(3)]
                    colors.append(tuple(rgb))
                except ValueError:
                    continue  # Skip lines that do not contain proper numeric RGB values
    return colors

def create_colormap(colors):
    return LinearSegmentedColormap.from_list("custom_colormap", colors, N=len(colors))

def return_rwth_color(n):
    return read_colors_from_gsl(os.path.join(current_dir,'farbbibliothek_202307.gpl'))[n]

def make_rwth_cmap_white(rwth_c_n = [8,45,55,0], positions = [0.0, 0.25, 0.5, 0.75, 1.0]):
    colors_list = [(0,0,0)]
    for col in rwth_c_n:
        colors_list.append(return_rwth_color(col))
    # Create a list of tuples that combines positions and colors
    col_pos_list = [(pos, col) for pos, col in zip(positions, colors_list)]

    # Create the custom colormap
    custom_cmap_bprgw = LinearSegmentedColormap.from_list("custom_cmap_rwth_bprgw", col_pos_list)
    
    return custom_cmap_bprgw

def plot_error_bars(x, y, yerr, linewidth=0.7, capthick=0.7, capsize=2, markersize = 3, markeredgewidth=0.7, markeredgecolor=return_rwth_color(0), markerfacecolor='white', axes=None, c = return_rwth_color(0)):
    """
    Plots data points with asymmetric error bars and customizes their appearance.
    
    Parameters:
    x : array-like
        X-coordinates of the data points.
    y : array-like
        Y-coordinates of the data points.
    yerr : 2xN array-like
        Asymmetric error values for the error bars.
    linewidth : float
        Line width for error bars.
    capthick : float
        Thickness of the error bar caps.
    capsize : float
        Size of the error bar caps.
    axes : matplotlib.axes.Axes, optional
        The axes object on which to draw the error bars. If None, uses the current axes.
    """
    if axes is None:
        axes = plt.gca()

    errorbar_plot = axes.errorbar(x, y, yerr=yerr, fmt='o', capsize=capsize, markersize = markersize, 
                                  elinewidth=linewidth, capthick=capthick,
                                  linestyle='None', markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, color = c)

    lines(linewidth=linewidth, ax_width=linewidth, tick_width=0.4, tick_length=capsize, axes=axes)
    return errorbar_plot


# for i in range(65):
#     print(str(i) + '\t' +  str(tuple(255*x for x in return_rwth_color(i))))
    



# width(1,0.5)
# fonts()
# plt.plot([1,2],[3,4])
# plt.xlabel(r"$\alpha \gamma \mu \Gamma \phi$")
# plt.savefig("Latex_fonts.svg")

# ##############
# # Usage Example
# filename = 'farbbibliothek_202307.gpl'
# colors_rwth = read_colors_from_gsl(filename)

# color_1 = colors_rwth[25] # blue
# color_2 = colors_rwth[50] # dark red
# color_3 = colors_rwth[55] # purple
# color_4 = colors_rwth[8] # grey
# color_5 = (1,1,1) # white
# color_6 = colors_rwth[20] # darker blue
# color_7 = colors_rwth[45] # zahlen
# color_8 = colors_rwth[0] # dark blue
# color_9 = colors_rwth[7]

# rwth_blue = colors_rwth[0]
# rwth_purple = colors_rwth[55]
# rwth_red = colors_rwth[45]
# rwth_grey = colors_rwth[8]
# rwth_white = (0,0,0)
 
# #selected_colors = [color_5, color_4, color_2, color_3, color_6]
# selected_colors = [color_5, color_4, color_7, color_3, color_8]

# positions = [0.0, 0.25, 0.5, 0.75, 1.0]  # Adjust these values based on your needs

# # Create a list of tuples that combines positions and colors
# color_list = [(pos, col) for pos, col in zip(positions, selected_colors)]

# # Create the custom colormap
# custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", color_list)

