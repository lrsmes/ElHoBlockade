# Matplotlib style sheet for use in the 2. Institut of physics B at the RWTH University Aachen:
# Import this module into your scipt file/ notebook. 
# Then call the function 'set_custom_style'.
# All specifications defined here will apply to plot created in your file.

import matplotlib.pyplot as plt

def set_custom_style(figure_size, aspect_ratio=0.6):

    #region Institute style guidelines -----------------------------
    # --------------------------------------------------------------
    # --------------------------------------------------------------

    """ 
    0.
    """
    print('\n----------------------Advice: --------------------------------------------')
    print("- If you have trouble starting: hand-draw plots to envision your figures (- C.S. 'saves time')")
    print('\n--------------None-implemented style guidelines: -------------------------')
    print("- For units don't use []. Instead use () parenthesis")
    print("- Align the boxes of plots, rather than the axis-labels or ticks.")
    print("- Plot colorbars horizontal, above the plots bax and aligned with the right edge.\n" +
          "  shirnk the colorbars so that ticks and colorbar labels have the correct font-size,\n" + 
          "  given that both main figure and colorbar are merged unscaled.")
    print("- Please use either (a), (b) (c) or a, b, c... for labeling the different panels (or\n" + 
          "  the form requested from the journal. In this context also font size 9 can in some cases be used")
    print("- About ticks:\n" + \
          "  + ticks need to be equidistant.\n" + \
          "  + the first and last tick do not need to be aligned with the edges of plots.\n" + \
          "  + choose tick-values with minimal digits")
    print("- Save plots in the svg file format!")
    

    """
    1. 
    Single columne figures should be 85 mm wide. Double columne figures should be 180 mm wide.
    mm = 1/2.54 * 1e-1 # Millimeters in Inches
    if figure_size == 'one column':
        # plt.rcParams['figure.figsize'] = (85 * mm, 85 * mm * aspect_ratio)   # 85mm wide
        # For some reason the figures do not come out as 80mm wide when using reasonable scales ...
        plt.rcParams['figure.figsize'] = (82 * mm, 82 * mm * aspect_ratio)   # 85mm wide
    elif figure_size == 'two columns':
        plt.rcParams['figure.figsize'] = (180 * mm, 180 * mm * aspect_ratio)   # 180mm wide
    elif figure_size == 'custom':
        plt.rcParams['figure.figsize'] = (41 * mm, 41 * mm * aspect_ratio)   # 41mm wide
    else:
        print("Shot")
        raise Exception('The chosen figure size does not comply with the style requirements!')
    """

    """
    2.
    Please use font Arial (or Helvetia) and font size should be between 5pt and 8pt. Use the 5pt for 
    subscript and superscript text. 7.5pt works well for numbers to label axis.
    -!-!-!-!-!-
    Note: The ratio between base and subscript/superscript is 0.7 
    by default and can not be changed to the best of my knowlegde. 
    This needs to be taken care of in Corel-Draw.
    -!-!-!-!-!-
    """
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.size'] = 8
    
    # New setting for label and tick fontsize
    plt.rcParams['xtick.labelsize'] = 7.5
    plt.rcParams['ytick.labelsize'] = 7.5
    
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 8


    """
    3.
    For boxes, axeses and ticks use a line thickness between 0.3 and 0.5pt.
    """
    # Setting for axis linewidth
    plt.rcParams['axes.linewidth'] = 0.3 # 0.5 #pt C.S. allows both 
    
    # Setting for box (frame) linewidth
    plt.rcParams['patch.linewidth'] = 0.3 # 0.5 #pt C.S. allows both 

    # Setting for tick linewidth

    plt.rcParams['xtick.major.width'] = 0.3 # 0.5 #pt C.S. allows both 
    plt.rcParams['ytick.major.width'] = 0.3 # 0.5 #pt C.S. allows both 

    # Single columne figures should be 85 mm wide. Double columne figures should be 180 mm wide.


    """
    4.
    Please use RGB colour spectrum.
    """
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        (0, 0, 0),    # Black
        (1, 0, 0),    # Red
        (1, 0.647, 0), # Orange
        (1, 1, 0),    # Yellow
        (0, 1, 0),    # Green
        (0, 0, 1),    # Blue
        (0.545, 0, 0.545),  # Purple
    ])


    """
    5.
    Ticks on all sides and on the inside rather than outside.
    """
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.right'] = False
    plt.rcParams['ytick.left'] = True

    #endregion


    #region Personal style guidelines ------------------------------
    # --------------------------------------------------------------
    # --------------------------------------------------------------

    """ 
    7.
    Personal settings: dpi for saving images with plt.savefig()
    """
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.bbox'] = 'tight'


    """ 
    7.
    Personal settings:
    6.1 dpi for saving images with plt.savefig()
    """
    color_map = 'viridis'
    


    """
    8.
    Ticks on all sides and on the inside rather than outside.
    """
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    # plt.rcParams['xtick.top'] = False
    # plt.rcParams['xtick.bottom'] = True
    # plt.rcParams['ytick.right'] = False
    # plt.rcParams['ytick.left'] = True
    #endregion