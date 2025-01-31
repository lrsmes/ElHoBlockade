import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import autograd.numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap



def plot_bands_blg(eigen_e, eigen_vec, fig_axis, energy_lim, indices={'M': 3175 - 53, 'K': 3175, 'Gamma': 3175 + 44},
                   projection="Upper Layer", add_colorbar=True,
                   show_y_labels=True, add_lines=True, xlims=(3100, 3200)):
    # Normalize for colors
    norm = Normalize(vmin=0, vmax=np.max(np.abs(eigen_vec) ** 2) * 1.0)

    # Truncate colormap to exclude bottom and top 10%
    original_cmap = plt.cm.twilight_shifted
    truncated_cmap = LinearSegmentedColormap.from_list(
        'truncated_cmap',
        original_cmap(np.linspace(0.15, 0.85, 256))
    )

    # Define projection indices based on layer selection
    if projection == "Upper Layer":
        p_idx = [4, 5, 6, 7]
    elif projection == "Lower Layer":
        p_idx = [0, 1, 2, 3]

    # Limit x-axis range
    x_min, x_max = xlims

    # Plot bands with colors
    for band_idx, band in enumerate(eigen_e.T):
        # Slice data to the desired x-axis range
        valid_indices = np.where((np.arange(band.size) >= x_min) & (np.arange(band.size) <= x_max))[0]
        if valid_indices.size == 0:
            continue  # Skip if no points are in the range

        # Select only the relevant range of band energies and eigenvectors
        band_limited = band[valid_indices]
        x_values = np.arange(band.size)[valid_indices]

        # Calculate color values at the valid indices
        color_values = ((np.abs(eigen_vec[valid_indices, band_idx, p_idx[0]]) ** 2 +
                         np.abs(eigen_vec[valid_indices, band_idx, p_idx[1]]) ** 2 +
                         np.abs(eigen_vec[valid_indices, band_idx, p_idx[2]]) ** 2 +
                         np.abs(eigen_vec[valid_indices, band_idx, p_idx[3]]) ** 2) /
                        ((np.abs(eigen_vec[valid_indices, band_idx, 0]) ** 2 +
                          np.abs(eigen_vec[valid_indices, band_idx, 1]) ** 2 +
                          np.abs(eigen_vec[valid_indices, band_idx, 2]) ** 2 +
                          np.abs(eigen_vec[valid_indices, band_idx, 3]) ** 2) +
                         (np.abs(eigen_vec[valid_indices, band_idx, 4]) ** 2 +
                          np.abs(eigen_vec[valid_indices, band_idx, 5]) ** 2 +
                          np.abs(eigen_vec[valid_indices, band_idx, 6]) ** 2 +
                          np.abs(eigen_vec[valid_indices, band_idx, 7]) ** 2)))

        # Plot as individual points (closely spaced to mimic a continuous line)
        fig_axis.scatter(x_values, band_limited, c=color_values, cmap=truncated_cmap, norm=norm, s=2, lw=0)

    # Set x and y axis limits
    if add_lines:
        fig_axis.axvline(3175 + 36, color='black', linewidth=0.5)  # Vertical line at K + 36
        fig_axis.axvline(3175 - 36, color='black', linewidth=0.5)  # Vertical line at K - 36

    fig_axis.set_ylim(energy_lim[0] * 1000, energy_lim[1] * 1000)
    fig_axis.set_xlim(xlims[0], xlims[1])  # Set range around K

    # X-axis labels and ticks
    fig_axis.set_xlabel('')
    high_sym_labels = [r'$\Gamma$', r'<-   K    -> ', r'M']
    high_sym_positions = [indices[key] for key in ['Gamma', 'K', 'M']]
    fig_axis.set_xticks(high_sym_positions)
    fig_axis.set_xticklabels(high_sym_labels)

    # Y-axis label logic based on show_y_labels
    if show_y_labels:
        fig_axis.set_ylabel('E - E$_{F}$ (meV)')
        fig_axis.yaxis.set_label_coords(-0.39, 0.5, transform=None)
    else:
        fig_axis.set_yticklabels([])  # Hide y-axis labels if show_y_labels is False

    # Customize ticks
    fig_axis.yaxis.set_major_locator(MaxNLocator(3))
    fig_axis.tick_params(axis='x', which='major', direction='in', width=0.5)
    fig_axis.tick_params(axis='y', which='major', direction='in', width=0.5)

    # Optionally add a horizontal colorbar based on add_colorbar
    if add_colorbar:
        # Create colorbar with custom positioning and orientation
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Define an inset axis for the colorbar: 50% of the plot's width, located in the upper left
        cbar_ax = inset_axes(fig_axis, width="50%", height="5%", loc='upper left',
                             bbox_to_anchor=(0.0, .1, 1, 1), bbox_transform=fig_axis.transAxes, borderpad=0)

        sm = ScalarMappable(norm=norm, cmap=truncated_cmap)
        sm.set_array([])  # Required for ScalarMappable

        # Set colorbar with ticks at the beginning and the end, rounded to 1 decimal
        cbar_ticks = [np.round(norm.vmin, 1), np.round(norm.vmax, 1)]  # Set ticks at min and max
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=cbar_ticks)

        # Set the label to the right of the colorbar at the same height
        fig_axis.text(.6, 1.09, '$|\psi_{L1}|^{2}$', transform=fig_axis.transAxes,
                      fontsize=fig_axis.yaxis.label.get_size(), va='center')

        # Adjust colorbar ticks inside, move labels to the top, and set size
        cbar.ax.tick_params(axis='x', direction='in', labelsize=fig_axis.yaxis.label.get_size(), labeltop=True,
                            labelbottom=False)
        # Update tick labels to rounded values
        cbar.ax.set_xticklabels([f'{tick:.1f}' for tick in cbar_ticks])



