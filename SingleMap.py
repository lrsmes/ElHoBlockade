import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import math
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from sklearn.cluster import DBSCAN, HDBSCAN
from skimage.filters import threshold_otsu
from skimage.graph import rag_mean_color
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb, rgb2gray
from Data_analysis_and_transforms import correct_median_diff
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture


class SingleMap:
    def __init__(self, FG12=None, FG14=None, Demod1R=None, tini=None, tread=None,
                 pulse_dir=None, mode=None, file_dir=None):
        """
        Initialize the SingleMap class with optional parameters.

        Parameters:
            FG12 (np.array): Finger gate voltage FG12.
            FG14 (np.array): Finger gate voltage FG14.
            Demod1R (np.ndarray): 2D numpy array representing the map.
            tini (float): Initial time parameter.
            tread (float): Reading time parameter.
            pulse_dir (any): Direction of the pulse.
            comp_fac (any): Compensation factor.
            mode (int): Mode for region splitting and ratio calculation.
        """
        self.FG12 = FG12[0, :] if FG12 is not None else None
        self.FG14 = FG14[:, 0] if FG14 is not None else None
        self.map = Demod1R.T if Demod1R is not None else None
        self.tini = tini
        self.tread = tread
        self.pulse_dir = pulse_dir
        self.comp_fac = None
        self.comp_fac_y = None
        self.mode = mode
        self.dir = file_dir
        self.horizontal_lines = None
        self.vertical_lines = None
        self.transport_triangle = None
        self.blockade_triangle = None
        self.transport_mask = None
        self.transport_vertices = None
        self.centers = None
        self.regions = []
        self.region_boundaries = []

    # def add_triangle(self, lines=None):
    #     """
    #     Mask the two triangles defined by the intersecting points of the electron and hole resonances.
    #
    #     Parameters:
    #         lines (list of tuples): A list of four lines, each defined by slope-intercept form (m, b)
    #                                 for the equation y = mx + b.
    #     """
    #     if lines:
    #         self.lines = lines
    #
    #     if len(self.vertical_lines) == 2 and len(self.horizontal_lines) == 2:
    #         # Find intersection points of the lines
    #         intersections = [
    #             find_intersection(self.vertical_lines[i], self.horizontal_lines[j])
    #             for i in range(len(self.vertical_lines))
    #             for j in range(len(self.horizontal_lines))
    #             if find_intersection(self.vertical_lines[i], self.horizontal_lines[j])
    #         ]
    #
    #         # Filter unique intersection points
    #         intersections = list(set(intersections))
    #
    #         # Sort intersections to define the parallelogram
    #         parallelogram = sorted(intersections, key=lambda p: (p[0], p[1]))
    #
    #         # Define the two triangles by choosing a diagonal (e.g., between points 0 and 3)
    #         triangle1 = [parallelogram[0], parallelogram[1], parallelogram[2]]
    #         triangle2 = [parallelogram[1], parallelogram[2], parallelogram[3]]
    #
    #         # Create masks for transport and blockade triangles
    #         mask_blockade = np.zeros_like(self.map, dtype=float)
    #         mask_transport = np.zeros_like(self.map, dtype=float)
    #
    #         for i, x in enumerate(self.FG14):
    #             for j, y in enumerate(self.FG12):
    #                 if is_point_in_polygon(x, y, triangle1):
    #                     mask_blockade[j, i] = 1.0
    #                 if is_point_in_polygon(x, y, triangle2):
    #                     mask_transport[j, i] = 1.0
    #
    #         # Set values to NaN outside the defined regions
    #         self.blockade_triangle = np.array(self.map * mask_blockade)
    #         self.blockade_triangle[self.blockade_triangle == 0] = np.nan
    #         self.transport_triangle = np.array(self.map * mask_transport)
    #         self.transport_triangle[self.transport_triangle == 0] = np.nan
    #         self.transport_mask = mask_transport
    #         self.transport_vertices = triangle2
    #     else:
    #         # Find intersection points of the lines
    #         intersections = [
    #             find_intersection(self.vertical_lines[i], self.horizontal_lines[j])
    #             for i in range(len(self.vertical_lines))
    #             for j in range(len(self.horizontal_lines))
    #             if find_intersection(self.vertical_lines[i], self.horizontal_lines[j])
    #         ]
    #
    #         # Filter unique intersection points
    #         intersections = list(set(intersections))
    #         self.blockade_triangle = np.zeros_like(self.map)
    #         self.transport_triangle = np.zeros_like(self.map)
    #         self.transport_vertices = intersections

    def add_triangle(self, lines=None, offset=0):
        """
        Mask the two triangles defined by the intersecting points of the electron and hole resonances,
        with an optional offset to create a hexagonal blockade region.

        Parameters:
            lines (list of tuples): A list of four lines, each defined by slope-intercept form (m, b)
                                    for the equation y = mx + b.
            offset (float): Distance to offset from the diagonal shared by both triangles to form a hexagonal region.
        """
        if lines:
            self.lines = lines

        if len(self.vertical_lines) == 2 and len(self.horizontal_lines) == 2:
            # Find intersection points of the lines
            intersections = [
                find_intersection(self.vertical_lines[i], self.horizontal_lines[j])
                for i in range(len(self.vertical_lines))
                for j in range(len(self.horizontal_lines))
                if find_intersection(self.vertical_lines[i], self.horizontal_lines[j])
            ]

            # Filter unique intersection points
            intersections = list(set(intersections))

            # Sort intersections to define the parallelogram
            parallelogram = sorted(intersections, key=lambda p: (p[0], p[1]))

            # Define the two triangles by choosing a diagonal (e.g., between points 0 and 3)
            triangle1 = [parallelogram[0], parallelogram[1], parallelogram[2]]
            triangle2 = [parallelogram[1], parallelogram[2], parallelogram[3]]

            # Compute offset points for hexagonal cut
            if offset > 0:
                p1, p2 = parallelogram[1], parallelogram[2]
                if (p2[0] - p1[0]) != 0:
                    slope_main = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    intercept_main = p1[1] - slope_main * p1[0]
                else:
                    slope_main = None
                    intercept_main = p1[0]

                if slope_main is not None:
                    slope_offset = slope_main
                    intercept_offset = intercept_main + offset
                    offset_line = (slope_offset, intercept_offset)
                else:
                    x_offset = intercept_main + offset
                    offset_line = None

                hex_blockade = [p1, p2]
                for i in range(3):
                    edge_start, edge_end = triangle1[i], triangle1[(i + 1) % 3]
                    if (edge_end[0] - edge_start[0]) != 0:
                        slope_edge = (edge_end[1] - edge_start[1]) / (edge_end[0] - edge_start[0])
                        intercept_edge = edge_start[1] - slope_edge * edge_start[0]
                        edge_line = (slope_edge, intercept_edge)
                        intersection = find_intersection(offset_line, edge_line) if offset_line else None
                    else:
                        x_intersection = edge_start[0]
                        y_intersection = slope_offset * x_intersection + intercept_offset if offset_line else None
                        intersection = (x_intersection, y_intersection) if y_intersection is not None else None

                    if intersection:
                        x, y = intersection
                        if min(edge_start[0], edge_end[0]) <= x <= max(edge_start[0], edge_end[0]) and \
                                min(edge_start[1], edge_end[1]) <= y <= max(edge_start[1], edge_end[1]):
                            hex_blockade.append(intersection)

            else:
                hex_blockade = triangle1

            # Create masks for transport and blockade triangles
            mask_blockade = np.zeros_like(self.map, dtype=float)
            mask_transport = np.zeros_like(self.map, dtype=float)

            for i, x in enumerate(self.FG14):
                for j, y in enumerate(self.FG12):
                    if is_point_in_polygon(x, y, hex_blockade):
                        mask_blockade[j, i] = 1.0
                    if is_point_in_polygon(x, y, triangle2):
                        mask_transport[j, i] = 1.0

            # Set values to NaN outside the defined regions
            self.blockade_triangle = np.array(self.map * mask_blockade)
            self.blockade_triangle[self.blockade_triangle == 0] = np.nan
            self.transport_triangle = np.array(self.map * mask_transport)
            self.transport_triangle[self.transport_triangle == 0] = np.nan
            self.transport_mask = mask_transport
            self.transport_vertices = triangle2
        else:
            # Find intersection points of the lines
            intersections = [
                find_intersection(self.vertical_lines[i], self.horizontal_lines[j])
                for i in range(len(self.vertical_lines))
                for j in range(len(self.horizontal_lines))
                if find_intersection(self.vertical_lines[i], self.horizontal_lines[j])
            ]

            # Filter unique intersection points
            intersections = list(set(intersections))
            self.blockade_triangle = np.zeros_like(self.map)
            self.transport_triangle = np.zeros_like(self.map)
            self.transport_vertices = intersections

        #plt.figure()
        #plt.imshow(self.blockade_triangle)
        #plt.show()

    """
    def add_region(self, split_points):
        ""
        Add a custom region defined by vertices to the map.

        Parameters:
            split_points (list of tuples): List of (x, y) coordinates defining split points.
        ""
        if self.mode == 1:
            # Define the smaller triangle
            smaller_triangle = [self.transport_vertices[0], split_points[0], split_points[1]]
            region = np.zeros_like(self.map, dtype=float)
            for i, x in enumerate(self.FG14):
                for j, y in enumerate(self.FG12):
                    if is_point_in_polygon(x, y, smaller_triangle):
                        region[j, i] = 1.0
            self.regions.append(self.map * region)

        elif self.mode == 2:
            # Define the two quadrilaterals
            quadrilateral1 = [split_points[0], self.transport_vertices[1], self.transport_vertices[2], split_points[1]]
            quadrilateral2 = [split_points[4], split_points[2], split_points[3], split_points[5]]

            for quad in [quadrilateral1, quadrilateral2]:
                region = np.zeros_like(self.map, dtype=float)
                for i, x in enumerate(self.FG14):
                    for j, y in enumerate(self.FG12):
                        if is_point_in_polygon(x, y, quad):
                            region[j, i] = 1.0
                self.regions.append(self.map * region)

        elif self.mode == 3:
            # Define the smaller triangle and quadrilateral
            smaller_triangle = [self.transport_vertices[0], split_points[0], split_points[1]]
            quadrilateral = [split_points[2], self.transport_vertices[1], self.transport_vertices[2], split_points[3]]

            for poly in [quadrilateral, smaller_triangle]:
                region = np.zeros_like(self.map, dtype=float)
                for i, x in enumerate(self.FG14):
                    for j, y in enumerate(self.FG12):
                        if is_point_in_polygon(x, y, poly):
                            region[j, i] = 1.0
                self.regions.append(self.map * region)
        """

    def add_region(self, offset):
        """
        Add a custom region to the map, defined by the offset line
        intersecting the edges of a triangle.

        Parameters:
            offset (float): Offset to create the parallel line.
        """
        if self.transport_vertices is None or len(self.transport_vertices) < 3:
            print("Insufficient transport vertices to define a triangle.")
            return
        if self.region_boundaries:
            #triangle = [*self.region_boundaries[-1], self.transport_vertices[2]]
            p1, p2 = self.region_boundaries[-1][0], self.region_boundaries[-1][1]
        else:
            p1, p2 = self.transport_vertices[0], self.transport_vertices[1]

        triangle = self.transport_vertices  # Assume the first three vertices form the triangle

        # Define the main line equation (slope and intercept)
        if (p2[0] - p1[0]) != 0:
            slope_main = (p2[1] - p1[1]) / (p2[0] - p1[0])
            intercept_main = p1[1] - slope_main * p1[0]
        else:
            slope_main = None  # Vertical line
            intercept_main = p1[0]  # Use x-intercept for vertical line

        # Define the offset line
        if slope_main is not None:
            slope_offset = slope_main
            intercept_offset = intercept_main + offset
            offset_line = (slope_offset, intercept_offset)
        else:
            # Vertical line offset
            x_offset = intercept_main + offset
            offset_line = None

        # Find intersection points between the offset line and triangle edges
        intersections = []
        for i in range(3):
            edge_start, edge_end = triangle[i], triangle[(i + 1) % 3]

            # Calculate the slope and intercept of the current edge
            if (edge_end[0] - edge_start[0]) != 0:
                slope_edge = (edge_end[1] - edge_start[1]) / (edge_end[0] - edge_start[0])
                intercept_edge = edge_start[1] - slope_edge * edge_start[0]
                edge_line = (slope_edge, intercept_edge)

                # Find intersection
                if offset_line is not None:
                    intersection = find_intersection(offset_line, edge_line)
                else:
                    # Vertical line intersection
                    x_intersection = x_offset
                    y_intersection = slope_edge * x_intersection + intercept_edge
                    intersection = (x_intersection, y_intersection)
            else:
                # Edge is vertical
                x_intersection = edge_start[0]
                if offset_line is not None:
                    y_intersection = slope_offset * x_intersection + intercept_offset
                else:
                    # Both lines are vertical, no intersection
                    intersection = None
                    continue
                intersection = (x_intersection, y_intersection)

            # Check if intersection is within the bounds of the edge
            if intersection:
                x, y = intersection
                if min(edge_start[0], edge_end[0]) <= x <= max(edge_start[0], edge_end[0]) and \
                        min(edge_start[1], edge_end[1]) <= y <= max(edge_start[1], edge_end[1]):
                    intersections.append(intersection)

        # Ensure we have two valid intersection points
        if len(intersections) < 2:
            print("Offset line does not properly intersect the triangle.")
            return

        parallelogram = [p1, p2, intersections[0], intersections[1]]
        smaller_triangle = [p1, p2, self.transport_vertices[2]]
        #parallelogram = sorted(parallelogram, key=lambda p: (p[0], p[1]))

        if self.mode == 2:
            # Create region between the last region boundary and offset line
            region = np.zeros_like(self.map, dtype=float)
            for i, x in enumerate(self.FG14):
                for j, y in enumerate(self.FG12):
                    if is_point_in_polygon(x, y, parallelogram):
                        y_new = slope_offset * x + intercept_offset if slope_offset is not None else None
                        if y_new and min(intersections[0][1], intersections[1][1]) <= y <= max(intersections[0][1], y_new):
                            region[j, i] = 1.0

            # Update regions and map
            self.region_boundaries.append(intersections)
            self.regions.append(self.map * region)
            self.regions[-1][self.regions[-1] == 0] = np.nan

        elif self.mode == 3:
            region = np.zeros_like(self.map, dtype=float)
            for i, x in enumerate(self.FG14):
                for j, y in enumerate(self.FG12):
                    if is_point_in_polygon(x, y, smaller_triangle):
                        y_new = slope_offset * x + intercept_offset if slope_offset is not None else None
                        if y_new and min(intersections[0][1], intersections[1][1]) <= y <= max(intersections[0][1], y_new):
                            region[j, i] = 1.0

            # Update regions and map
            self.region_boundaries.append(intersections)
            self.transport_triangle = np.array(self.map * region)
            self.transport_triangle[self.transport_triangle == 0] = np.nan


    def get_ratio(self):
        """
        Calculate the ratio and uncertainties based on the mode.

        Returns:
            tuple: Depending on the mode, returns different ratio and uncertainty values.
        """
        blockade_flatten = self.blockade_triangle.flatten()
        if self.mode == 1 or self.mode == 3:
            transport_flatten = self.transport_triangle.flatten()

            # Calculate ratio and uncertainty
            # ratio = np.nanmean(transport_flatten) / np.nanmean(blockade_flatten)
            # uncertainty = np.sqrt(ratio ** 2 * (
            #         (np.nanstd(transport_flatten) / np.nanmean(transport_flatten)) ** 2 +
            #         (np.nanstd(blockade_flatten) / np.nanmean(blockade_flatten)) ** 2
            # ))

            mean_trans, std_trans, mean_block, std_block = self.make_histogram(transport_flatten, blockade_flatten)

            if self.pulse_dir == -1:
                ratio = mean_trans / mean_block
                uncertainty = np.sqrt(ratio ** 2 * (
                        (std_trans / mean_trans) ** 2 +
                        (std_block / mean_block) ** 2
                ))
            elif self.pulse_dir == 1:
                ratio = mean_block / mean_trans
                uncertainty = np.sqrt(ratio ** 2 * (
                        (std_trans / mean_trans) ** 2 +
                        (std_block / mean_block) ** 2
                ))


            print(f'Read-out time: {self.tread}')
            print('Blockade: ', np.round(mean_block, 3), " +- ",
                  np.round(std_block, 3))
            print('Non-Blockade: ', np.round(mean_trans, 3), " +- ",
                  np.round(std_trans, 3))

            return ratio, uncertainty

        elif self.mode == 2:
            ratios = []
            uncertainties = []
            for i, region in enumerate(self.regions):
                region_flatten = region.flatten()
                # region_mean = np.nanmean(region_flatten) / np.nanmean(blockade_flatten)
                # uncertainty = np.sqrt(region_mean ** 2 * (
                #         (np.nanstd(region_flatten) / np.nanmean(region_flatten)) ** 2 +
                #         (np.nanstd(blockade_flatten) / np.nanmean(blockade_flatten)) ** 2
                # ))
                #
                # ratios.append(region_mean)
                # uncertainties.append(uncertainty)
                mean_trans, std_trans, mean_block, std_block = self.make_histogram(region_flatten,
                                                                                   blockade_flatten, reg=i)

                ratio = mean_trans / mean_block
                uncertainty = np.sqrt(ratio ** 2 * (
                        (std_trans / mean_trans) ** 2 +
                        (std_block / mean_block) ** 2
                ))

                ratios.append(ratio)
                uncertainties.append(uncertainty)

                print(f'Read-out time: {self.tread}')
                print('Blockade: ', np.round(mean_block, 3), " +- ",
                      np.round(std_block, 3))
                print('Non-Blockade: ', np.round(mean_trans, 3), " +- ",
                      np.round(std_trans, 3))

            return ratios, uncertainties

    def make_histogram(self, data_trans, data_block, reg=0):
        # find parameters to fit a skewnorm to the data
        data_trans = data_trans[np.logical_not(np.isnan(data_trans))]
        data_block = data_block[np.logical_not(np.isnan(data_block))]

        shape_trans, loc_trans, scale_trans = skewnorm.fit(data_trans, 10, loc=0.5, scale=0.25)
        shape_block, loc_block, scale_block = skewnorm.fit(data_block, 10, loc=0.5, scale=0.25)

        # Compute mean and standard deviation using fitted parameters
        mean_trans = skewnorm.mean(shape_trans, loc=loc_trans, scale=scale_trans)
        std_trans = skewnorm.std(shape_trans, loc=loc_trans, scale=scale_trans)

        mean_block = skewnorm.mean(shape_block, loc=loc_block, scale=scale_block)
        std_block = skewnorm.std(shape_block, loc=loc_block, scale=scale_block)

        # Plot histogram and fitted distribution
        x_trans = np.linspace(min(data_trans), max(data_block), 300)
        x_block = np.linspace(min(data_trans), max(data_block), 300)
        pdf_trans = skewnorm.pdf(x_trans, shape_trans, loc=loc_trans, scale=scale_trans)
        pdf_block = skewnorm.pdf(x_block, shape_block, loc=loc_block, scale=scale_block)

        fig, axs = plt.subplots(1, 2)
        axs[0].hist(data_trans, bins=30, density=True)
        axs[1].hist(data_block, bins=30, density=True)
        x = np.linspace(10, 1, 300)
        axs[0].plot(x_trans, pdf_trans)
        axs[1].plot(x_block, pdf_block)
        axs[0].set_title('Transport')
        axs[1].set_title('Blockade')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, f'{self.pulse_dir}_{round(self.tread, 2)}_{reg}_hist.png'))
        plt.close()

        return mean_trans, std_trans, mean_block, std_block

    def save_map(self):
        np.save(os.path.join(self.dir, f'{self.pulse_dir}_{round(self.tread, 2)}_map.npy'), self.map)

        X, Y = np.meshgrid(self.FG14, self.FG12)

        np.save(os.path.join(self.dir, f'{self.pulse_dir}_{round(self.tread, 2)}_FG14.npy'), X)
        np.save(os.path.join(self.dir, f'{self.pulse_dir}_{round(self.tread, 2)}_FG12.npy'), Y)



    def plot_map(self):
        """
        Plot the map with optional regions using pcolormesh and include optional features like lines, vertices, and regions.

        Parameters:
            reg (bool): Whether to include regions in the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the 2D map using pcolormesh
        X, Y = np.meshgrid(self.FG14, self.FG12)
        c1 = ax.pcolormesh(X, Y, self.map, cmap="viridis_r", shading="auto", rasterized=True)#, vmin=0, vmax=1)

        # Add a colorbar to the figure
        #cbar_ax = fig.add_axes([0.85, 0.75, 0.03, 0.2])
        #cbar1 = fig.colorbar(c1, ax=ax, location='top', shrink=0.3, anchor=(1.0, 0.0), aspect=15, pad=0.01)
        #cbar1.set_ticks([0, 1])
        #cbar1.ax.tick_params(direction='in', width=1, length=15, labelsize=8)
        #cbar1.set_label("$R_{demod.} (a.u.)$", fontsize=18)
        #cbar1.ax.xaxis.set_label_position('bottom')

        ax.tick_params(axis='x', direction='in', length=8, labelsize=8, width=2)  # Move x ticks inside
        ax.tick_params(axis='y', direction='in', length=8, labelsize=8, width=2)  # Move y ticks inside

        # Set y-axis limits
        ax.set_ylim(np.min(self.FG12), np.max(self.FG12))
        ax.set_xlim(np.min(self.FG14), np.max(self.FG14))

        #ax.set_xticks([5.18, 5.19])
        #ax.set_yticks([5.27, 5.28])

        #ax.text(5.1825, 5.2925, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(self.tread * 125 * 1e-6, 2)),
        #        fontsize=8, color='white')

        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(2)

        # Set axis labels and title
        ax.set_xlabel(r"$FG_{1} (V)$", fontsize=9)
        ax.set_ylabel(r"$FG_{2} (V)$", fontsize=9)
        #ax.set_title("Original Map")

        # Display legend if any labels exist
        if ax.get_legend_handles_labels()[0]:  # Check if there are any legend entries
            ax.legend()

        # Adjust layout for better appearance
        plt.tight_layout()


        plt.savefig(os.path.join(self.dir, f'{self.pulse_dir}_{int(self.tread)}_thesis_map.svg'))

        # Add optional features (lines, vertices, regions)
        if self.vertical_lines or self.horizontal_lines:
            for m, b in self.vertical_lines:
                ax.plot(self.FG14, m * self.FG14 + b, label="Line")  # Plot lines using slope (m) and intercept (b)
            for m, b in self.horizontal_lines:
                ax.plot(self.FG14, m * self.FG14 + b, label="Line")  # Plot lines using slope (m) and intercept (b)

        if self.transport_vertices:
            vertices = np.array(self.transport_vertices)
            ax.scatter(vertices[:, 0], vertices[:, 1], color="red", label="Transport Vertices")  # Plot vertices
            ax.plot(vertices[:2, 0], vertices[:2, 1], "r--", label="Diagonal")

        if self.region_boundaries:
            for region in self.region_boundaries:
                vertices = np.array(region)  # Assuming regions store vertices as lists of (x, y)
                ax.plot(vertices[:, 0], vertices[:, 1], "r--", label="Region Boundary")  # Plot region boundaries

        if self.centers:
            for center in self.centers:
                #circle = plt.Circle(center, 0.05, fill=False, color="black", linewidth=2)
                #ax.add_patch(circle)  # Use add_patch instead of add_artist
                ax.scatter(center[0], center[1], color='red', label="Center")


        plt.savefig(os.path.join(self.dir, f'{self.pulse_dir}_{round(self.tread, 2)}_map.png'))
        plt.close()

        data = self.map
        n_peaks = 4
        bins = 256


        # Compute histogram
        hist, bin_edges = np.histogram(data.flatten(), bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Apply bin filtering
        #condition = np.where(hist > count_threshold / np.sum(hist))  # Normalize count threshold
        #bin_centers = bin_centers[condition]
        #hist = hist[condition]
        """
        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(data.flatten(), bins=bins, density=True, alpha=0.5, color='gray')
        plt.plot(bin_centers, hist, color='orangered')
        plt.axvline(0.1, color='orangered')
        plt.axvline(0.9, color='orangered')
        plt.xlabel("Intensity Value")
        plt.ylabel("Density")
        plt.title(f'{self.tread}_{self.pulse_dir}')
        #plt.legend()
        plt.grid()
        

        # Plot the histogram with KDE (continuous outline)
        sns.displot(data, kde=True, color='skyblue', bins=bins, kde_kws={'bw_method': 0.05})

        # Customize labels and title
        plt.title('Histogram with Continuous Outline (KDE)')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        """
        #plt.show()

    def detect_lines(self, slope_interval1=(-18, -8), slope_interval2=(-0.9, -0.6), slope_diag=(0.93, 1.08),
                     min_distance=0.0025, max_distance=0.006):
        """
        Detect lines and select four lines to form a parallelogram based on minimum distance.
        """

        # Apply edge detection
        edges = canny(self.map, sigma=0.8)  # Adjust sigma for edge sharpness

        # Apply probabilistic Hough Transform
        lines = probabilistic_hough_line(edges, threshold=10, line_length=10, line_gap=30)

        # Extract lines based on slope intervals
        lines_interval1 = []  # Red lines
        lines_interval2 = []  # Blue lines
        diagonal = []
        for line in lines:
            (x0, y0), (x1, y1) = line

            # Convert pixel coordinates to (X, Y)
            x0_proj = self.FG14[x0]
            y0_proj = self.FG12[y0]
            x1_proj = self.FG14[x1]
            y1_proj = self.FG12[y1]

            # Compute slope and intercept
            if x1_proj != x0_proj:  # Avoid division by zero
                slope = (y1_proj - y0_proj) / (x1_proj - x0_proj)
                intercept = y0_proj - slope * x0_proj

                # Classify lines into slope intervals
                if slope_interval1[0] <= slope <= slope_interval1[1]:
                    lines_interval1.append((slope, intercept))
                elif slope_interval2[0] <= slope <= slope_interval2[1]:
                    lines_interval2.append((slope, intercept))
                elif slope_diag[0] <= slope <= slope_diag[1]:
                    diagonal.append((slope, intercept))

        # Helper function to filter lines based on minimum distance
        def filter_lines_by_distance(lines, min_dist, max_dist):
            filtered_lines = []
            prev_vals = []
            for val, slope, intercept in lines:
                if not filtered_lines:
                    filtered_lines.append((slope, intercept))
                    prev_vals.append(val)
                else:
                    # Check distance to all previously added lines
                    distances = [abs(val - prev_val) for prev_val in
                                 prev_vals]
                    if all(d > min_dist and d < max_dist for d in distances):
                        filtered_lines.append((slope, intercept))
            return filtered_lines
        """
        plt.figure()
        for slope, intercept in diagonal:
            plt.scatter(slope, intercept)
        plt.title('diagonal')
        plt.show()
        """

        # Select two blue lines (interval 2) based on y-difference at FG14.min
        FG14_min = self.FG14[0]
        blue_lines = []
        if lines_interval2:
            distances = []
            # plt.figure()
            for slope, intercept in lines_interval2:
                y_val = slope * FG14_min + intercept
                distances.append((y_val, slope, intercept))
                # plt.scatter(y_val, slope)
            # plt.title(f'blue_{self.tread}')
            # plt.show()
            distances.sort()  # Sort by y-values
            filtered_lines = filter_lines_by_distance(
                [(val, slope, intercept) for val, slope, intercept in distances],
                min_distance, 0.01)
            if len(filtered_lines) >= 2:
                blue_lines = [filtered_lines[0], filtered_lines[-1]]
            else:
                blue_lines = filtered_lines

        # Select two red lines (interval 1) based on x-difference at FG12.max
        FG12_max = self.FG12[-1]
        red_lines = []
        if lines_interval1:
            distances = []
            # plt.figure()
            for slope, intercept in lines_interval1:
                x_val = (FG12_max - intercept) / slope
                distances.append((x_val, slope, intercept))
                # plt.scatter(x_val, slope)
            # plt.title(f'red_{self.tread}')
            # plt.show()
            distances.sort()  # Sort by x-values
            filtered_lines = filter_lines_by_distance(
                [(val, slope, intercept) for val, slope, intercept in distances],
                min_distance, max_distance)
            if len(filtered_lines) >= 2:
                red_lines = [filtered_lines[0], filtered_lines[-1]]
            else:
                red_lines = filtered_lines

        self.vertical_lines = red_lines
        self.horizontal_lines = blue_lines

        # Plot the edge map with lines
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the edge map using pcolormesh
        X, Y = np.meshgrid(self.FG14, self.FG12)
        ax.pcolormesh(X, Y, edges, cmap="gray", shading="auto")
        ax.set_title("Edge Map with Hough Lines")
        ax.set_xlabel("$FG_{14}$")
        ax.set_ylabel("$FG_{12}$")
        ax.set_ylim(np.min(self.FG12), np.max(self.FG12))

        # Plot blue lines (interval 2)
        for slope, intercept in blue_lines:
            y_vals = slope * self.FG14 + intercept
            ax.plot(self.FG14, y_vals, 'b-', label=f"Slope {slope:.2f}, Intercept {intercept:.2f}")

        # Plot red lines (interval 1)
        for slope, intercept in red_lines:
            y_vals = slope * self.FG14 + intercept
            ax.plot(self.FG14, y_vals, 'r-', label=f"Slope {slope:.2f}, Intercept {intercept:.2f}")

        for slope, intercept in diagonal:
            y_vals = slope * self.FG14 + intercept
            ax.plot(self.FG14, y_vals, 'g-', label=f"Slope {slope:.2f}, Intercept {intercept:.2f}")

        # Connect intersections to visualize the parallelogram
        if len(blue_lines) == 2 and len(red_lines) == 2:
            def compute_intersections(red_lines, blue_lines):
                """
                Compute the intersection points of the four lines forming a parallelogram.
                """
                intersections = []
                for line1 in red_lines:
                    for line2 in blue_lines:
                        slope1, intercept1 = line1
                        slope2, intercept2 = line2

                        if slope1 != slope2:  # Ensure lines are not parallel
                            x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
                            y_intersect = slope1 * x_intersect + intercept1
                            intersections.append([x_intersect, y_intersect])
                return intersections

            # Compute initial intersections
            intersections = compute_intersections(red_lines, blue_lines)
            intersections = np.array(intersections)

            # Plot intersections as a parallelogram
            if len(intersections) == 4:
                #intersections = np.array(intersections)
                ax.plot(
                    np.append(intersections[:, 0], intersections[0, 0]),
                    np.append(intersections[:, 1], intersections[0, 1]),
                    'g--', label="Parallelogram"
                )

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, f'{self.pulse_dir}_{int(self.tread)}_lines.png'))
        plt.close()

    def add_vertical_line(self, slope, x_val):
        """
        Add a new line to the lines list.

        Parameters:
            slope (float): Slope of the line (m in y = mx + b).
            intercept (float): Intercept of the line (b in y = mx + b).
        """
        if self.vertical_lines is None:
            self.vertical_lines = []
        intercept = self.FG12[-1] - x_val * slope
        self.vertical_lines.append((slope, intercept))
        print(f"Added line: y = {slope}x + {intercept}")

    def add_horizontal_line(self, slope, y_val):
        """
        Add a new line to the lines list.

        Parameters:
            slope (float): Slope of the line (m in y = mx + b).
            intercept (float): Intercept of the line (b in y = mx + b).
        """
        if self.horizontal_lines is None:
            self.horizontal_lines = []
        intercept = y_val - self.FG14[0] * slope
        self.horizontal_lines.append((slope, intercept))
        print(f"Added line: y = {slope}x + {intercept}")

    def move_vertical_line(self, index, new_slope, new_x_val):
        """
        Modify the slope and intercept of an existing vertical line.

        Parameters:
            index (int): Index of the vertical line to modify.
            new_slope (float): New slope for the line.
            new_intercept (float): New intercept for the line.
        """
        new_intercept = self.FG12[-1] - new_x_val * new_slope
        if self.vertical_lines is None:
            print("Invalid vertical line index. No changes made.")
            return
        elif index >= len(self.vertical_lines):
            self.vertical_lines.append((np.float64(new_slope), np.float64(new_intercept)))
        old_line = self.vertical_lines[index]
        self.vertical_lines[index] = (np.float64(new_slope), np.float64(new_intercept))
        print(
            f"Moved vertical line {index} from y = {old_line[0]}x + {old_line[1]} to y = {new_slope}x + {new_intercept}")

    def move_horizontal_line(self, index, new_slope, new_y_val):
        """
        Modify the slope and intercept of an existing horizontal line.

        Parameters:
            index (int): Index of the horizontal line to modify.
            new_slope (float): New slope for the line.
            new_intercept (float): New intercept for the line.
        """
        new_intercept = new_y_val - self.FG14[0] * new_slope
        if self.horizontal_lines is None:
            print("Invalid horizontal line index. No changes made.")
            return
        elif index >= len(self.horizontal_lines):
            self.horizontal_lines.append((np.float64(new_slope), np.float64(new_intercept)))
        else:
            self.horizontal_lines[index] = (np.float64(new_slope), np.float64(new_intercept))
        old_line = self.horizontal_lines[index]
        print(
            f"Moved horizontal line {index} from y = {old_line[0]}x + {old_line[1]} to y = {new_slope}x + {new_intercept}")

    def delete_vertical_line(self, index):
        """
        Remove a vertical line from the vertical lines list.

        Parameters:
            index (int): Index of the vertical line to remove.
        """
        if self.vertical_lines is None or index >= len(self.vertical_lines):
            print("Invalid vertical line index. No line deleted.")
            return
        removed_line = self.vertical_lines.pop(index)
        print(f"Deleted vertical line: y = {removed_line[0]}x + {removed_line[1]}")

    def delete_horizontal_line(self, index):
        """
        Remove a horizontal line from the horizontal lines list.

        Parameters:
            index (int): Index of the horizontal line to remove.
        """
        if self.horizontal_lines is None or index >= len(self.horizontal_lines):
            print("Invalid horizontal line index. No line deleted.")
            return
        removed_line = self.horizontal_lines.pop(index)
        print(f"Deleted horizontal line: y = {removed_line[0]}x + {removed_line[1]}")

    def get_vertical_lines(self):
        """
        Get the list of vertical lines.

        Returns:
            list of tuples: Each tuple represents a vertical line as (slope, intercept).
        """
        if self.vertical_lines is None:
            print("No vertical lines available.")
            return []
        return self.vertical_lines

    def get_horizontal_lines(self):
        """
        Get the list of horizontal lines.

        Returns:
            list of tuples: Each tuple represents a horizontal line as (slope, intercept).
        """
        if self.horizontal_lines is None:
            print("No horizontal lines available.")
            return []
        return self.horizontal_lines

    def subtract_background(self):
        """
        Subtract a polynomial background from the map and normalize the result.
        """

        # Define polynomial background subtraction
        #for i, col in enumerate(self.FG14):
        #    self.map[:, i] -= self.comp_fac * col
        #   self.map[:, i] = detrend(self.map[:, i])

        # Subtract median difference between consecutive rows
        self.map = correct_median_diff(self.map.T).T

        # Compute the derivative along FG14 (axis=1, as FG14 varies along columns)
        map_derivative = np.gradient(self.map, axis=1)#*10**4  # Derivative along FG14
        flattened_derivative = map_derivative.flatten()

        # Remove sharp jumps by thresholding extreme values (outliers)
        valid_derivatives = flattened_derivative[
            (flattened_derivative < np.percentile(flattened_derivative, 80)) &
            (flattened_derivative > np.percentile(flattened_derivative, 20))
            ]

        # Calculate the slope as the mean of the valid derivatives
        slope = np.median(valid_derivatives)
        print(slope)

        # Create a linear background to subtract
        if self.comp_fac and not self.comp_fac_y:
            FG14_slope = self.comp_fac * (self.FG14 - np.min(self.FG14))  # Linear trend along FG14
        elif self.comp_fac_y:
            FG14_slope = self.comp_fac * (self.FG14 - np.min(self.FG14))
            FG12_slope = self.comp_fac_y * (self.FG12 - np.min(self.FG12))
        else:
            FG14_slope = slope * (self.FG14 - np.min(self.FG14))  # Linear trend along FG14

        # Expand FG14 slope to 2D background
        background = np.tile(FG14_slope, (self.map.shape[0], 1))  # Repeat for all rows

        # If comp_fac_y is given, add FG12-based background component
        if self.comp_fac_y:
            FG12_slope_2D = np.tile(FG12_slope.reshape(-1, 1), (1, self.map.shape[1]))  # Repeat for all columns
            background += FG12_slope_2D  # Combine both directional backgrounds

        #plt.figure()
        #plt.plot(self.FG14, self.map[10, :])
        #plt.plot(self.FG14, average_slope * self.FG14 - 0.002)
        #plt.title('Original')
        #plt.show()

        # Subtract the background from the map
        self.map -= background

        # Remove outliers by clipping to a percentile range
        lower_percentile = np.percentile(self.map, 2)  # 2nd percentile
        upper_percentile = np.percentile(self.map, 98)  # 98th percentile
        self.map = np.clip(self.map, lower_percentile, upper_percentile)

        #plt.figure()
        #plt.plot(self.FG14, self.map[10, :])
        #plt.title('Substracted')
        #plt.show()

        # Apply Gaussian filter for smoothing (if needed)
        #self.map = gaussian_filter(self.map, sigma=(0.5, 0.5))

        # Normalize the map
        if self.centers:
            circle_min = []
            circle_max = []

            center_min, center_max = self.centers

            for i, x in enumerate(self.FG14):
                for j, y in enumerate(self.FG12):
                    if is_point_in_circle(x, y, center_min, 0.001):
                        circle_min.append(self.map[j, i])
                    if is_point_in_circle(x, y, center_max, 0.001):
                        circle_max.append(self.map[j, i])

            min_bin = np.mean(circle_min)
            max_bin = np.mean(circle_max)
            print(min_bin, max_bin)
        else:
            min_bin = np.min(self.map)
            max_bin = np.max(self.map)

        self.map = (self.map - min_bin) / (max_bin - min_bin)

    def set_comp_fac(self, comp_fac_x, comp_fac_y=None):
        self.comp_fac = comp_fac_x
        if comp_fac_y:
            self.comp_fac_y = comp_fac_y

    def set_centers(self, center_min, center_max):
        self.centers = [center_min, center_max]


def find_intersection(line1, line2):
    """
    Find the intersection point of two lines given in slope-intercept form.

    Parameters:
        line1 (tuple): (m1, b1) for the first line where y = m1 * x + b1.
        line2 (tuple): (m2, b2) for the second line where y = m2 * x + b2.

    Returns:
        tuple: (x, y) coordinates of the intersection point, or None if lines are parallel.
    """
    m1, b1 = line1
    m2, b2 = line2

    if m1 == m2:
        # Lines are parallel
        return None

    # Calculate intersection
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return x, y

def is_point_in_polygon(x, y, vertices):
    """
    Determine if a point is inside a polygon defined by its vertices.

    Parameters:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        vertices (list of tuples): List of vertices of the polygon.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    num_vertices = len(vertices)
    inside = False
    for i in range(num_vertices):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % num_vertices]
        if ((v1[1] > y) != (v2[1] > y)) and \
                (x < (v2[0] - v1[0]) * (y - v1[1]) / (v2[1] - v1[1]) + v1[0]):
            inside = not inside
    return inside


def get_value_range(data):
    hist, bins = np.histogram(data.flatten(), bins=1000)
    bins = bins[1:]

    count_threshold = 10
    condition = np.where(hist > count_threshold)
    bins_temp = bins[condition]

    range_width = abs(np.min(bins_temp) - np.max(bins_temp))
    pad = 0.1
    min_bin = np.min(bins_temp) - pad * range_width
    max_bin = np.max(bins_temp) + pad * range_width

    return min_bin, max_bin

def lorentzian(x, x0, gamma, A):
    """Single Lorentzian function."""
    return (A / np.pi) * (gamma / ((x - x0)**2 + gamma**2))

def multi_lorentzian(x, *params):
    """Sum of multiple Lorentzian functions."""
    n = len(params) // 3  # Each peak has (x0, gamma, A)
    y = np.zeros_like(x)
    for i in range(n):
        x0, gamma, A = params[3*i:3*i+3]
        y += lorentzian(x, x0, gamma, A)
    return y

def gaussian(x, x0, sigma, A):
    """Single Gaussian function."""
    return (A / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - x0)**2) / (2 * sigma**2))

def multi_gaussian(x, *params):
    """Sum of multiple Gaussian functions."""
    n = len(params) // 3  # Each peak has (x0, sigma, A)
    y = np.zeros_like(x)
    for i in range(n):
        x0, sigma, A = params[3*i:3*i+3]
        y += gaussian(x, x0, sigma, A)
    return y

def is_point_in_circle(x, y, center, radius):
    x_c, y_c = center
    # Calculate the Euclidean distance from the point to the center of the circle
    distance = math.sqrt((x - x_c)**2 + (y - y_c)**2)

    # If the distance is less than or equal to the radius, the point is inside or on the circle
    return distance <= radius


