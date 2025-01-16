import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.ndimage import gaussian_filter
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from sklearn.cluster import DBSCAN
from skimage.filters import threshold_otsu


class SingleMap:
    def __init__(self, FG12=None, FG14=None, Demod1R=None, tini=None, tread=None, pulse_dir=None, comp_fac=None, mode=None):
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
        self.comp_fac = comp_fac
        self.mode = mode
        self.lines = None
        self.transport_triangle = None
        self.blockade_triangle = None
        self.transport_mask = None
        self.transport_vertices = None
        self.regions = []

    def add_triangle(self, lines):
        """
        Mask the two triangles defined by the intersecting points of the electron and hole resonances.

        Parameters:
            lines (list of tuples): A list of four lines, each defined by slope-intercept form (m, b)
                                    for the equation y = mx + b.
        """
        self.lines = lines

        # Find intersection points of the lines
        intersections = [
            find_intersection(lines[i], lines[j])
            for i in range(len(lines))
            for j in range(i + 1, len(lines))
            if find_intersection(lines[i], lines[j])
        ]

        # Filter unique intersection points
        intersections = list(set(intersections))

        # Sort intersections to define the parallelogram
        parallelogram = sorted(intersections, key=lambda p: (p[0], p[1]))

        # Define the two triangles by choosing a diagonal (e.g., between points 0 and 3)
        triangle1 = [parallelogram[0], parallelogram[1], parallelogram[2]]
        triangle2 = [parallelogram[1], parallelogram[2], parallelogram[3]]

        # Create masks for transport and blockade triangles
        mask_blockade = np.zeros_like(self.map, dtype=float)
        mask_transport = np.zeros_like(self.map, dtype=float)

        for i, x in enumerate(self.FG14):
            for j, y in enumerate(self.FG12):
                if is_point_in_polygon(x, y, triangle1):
                    mask_blockade[j, i] = 1.0
                if is_point_in_polygon(x, y, triangle2):
                    mask_transport[j, i] = 1.0

        # Set values to zero outside the defined regions
        self.blockade_triangle = self.map * mask_blockade
        self.transport_triangle = self.map * mask_transport
        self.transport_mask = mask_transport
        self.transport_vertices = triangle2

    def add_region(self, split_points):
        """
        Add a custom region defined by vertices to the map.

        Parameters:
            split_points (list of tuples): List of (x, y) coordinates defining split points.
        """
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

    def get_ratio(self):
        """
        Calculate the ratio and uncertainties based on the mode.

        Returns:
            tuple: Depending on the mode, returns different ratio and uncertainty values.
        """
        if self.mode == 1:
            ratio = np.mean(self.transport_triangle) / np.mean(self.blockade_triangle)
            uncertainty = np.sqrt(ratio**2 * (
                (np.std(self.transport_triangle) / np.mean(self.transport_triangle))**2 +
                (np.std(self.blockade_triangle) / np.mean(self.blockade_triangle))**2
            ))
            return ratio, uncertainty

        elif self.mode in [2, 3]:
            ratios = []
            uncertainties = []
            for region in self.regions:
                region_mean = np.mean(self.blockade_triangle) / np.mean(region)
                ratios.append(region_mean)
                uncertainties.append(np.std(region) / np.mean(region))
            return ratios, uncertainties

    def plot_map(self, reg=False, slope_interval=(-18, 1), eps_slope=0.2, eps_intercept=0.1):
        """
        Plot the map with optional regions using pcolormesh and include a gradient map.

        Parameters:
            reg (bool): Whether to include regions in the plot.
            slope_interval (tuple): A tuple specifying the acceptable range of slopes for detected lines (min_slope, max_slope).
            eps_slope (float): DBSCAN parameter for clustering similar slopes.
            eps_intercept (float): DBSCAN parameter for clustering similar intercepts.
        """
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 3]})

        # First subplot: Original 2D map
        ax1 = axs[0]
        X, Y = np.meshgrid(self.FG14, self.FG12)
        c1 = ax1.pcolormesh(X, Y, self.map, cmap="viridis_r", shading="auto", vmin=0, vmax=1)
        cbar1 = fig.colorbar(c1, ax=ax1)
        cbar1.set_label("$U_{demod.} (a.u.)$")
        ax1.set_ylim(np.min(self.FG12), np.max(self.FG12))

        # Add optional features (lines, vertices, regions)
        if hasattr(self, 'lines') and self.lines:
            for m, b in self.lines:
                ax1.plot(self.FG14, m * self.FG14 + b, label="Line")

        if self.transport_vertices:
            vertices = np.array(self.transport_vertices)
            ax1.scatter(vertices[:, 0], vertices[:, 1], color="red", label="Transport Vertices")

        if reg and self.regions:
            for region in self.regions:
                vertices = np.array(region)  # Assuming regions store vertices as lists of (x, y)
                ax1.plot(vertices[:, 0], vertices[:, 1], "r--", label="Region Boundary")

        ax1.set_xlabel("$FG_{14}$")
        ax1.set_ylabel("$FG_{12}$")
        ax1.set_title("Original Map")

        # Second subplot: Grayscale gradient map with edges and filtered Hough lines
        ax2 = axs[1]

        # Apply edge detection
        edges = canny(self.map, sigma=0.6)  # Adjust sigma for edge sharpness

        # Plot edges using pcolormesh
        ax2.pcolormesh(X, Y, edges, cmap="gray", shading="auto")
        ax2.set_title("Grayscale Gradient Map with Filtered Hough Transform")
        ax2.set_xlabel("$FG_{14}$")
        ax2.set_ylabel("$FG_{12}$")

        # Apply probabilistic Hough Transform
        lines = probabilistic_hough_line(edges, threshold=55, line_length=35, line_gap=30)

        # Extract slope and intercept for each line
        slopes_intercepts = []
        line_coordinates = []
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

                # Filter lines based on slope
                if slope_interval[0] <= slope <= slope_interval[1]:
                    slopes_intercepts.append([slope, intercept])
                    line_coordinates.append(((x0_proj, y0_proj), (x1_proj, y1_proj)))

        # Cluster lines using DBSCAN
        if slopes_intercepts:
            slopes_intercepts = np.array(slopes_intercepts)
            dbscan = DBSCAN(eps=max(eps_slope, eps_intercept), min_samples=1).fit(slopes_intercepts)

            # Randomly select one line from each cluster
            unique_labels = set(dbscan.labels_)
            selected_lines = []
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                cluster_indices = np.where(dbscan.labels_ == label)[0]
                random_index = random.choice(cluster_indices)
                selected_lines.append(slopes_intercepts[random_index])

            # Plot selected lines
            print(selected_lines)
            for slope, intercept in #selected_lines:
                x_vals = np.linspace(np.min(self.FG14), np.max(self.FG14), 500)
                y_vals = slope * x_vals + intercept
                ax2.plot(x_vals, y_vals, 'r-', label="Selected Line")

        ax2.set_ylim(np.min(self.FG12), np.max(self.FG12))
        ax2.set_xlim(np.min(self.FG14), np.max(self.FG14))

        # Tight layout for better appearance
        plt.tight_layout()
        plt.show()

    def detect_lines(self, slope_interval=(-18, 1), eps_slope=0.1, eps_intercept=0.1):
        # Apply edge detection
        edges = canny(self.map, sigma=0.6)  # Adjust sigma for edge sharpness

        # Apply probabilistic Hough Transform
        lines = probabilistic_hough_line(edges, threshold=35, line_length=25, line_gap=30)

        # Extract slope and intercept for each line
        slopes_intercepts = []
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

                # Filter lines based on slope
                if slope_interval[0] <= slope <= slope_interval[1]:
                    slopes_intercepts.append([slope, intercept])

        # Cluster lines using DBSCAN
        if slopes_intercepts:
            slopes_intercepts = np.array(slopes_intercepts)
            dbscan = DBSCAN(eps=max(eps_slope, eps_intercept), min_samples=1).fit(slopes_intercepts)

            # Randomly select one line from each cluster
            unique_labels = set(dbscan.labels_)
            selected_lines = []
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                cluster_indices = np.where(dbscan.labels_ == label)[0]
                random_index = random.choice(cluster_indices)
                selected_lines.append(slopes_intercepts[random_index])



    def subtract_background(self):
        """
        Subtract a polynomial background from the map and normalize the result.
        """
        # Define polynomial background subtraction
        for i, col in enumerate(self.FG12):
            self.map[:, i] -= 0.0095 * col

        # Apply Gaussian filter for smoothing (if needed)
        self.map = gaussian_filter(self.map, sigma=(0.5, 0.5))

        # Remove outliers by clipping to a percentile range
        lower_percentile = np.percentile(self.map, 2)  # 2nd percentile
        upper_percentile = np.percentile(self.map, 98)  # 98th percentile
        self.map = np.clip(self.map, lower_percentile, upper_percentile)

        # Normalize the map
        min_bin = np.min(self.map)
        max_bin = np.max(self.map)
        self.map = (self.map - min_bin) / (max_bin - min_bin)


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

