import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from scipy.ndimage import gaussian_filter
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from sklearn.cluster import DBSCAN
from skimage.filters import threshold_otsu


class SingleMap:
    def __init__(self, FG12=None, FG14=None, Demod1R=None, tini=None, tread=None,
                 pulse_dir=None, comp_fac=None, mode=None, file_dir=None):
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
        self.dir = file_dir
        self.lines = None
        self.transport_triangle = None
        self.blockade_triangle = None
        self.transport_mask = None
        self.transport_vertices = None
        self.regions = []

    def add_triangle(self, lines=None):
        """
        Mask the two triangles defined by the intersecting points of the electron and hole resonances.

        Parameters:
            lines (list of tuples): A list of four lines, each defined by slope-intercept form (m, b)
                                    for the equation y = mx + b.
        """
        if lines:
            self.lines = lines

        # Find intersection points of the lines
        intersections = [
            find_intersection(self.lines[i], self.lines[j])
            for i in range(len(self.lines))
            for j in range(i + 1, len(self.lines))
            if find_intersection(self.lines[i], self.lines[j])
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

    def plot_map(self, reg=False):
        """
        Plot the map with optional regions using pcolormesh and include optional features like lines, vertices, and regions.

        Parameters:
            reg (bool): Whether to include regions in the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the 2D map using pcolormesh
        X, Y = np.meshgrid(self.FG14, self.FG12)
        c1 = ax.pcolormesh(X, Y, self.map, cmap="viridis_r", shading="auto", vmin=0, vmax=1)

        # Add a colorbar to the figure
        cbar1 = fig.colorbar(c1, ax=ax)
        cbar1.set_label("$U_{demod.} (a.u.)$")

        # Set y-axis limits
        ax.set_ylim(np.min(self.FG12), np.max(self.FG12))

        # Add optional features (lines, vertices, regions)
        if self.lines:
            for m, b in self.lines:
                ax.plot(self.FG14, m * self.FG14 + b, label="Line")  # Plot lines using slope (m) and intercept (b)

        if self.transport_vertices:
            vertices = np.array(self.transport_vertices)
            ax.scatter(vertices[:, 0], vertices[:, 1], color="red", label="Transport Vertices")  # Plot vertices

        if reg and hasattr(self, 'regions') and self.regions:
            for region in self.regions:
                vertices = np.array(region)  # Assuming regions store vertices as lists of (x, y)
                ax.plot(vertices[:, 0], vertices[:, 1], "r--", label="Region Boundary")  # Plot region boundaries

        # Set axis labels and title
        ax.set_xlabel("$FG_{14}$")
        ax.set_ylabel("$FG_{12}$")
        ax.set_title("Original Map")

        # Display legend if any labels exist
        if ax.get_legend_handles_labels()[0]:  # Check if there are any legend entries
            ax.legend()

        # Adjust layout for better appearance
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, f'{self.pulse_dir}_{np.round(self.tread)}_map.png'))
        plt.close()

    def detect_lines(self, slope_interval1=(-18, -10), slope_interval2=(-0.92, -0.7)):
        """
        Detect lines and select four lines to form a parallelogram based on minimum distance.
        """
        # Apply edge detection
        edges = canny(self.map, sigma=0.4)  # Adjust sigma for edge sharpness

        # Apply probabilistic Hough Transform
        lines = probabilistic_hough_line(edges, threshold=8, line_length=5, line_gap=60)

        # Extract lines based on slope intervals
        lines_interval1 = []  # Red lines
        lines_interval2 = []  # Blue lines
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

        # Select two blue lines (interval 2) based on y-difference at FG14.min
        FG14_min = self.FG14[0]
        blue_lines = []
        if lines_interval2:
            distances = []
            for slope, intercept in lines_interval2:
                y_val = slope * FG14_min + intercept
                distances.append((y_val, slope, intercept))
            distances.sort()  # Sort by y-values
            blue_lines.append((distances[0][1], distances[0][2]))  # Smallest y-value
            blue_lines.append((distances[-1][1], distances[-1][2]))  # Largest y-value

        # Select two red lines (interval 1) based on x-difference at FG12.max
        FG12_max = self.FG12[-1]
        red_lines = []
        if lines_interval1:
            distances = []
            for slope, intercept in lines_interval1:
                x_val = (FG12_max - intercept) / slope
                distances.append((x_val, slope, intercept))
            distances.sort()  # Sort by x-values
            red_lines.append((distances[0][1], distances[0][2]))  # Smallest x-value
            red_lines.append((distances[-1][1], distances[-1][2]))  # Largest x-value

        self.lines = [*red_lines, *blue_lines]

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
            ax.plot(self.FG14, y_vals, 'b-', label=f"Slope {slope:.2f}, Interval 2")

        # Plot red lines (interval 1)
        for slope, intercept in red_lines:
            y_vals = slope * self.FG14 + intercept
            ax.plot(self.FG14, y_vals, 'r-', label=f"Slope {slope:.2f}, Interval 1")

        # Connect intersections to visualize the parallelogram
        if len(blue_lines) == 2 and len(red_lines) == 2:
            intersections = []
            for line1 in red_lines:
                for line2 in blue_lines:
                    slope1, intercept1 = line1
                    slope2, intercept2 = line2

                    # Solve for intersection point
                    if slope1 != slope2:  # Ensure lines are not parallel
                        x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
                        y_intersect = slope1 * x_intersect + intercept1
                        intersections.append([x_intersect, y_intersect])

            # Plot intersections as a parallelogram
            if len(intersections) == 4:
                intersections = np.array(intersections)
                ax.plot(
                    np.append(intersections[:, 0], intersections[0, 0]),
                    np.append(intersections[:, 1], intersections[0, 1]),
                    'g--', label="Parallelogram"
                )

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, f'{self.pulse_dir}_{np.round(self.tread)}_lines.png'))
        plt.close()

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

