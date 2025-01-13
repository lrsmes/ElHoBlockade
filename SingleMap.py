import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter


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
        self.FG12 = FG12
        self.FG14 = FG14
        self.map = pd.DataFrame(Demod1R, index=FG14[:, 0], columns=FG12[0, :]).T
        self.tini = tini
        self.tread = tread
        self.pulse_dir = pulse_dir
        self.comp_fac = comp_fac
        self.mode = mode
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
        triangle2 = [parallelogram[3], parallelogram[2], parallelogram[0]]

        # Create masks for transport and blockade triangles
        mask_blockade = self.map.copy()
        mask_blockade[:] = self.map.index.map(
            lambda x: [float(is_point_in_polygon(x, y, triangle2)) for y in self.map.columns]
        )

        mask_transport = self.map.copy()
        mask_transport[:] = self.map.index.map(
            lambda x: [float(is_point_in_polygon(x, y, triangle1)) for y in self.map.columns]
        )

        # Set values to zero outside the defined regions
        self.blockade_triangle = self.map * mask_blockade
        self.transport_triangle = self.map * mask_transport
        self.transport_mask = mask_transport.astype('bool')
        self.transport_vertices = triangle2

    def add_region(self, split_points):
        """
        Add a custom region defined by vertices to the map.

        Parameters:
            vertices (list of tuples): List of (x, y) coordinates defining the region's vertices.
        """
        if self.mode == 1:
            # Define the smaller triangle
            smaller_triangle = [self.transport_vertices[0], split_points[0], split_points[1]]
            region = self.map.copy()
            region[:] = self.map.index.map(
                lambda x: [float(is_point_in_polygon(x, y, smaller_triangle)) for y in self.map.columns]
            )
            self.regions.append(self.map*region)
        elif self.mode == 2:
            # Define the two quadrilaterals
            quadrilateral1 = [split_points[0], self.transport_vertices[1], self.transport_vertices[2], split_points[1]]
            quadrilateral2 = [split_points[4], split_points[2], split_points[3], split_points[5]]
            region1 = self.map.copy()
            region1[:] = self.map.index.map(
                lambda x: [float(is_point_in_polygon(x, y, quadrilateral1)) for y in self.map.columns]
            )
            self.regions.append(self.map*region1)
            region2 = self.map.copy()
            region2[:] = self.map.index.map(
                lambda x: [float(is_point_in_polygon(x, y, quadrilateral2)) for y in self.map.columns]
            )
            self.regions.append(self.map*region2)
        elif self.mode == 3:
            # Define the smaller triangle and quadrilateral
            smaller_triangle = [self.transport_vertices[0], split_points[0], split_points[1]]
            quadrilateral = [split_points[2], self.transport_vertices[1], self.transport_vertices[2], split_points[3]]
            region1 = self.map.copy()
            region1[:] = self.map.index.map(
                lambda x: [float(is_point_in_polygon(x, y, quadrilateral)) for y in self.map.columns]
            )
            self.regions.append(self.map*region1)
            region2 = self.map.copy()
            region2[:] = self.map.index.map(
                lambda x: [float(is_point_in_polygon(x, y, smaller_triangle)) for y in self.map.columns]
            )
            self.regions.append(self.map*region2)

    def get_ratio(self):
        """
        Calculate the ratio and uncertainties based on the mode.

        Returns:
            tuple: Depending on the mode, returns different ratio and uncertainty values.
        """
        if self.mode == 1:
            ratio = self.blockade_triangle.mean().mean() / self.transport_triangle.mean().mean()
            uncertainty = np.sqrt(ratio ** 2 * (
                (self.transport_triangle.std().std() / self.transport_triangle.mean().mean()) ** 2 +
                (self.blockade_triangle.std().std() / self.blockade_triangle.mean().mean()) ** 2
            ))
            return ratio, uncertainty
        elif self.mode == 2 or self.mode == 3:
            ratios = []
            uncertainties = []
            for region in self.regions:
                region_mean = self.blockade_triangle.mean().mean() / region.mean().mean()
                ratios.append(region_mean)
                uncertainties.append(region.std().std() / region_mean)
            return ratios, uncertainties

    def plot_map(self, reg=False):
        """
        Plot the map with optional regions using seaborn.

        Parameters:
            reg (bool): Whether to include regions in the plot.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.map, cmap="viridis", cbar_kws={"label": "Intensity"}, vmin=0, vmax=1,
                    xticklabels=15, yticklabels=15, mask=self.transport_mask)
        plt.xlabel("$FG_{14}$")
        plt.ylabel("$FG_{12}$")
        plt.title("2D Map Visualization")
        if reg and self.regions:
            for region in self.regions:
                plt.plot([v[0] for v in region], [v[1] for v in region], "r--")
        plt.show()

    def substract_background(self):
        """
        Subtract a polynomial background from the map and normalize the result.
        """
        for col in self.map.columns:
            self.map[col] -= 0.0095 * col
        #self.map = self.map.apply(lambda x: gaussian_filter(x, sigma=(0.5, 0.5)), axis=0)
        min_bin, max_bin = get_value_range(self.map.values)
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

