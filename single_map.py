import numpy as np
import matplotlib.pyplot as plt
from Data_analysis_and_transforms import evaluate_poly_background_2d

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
        self.map = Demod1R
        self.tini = tini
        self.tread = tread
        self.pulse_dir = pulse_dir
        self.comp_fac = comp_fac
        self.mode = mode
        self.transport_triangle = None
        self.blockade_triangle = None
        self.transport_vertices = None

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
        triangle1 = [parallelogram[0], parallelogram[1], parallelogram[3]]
        triangle2 = [parallelogram[3], parallelogram[2], parallelogram[0]]

        # Create a mask for transport and blockade triangles
        rows, cols = self.map.shape
        y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        points = np.stack((x.ravel(), y.ravel()), axis=-1)

        mask_transport = np.array([is_point_in_triangle(px, py, triangle1) for px, py in points]).reshape(rows, cols)
        mask_blockade = np.array([is_point_in_triangle(px, py, triangle2) for px, py in points]).reshape(rows, cols)

        self.blockade_triangle = self.map[mask_blockade]
        self.transport_triangle = self.map[mask_transport]
        self.transport_vertices = triangle2

    def get_ratio(self):
        """
        Calculate the ratio and uncertainties based on the mode.

        Returns:
            tuple: Depending on the mode, returns different ratio and uncertainty values.
        """
        if self.mode == 1:
            ratio = np.mean(self.blockade_triangle) / np.mean(self.transport_triangle)
            uncertainty = np.sqrt(ratio ** 2 * ((np.std(self.transport_triangle) / np.mean(self.transport_triangle)) ** 2 +
                                                (np.std(self.blockade_triangle) / np.mean(self.blockade_triangle)) ** 2))
            return ratio, uncertainty
        elif self.mode == 2:
            ratio_reg1 = np.mean(self.blockade_triangle) / np.mean(self.transport_region1)
            uncertainty_reg1 = np.sqrt(ratio_reg1 ** 2 * ((np.std(self.transport_region1) / np.mean(self.transport_region1)) ** 2 +
                                                          (np.std(self.blockade_triangle) / np.mean(self.blockade_triangle)) ** 2))
            ratio_reg2 = np.mean(self.blockade_triangle) / np.mean(self.transport_region2)
            uncertainty_reg2 = np.sqrt(ratio_reg2 ** 2 * ((np.std(self.transport_region2) / np.mean(self.transport_region2)) ** 2 +
                                                          (np.std(self.blockade_triangle) / np.mean(self.blockade_triangle)) ** 2))
            return ratio_reg1, uncertainty_reg1, ratio_reg2, uncertainty_reg2
        elif self.mode == 3:
            ratio_triangle = np.mean(self.blockade_triangle) / np.mean(self.transport_triangle)
            uncertainty_triangle = np.sqrt(ratio_triangle ** 2 * ((np.std(self.transport_triangle) / np.mean(self.transport_triangle)) ** 2 +
                                                                  (np.std(self.blockade_triangle) / np.mean(self.blockade_triangle)) ** 2))
            ratio_reg1 = np.mean(self.blockade_triangle) / np.mean(self.transport_region1)
            uncertainty_reg1 = np.sqrt(ratio_reg1 ** 2 * ((np.std(self.transport_region1) / np.mean(self.transport_region1)) ** 2 +
                                                          (np.std(self.blockade_triangle) / np.mean(self.blockade_triangle)) ** 2))
            return ratio_triangle, uncertainty_triangle, ratio_reg1, uncertainty_reg1

    def add_region(self, split_points):
        """
        Divide the transport triangle into smaller regions based on split points.

        Parameters:
            split_points (list of tuples): Points used to divide the regions.
        """
        rows, cols = self.transport_triangle.shape
        y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        points = np.stack((x.ravel(), y.ravel()), axis=-1)

        if self.mode == 1:
            # Define the smaller triangle
            smaller_triangle = [self.transport_vertices[0], split_points[0], split_points[1]]

            triangle_mask = np.array([is_point_in_triangle(px, py, smaller_triangle) for px, py in points]).reshape(rows, cols)
            self.transport_triangle = triangle_mask

        elif self.mode == 2:
            # Define the two quadrilaterals
            quadrilateral1 = [split_points[0], self.transport_vertices[1], self.transport_vertices[2], split_points[1]]
            quadrilateral2 = [split_points[4], split_points[2], split_points[3], split_points[5]]

            quadrilateral1_mask = np.array([is_point_in_quadrilateral(px, py, quadrilateral1) for px, py in points]).reshape(rows, cols)
            quadrilateral2_mask = np.array([is_point_in_quadrilateral(px, py, quadrilateral2) for px, py in points]).reshape(rows, cols)

            self.transport_region1 = quadrilateral1_mask
            self.transport_region2 = quadrilateral2_mask

        elif self.mode == 3:
            # Define the smaller triangle and quadrilateral
            smaller_triangle = [self.transport_vertices[0], split_points[0], split_points[1]]
            quadrilateral = [split_points[2], self.transport_vertices[1], self.transport_vertices[2], split_points[3]]

            triangle_mask = np.array([is_point_in_triangle(px, py, smaller_triangle) for px, py in points]).reshape(rows, cols)
            quadrilateral_mask = np.array([is_point_in_quadrilateral(px, py, quadrilateral) for px, py in points]).reshape(rows, cols)

            self.transport_region1 = quadrilateral_mask
            self.transport_triangle = triangle_mask

    def plot_map(self, reg=False):
        """
        Plot the map with optional regions.

        Parameters:
            reg (bool): Whether to include regions in the plot.
        """
        fig = plt.figure(figsize=(12,12))
        im = plt.imshow(self.map, vmin=0, vmax=1)
        plt.colorbar('virdis_r')
        plt.xlabel('$FG_12$')
        plt.ylabel('$FG_14$')
        plt.xticks(ticks=range(self.map.shape[0]), labels=self.FG12[::20])
        plt.xticks(ticks=range(self.map.shape[1]), labels=self.FG14[::20])

        return fig

    def substract_background(self):
        """
        Subtracts a polynomial background from the map and normalizes the result.
        """
        # Substract background
        background = evaluate_poly_background_2d(self.FG12, self.FG14, self.map, 1, 0)
        self.map = self.map - background
        # Normalize
        min_bin, max_bin = get_value_range(self.map)
        self.map = (self.map - min_bin) / (max_bin - min_bin)


def find_intersection(line1, line2):
    """
    Find the intersection point of two lines.
    Each line is represented in slope-intercept form as (m, b) for y = mx + b.

    Parameters:
        line1 (tuple): Coefficients (m, b) of the first line.
        line2 (tuple): Coefficients (m, b) of the second line.

    Returns:
        tuple: Intersection point (x, y) or None if lines are parallel.
    """
    m1, b1 = line1
    m2, b2 = line2

    if m1 == m2:
        return None  # Parallel lines do not intersect

    # Calculate intersection point
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y


def is_point_in_triangle(x, y, triangle):
    """
    Check if a point (x, y) is inside a triangle defined by three vertices.

    Parameters:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        triangle (list of tuples): List of three vertices defining the triangle.

    Returns:
        bool: True if the point is inside the triangle, False otherwise.
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign((x, y), triangle[0], triangle[1]) < 0.0
    b2 = sign((x, y), triangle[1], triangle[2]) < 0.0
    b3 = sign((x, y), triangle[2], triangle[0]) < 0.0

    return (b1 == b2) and (b2 == b3)


def is_point_in_quadrilateral(x, y, quadrilateral):
    """
    Check if a point (x, y) is inside a quadrilateral defined by four vertices.
    The quadrilateral is treated as two triangles.

    Parameters:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        quadrilateral (list of tuples): List of four vertices defining the quadrilateral.

    Returns:
        bool: True if the point is inside the quadrilateral, False otherwise.
    """
    triangle1 = [quadrilateral[0], quadrilateral[1], quadrilateral[2]]
    triangle2 = [quadrilateral[2], quadrilateral[3], quadrilateral[0]]
    return is_point_in_triangle(x, y, triangle1) or is_point_in_triangle(x, y, triangle2)

def get_value_range(map):
    hist, bins = np.histogram(map.flatten(), bins=1000)
    bins = bins[1::]

    count_threshold = 10
    condition = np.where(hist > count_threshold)
    bins_temp = bins[condition]

    range_width = abs(np.min(bins_temp) - np.max(bins_temp))
    pad = 0.1
    min_bin = np.min(bins_temp) - pad * range_width
    max_bin = np.max(bins_temp) + pad * range_width

    return min_bin, max_bin
