import numpy as np

class SingleMap():
    def __init__(self, FG12=None, FG14=None, Demod1R=None, tini=None, tread=None, pulse_dir=None, comp_fac=None):

        self.FG12 = FG12
        self.FG14 = FG14
        self.map = Demod1R
        self.tini = tini
        self.tread = tread
        self.pulse_dir = pulse_dir
        self.comp_fac = comp_fac
        self.transport_triangle = None
        self.blockade_triangle = None

    def substract_background(self):
        pass

    def find_intersection(self, line1, line2):
        """
        Find the intersection point of two lines.
        Each line is represented in slope-intercept form as (m, b) for y = mx + b.
        """
        m1, b1 = line1
        m2, b2 = line2

        if m1 == m2:
            return None  # Parallel lines do not intersect

        # Calculate intersection point
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x, y

    def add_triangle(self, lines):
        """
        Mask the two triangles defined by the intersecting points of the electron and hole resonances.

        Parameters:
            map (np.array): The input 2D numpy array.
            lines (list of tuples): A list of four lines, each defined by slope-intercept form (m, b)
                                for the equation y = mx + b
        Returns:
            np.array: A new array with the triangular region masked.
        """

        # Find intersection points of the lines
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                intersection = self.find_intersection(lines[i], lines[j])
                if intersection:
                    intersections.append(intersection)

        # Filter unique intersection points
        intersections = list(set(intersections))

        # Sort intersections to define the parallelogram
        parallelogram = sorted(intersections, key=lambda p: (p[0], p[1]))

        # Define the two triangles by choosing a diagonal (e.g., between points 0 and 3)
        triangle1 = [parallelogram[0], parallelogram[1], parallelogram[3]]
        triangle2 = [parallelogram[3], parallelogram[2], parallelogram[0]]

        # Create a mask
        mask_transport = np.zeros_like(self.map, dtype=bool)
        mask_blockade = np.zeros_like(self.map, dtype=bool)
        rows, cols = self.map.shape

        def is_point_in_triangle(x, y, triangle):
            """
            Check if a point (x, y) is inside a triangle defined by three vertices.
            """

            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

            b1 = sign((x, y), triangle[0], triangle[1]) < 0.0
            b2 = sign((x, y), triangle[1], triangle[2]) < 0.0
            b3 = sign((x, y), triangle[2], triangle[0]) < 0.0

            return (b1 == b2) and (b2 == b3)

        for i in range(rows):
            for j in range(cols):
                x, y = j, i
                if (0 <= x < cols and 0 <= y < rows) and (is_point_in_triangle(x, y, triangle1)):
                    mask_transport[i, j] = True
                if (0 <= x < cols and 0 <= y < rows) and (is_point_in_triangle(x, y, triangle2)):
                    mask_blockade[i, j] = True

        self.blockade_triangle = self.map[mask_blockade]
        self.transport_triangle = self.map[mask_transport]

        return mask_transport, mask_blockade

    def get_ratio(self):
        ratio = np.mean(self.blockade_triangle) / np.mean(self.transport_triangle)
        uncertainty = np.sqrt(ratio ** 2 * ((np.std(self.transport_triangle) / np.mean(self.transport_triangle)) ** 2 +
                                            (np.std(self.blockade_triangle) / np.mean(self.blockade_triangle)) ** 2))
        return ratio, uncertainty

    def add_region(self, points):
        if len(points) == 2:
            pl, pr = points
        if len(points) == 3:
            p1, p2, corner = points
        if len(points) == 4:
            pl1, pr1, pl2, pr2 = points

    def plot_map(self, reg=False):
        pass

