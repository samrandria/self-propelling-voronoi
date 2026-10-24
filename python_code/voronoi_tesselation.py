"""
Voronoi tesselation of space using scipy

Use the Qhull library (can compute Delaunay triangulation and Voronoi diagram).
Cannot compute Voronoi volumes or weighted Voronoi diagrams.
"""

import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
from scipy.spatial import Voronoi, ConvexHull
import numpy as np
# Note that "random" in python already implements the Mersenne Twister algorithm
import random
# from flocking_confluent_marchetti import get_cell_energy, get_tissue_energy, compute_force_every_cell

###### Important ######
# Attribute ridge_vertices : each element in the list is a pair of integers
# Each integer is an index into the "vertices" list
# So each element defines a line to be drawn.
# SEE BELOW AN EXAMPLE
# # for index_pair in vor.ridge_vertices:
#     if index_pair[0] >= 0 and index_pair[1] >= 0:
#         v0 = vor.vertices[index_pair[0]]
#         v1 = vor.vertices[index_pair[1]]
#         # Draw a line from v0 to v1.
#         plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)

def voronoi_finite_polygons_2d(vor, point_number=None, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    point_number : int
        Limits the number of regions to sort. If None is given, the sorting algorithm will run through all the regions, including the ones outside the observation box

    Returns
    -------
    regions : list of lists
        Indices of vertices in each revised Voronoi regions.
        Vertices are sorted counter-clockwise.
    vertices : list of lists
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # XXX : Removed the somehow buggy arctan sorting, let's see how it works.
        # finish
        new_regions.append(new_region)

    ### FIXME : check if every polygon is properly sorted counterclockwise.
    if point_number is None :
        # Run against all regions.
        sorted_counterclockwise_truth_table = is_sorted_counterclockwise(new_regions, np.asarray(new_vertices))
        new_regions = sort_vertices_counterclockwise(new_regions, np.asarray(new_vertices), sorted_counterclockwise_truth_table)
    else:
        sorted_counterclockwise_truth_table = is_sorted_counterclockwise(new_regions[0:point_number], np.asarray(new_vertices))
        new_regions = sort_vertices_counterclockwise(new_regions[0:point_number], np.asarray(new_vertices), sorted_counterclockwise_truth_table)

    return new_regions, np.asarray(new_vertices)





def is_sorted_counterclockwise(regions, vertices):
    """
    Compute the determinant of the orientation matrix for three vertices per region of a simple convex polygon.
    If positive, the polygon is sorted counterclockwise. Otherwise it is sorted clockwise (assuming there is self intersection or hole in the polygon)

    Parameters
    -----
    regions: list of lists
        Indices of vertices in each revised Voronoi regions.
    vertices : numpy array
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Returns
    -----
    sorted_counterclockwise_truth_table: np array
        Contains True if the determinant is positive (and the polygon is sorted counter clockwise), False otherwise
    """

    # Get 3 points by region
#    points_list = np.zeros([len(regions) * 3, 2])
    determinants_table = np.zeros(len(regions))
    for index, region in enumerate(regions):
        # FIXME (maybe) : Mnemonics for easier manipulation, maybe costly. Replace them with direct call to vertices if needed.
        x_A = vertices[region, 0][0]
        x_B = vertices[region, 0][1]
        x_C = vertices[region, 0][2]
        y_A = vertices[region, 1][0]
        y_B = vertices[region, 1][1]
        y_C = vertices[region, 1][2]
        determinants_table[index] = (x_B - x_A) * (y_C - y_A) - (x_C - x_A) * (y_B - y_A)

    # Puts True if positive, False otherwise
    # If the determinant is zero, this would mean that the three vertices are on a line.
#    sorted_counterclockwise_truth_table = np.greater_equal(determinants_table,0)
    sorted_counterclockwise_truth_table = determinants_table >= 0

    return sorted_counterclockwise_truth_table


def sort_vertices_counterclockwise(regions, vertices, sorted_counterclockwise_truth_table):
    """
    Reverse the regions where sorted_counterclockwise_truth_table is false.

    Parameters
    -----
    regions: list of lists
        Indices of vertices in each revised Voronoi regions.
    vertices : numpy array
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    sorted_counterclockwise_truth_table: np array
        Contains True if the determinant is positive (and the polygon is sorted counter clockwise), False otherwise

    Returns
    -----
    regions_sorted_counterclockwise: list of lists
        Indices of vertices in each revised Voronoi regions sorted counter clockwisely
    """

    regions_sorted_counterclockwise = [ regions[i][::-1] if not sorted_counterclockwise_truth_table[i] else regions[i] for i in range(len(regions))]

    return regions_sorted_counterclockwise




def plot_polygons(vor, regions, vertices):
    """
    Graphical function to plot polygons by circling through their vertices.

    Note that while the Voronoi regions are finite, they may still go beyond the box

    Parameters
    -----
    vor : A Voronoi object
    regions : list of lists
        Indices of vertices in each revised Voronoi regions.
    vertices : list of lists
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """
    for region in regions:
        # Build a container with n+1 points to plot the last edge between the nth and the 0th point
        polygon = np.zeros([len(vertices[region])+1,2])
        polygon[0:len(vertices[region])] = vertices[region][:]
        # Append the 0th point at the end
        polygon[-1] = vertices[region][0]
        # plt.fill(*zip(*polygon), alpha=0.4) # Color the polygons
        plt.plot(polygon[:,0], polygon[:,1], '--k')





# def get_polygons_properties_2d(regions, vertices):
#     """
#     Extract the perimeter and area for each finite polygon in 2D

#     Parameters
#     -----
#     regions : list of lists
#         Indices of vertices in each revised Voronoi regions.
#     vertices : list of lists
#         Coordinates for revised Voronoi vertices. Same as coordinates
#         of input vertices, with 'points at infinity' appended to the
#         end.

#     Returns
#     -----
#     properties_index : list
#         List of index for each region in regions treated (probably not useful since perimeter and area are
#         already acquired in the order of region in regions)
#     properties_perimeter : list of float
#         Perimeter of each region in regions treated
#     properties_area : list of float
#         Area of each region in regions treated
#         Warning : In 2D : ConvexHull.area is the perimeter, while ConvexHull.volume is the area
#     """

#     properties_index = []
#     properties_perimeter = []
#     properties_area = []


#     index = 0
#     for index, region in enumerate(regions):
#         polygon = vertices[region]
#         properties_index.append(index)
#         properties_perimeter.append(ConvexHull(polygon).area)
#         properties_area.append(ConvexHull(polygon).volume)

#     return properties_index,properties_perimeter,properties_area

# XXX : use the function that doesn't involve Qhull
#def get_polygons_properties_2d(regions, vertices):
def get_polygons_properties_2d_Qhull(regions, vertices):
    """
    Extract the perimeter and area for each finite polygon in 2D
    Partially Cython typed.

    Parameters
    -----
    regions : list of lists
        Indices of vertices in each revised Voronoi regions.
    vertices : numpy array
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Returns
    -----
    properties_index : numpy array
        List of index for each region in regions treated (probably not useful since perimeter and area are
        already acquired in the order of region in regions)
    properties_perimeter : numpy array
        Perimeter of each region in regions treated
    properties_area : numpy array
        Area of each region in regions treated
        Warning : In 2D : ConvexHull.area is the perimeter, while ConvexHull.volume is the area
    """

    ## Typing variables
    cdef int len_regions = len(regions)
    cdef Py_ssize_t index

    properties_index_np = np.zeros(len_regions, dtype=np.int)
    properties_perimeter_np = np.zeros(len_regions, dtype=np.float64)
    properties_area_np = np.zeros(len_regions, dtype=np.float64)

    cdef Py_ssize_t[:] properties_index = properties_index_np
    cdef double[:] properties_perimeter = properties_perimeter_np
    cdef double[:] properties_area = properties_area_np
    cdef double convHull_area
    cdef double convHull_volume
    # cdef double[:,:] polygon

    for index in range(len_regions):
        polygon = vertices[regions[index]]
        properties_index[index] = index
        convHull = ConvexHull(polygon)
        convHull_area = convHull.area
        convHull_volume = convHull.volume
        properties_perimeter[index] = convHull_area
        properties_area[index] = convHull_volume

    return properties_index,properties_perimeter,properties_area

def get_polygons_properties_2d(regions, vertices):
    """
    Computes the perimeters and the areas of every cell defined by the regions and vertices passed as parameters

    Parameters
    -----
    regions : list of lists
        Indices of vertices in each revised Voronoi regions.
    vertices : numpy array
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Returns
    -----
    None : None
        Placeholder for backward compatibility
    perimeters : numpy array
        Perimeter of each region in regions treated
    areas : numpy array
        Area of each region in regions treated
    """

    areas = np.zeros(len(regions))
    perimeters = np.zeros(len(regions))
    for index, region in enumerate(regions):
        # Load the numpy arrays
        x_i = vertices[region, 0]
        y_i = vertices[region, 1]
        coordinates_vectors = np.zeros([len(x_i),2])
        coordinates_vectors[:,0] = x_i
        coordinates_vectors[:,1] = y_i
        # use np.roll to have a "+1" array of next indices that wraps at the end
        x_i_p1 = np.roll(x_i, -1)
        y_i_p1 = np.roll(y_i, -1)

        ### Compute the areas (zeroth moment tensor) ###
        areas[index] = np.sum((x_i + x_i_p1) * (y_i - y_i_p1), axis=0)

        ### Compute the perimeters ###
        coordinates_vectors[:,0] = coordinates_vectors[:,0] - x_i_p1
        coordinates_vectors[:,1] = coordinates_vectors[:,1] - y_i_p1
#        print(coordinates_vectors)
#        print(np.linalg.norm(coordinates_vectors, axis=1))
        perimeters[index] = np.sum(np.linalg.norm(coordinates_vectors, axis=1))

    # Do all the division in one go at the end
    areas = areas * -0.5

    # FIXME : compute absolute value to avoid having negative areas due to vertices not sorted counter-clockwise
#    areas = np.abs(areas)
    # FIXME : Instead put an assert verifying that every area is positive. This can be disabled when running the script.
    assert np.all(areas > 0), "Not every area is positive, this may be due to vertices not sorted vounter-clockwise"

    return None, perimeters, areas

def get_shared_vertices(regions, index_interest):
    """
    Determines which vertices are shared between regions[index_interest] and the
    other regions.

    Parameters
    -----
        regions : list of lists
            Indices of vertices in each revised Voronoi regions.
        index_interest: int
            Index of the region of interest

    Returns
    -----
        list_adjacent_regions : list of indices
            Returns a list with the absolute indices of a "region" in the regions list obtained from voronoi_finite_polygons_2d
        list_shared_vertices : list of lists
            Returns a list of list of format [(list of shared vertices] between region[index_interest] and region[index_neighbor])
    """
    # Create once a set of the vertices of the regions for efficient lookup and duplicate removal
    set_region_treated = set(regions[index_interest])
    list_adjacent_regions = []
    list_shared_vertices = []

    # Check every region except itself
    # Probably really inefficient, see Fukuda2004 for ways to formulate as a Linear Programming problem
    # Check vertices of region j
    for j, reg in enumerate(regions):
        if(j != index_interest):
            set_region_compared = set(regions[j])
            # Compute the intersection between the two sets
            shared_vertices = set_region_treated.intersection(set_region_compared)

            # If the set is not empty, append it,
            if(len(shared_vertices) != 0):
                list_shared_vertices.append(list(shared_vertices)) # Convert into a list for easy indexing after
                list_adjacent_regions.append(j)
            # OPTIMIZE : Can and Should be rewritten as a one line comprehension


    return list_adjacent_regions, list_shared_vertices


def find_vertex_index_in_region(regions, index_region ,vertex_value):
    """
    Locates the relative index of the vertex absolute index "vertex_value" within the region "regions[index_region]"
    Useful when knowing that an absolute vertex index value is present and when looking for the adjacent vertices

    Parameters
    -----
        regions : list of lists
            Indices of vertices in each revised Voronoi regions.
        index_region : int, index of the region to check
        vertex_value : value of the vertex one wants to look for

    Returns
    -----
        found : boolean, indicates if the vertex_value actually exists in the region
        index_vertex : int, relative index WITHIN the region

    Example :
    Region is [4, 7, 9, 8, 5] and we're looking for the vertex 9
    return index_vertex = 2
    """

    region = regions[index_region]
    found = False
    # Need a try/catch to avoid getting killed by a ValueError in case of vertex not found
    try:
        index_vertex = region.index(vertex_value)
    except ValueError:
        found = False # The vertex_value is not in the list of points of region
        index_vertex = -1 # Default value returned
    else:
        found = True

    return index_vertex, found



def get_nearest_neighbors(region, index, vertices):
    """
    Computes the Euclidean distance between the point vertices[region[index]]
    and the other vertices WITHIN a region.

    Parameters
    -----
        region : a single finite region from the Voronoi tesselation
        index : the index of the point of interest of which we are looking for the nearest neighbors
        vertices : a list of lists (vertices)

    Returns
    -----
        nearest_neighbors : numpy array with indices referring to the vertices list ordered from closest to furthest
    """


    curr_x = vertices[index][0]
    curr_y = vertices[index][1]

    # Compute the Euclidean distance between the point (curr_x, curr_y) and the other vertices of the region
    cost = np.fromiter( ( ((look_x - curr_x)**2 + (look_y - curr_y)**2)**0.5 for look_x, look_y in vertices[region] ), float)

    # Look for the indices that would sort the array from min to max
    nearest_neighbors = np.argsort(cost)

    # At this point the index is the index of the point of interest
    # The two closest points are on the index 1 and 2.



    return nearest_neighbors




def plot_contour(boxsize):
    """
    Plot the contour of the observation box for cell symmetry observation

    Parameters
    -----

    boxsize : float (or int)
        Size of the observation box.

    Returns
    -----
    Nothing.
    """
    # Plot the contour of the observation box
    # contour = [[-boxsize/2,-boxsize/2],[boxsize/2,-boxsize/2],[boxsize/2,boxsize/2],[-boxsize/2,boxsize/2]]
    plt.plot( [-boxsize/2.0,-boxsize/2.0],[boxsize/2.0,-boxsize/2.0],'-b')
    plt.plot([boxsize/2.0,-boxsize/2.0],[boxsize/2.0,boxsize/2.0] ,'-b')
    plt.plot([boxsize/2.0,boxsize/2.0],[-boxsize/2.0,boxsize/2.0],'-b')
    plt.plot([-boxsize/2.0,boxsize/2.0],[-boxsize/2.0,-boxsize/2.0] , '-b' )

def generate_periodic_voronoi_cells(points, boxsize):
    """
    Copy the initialized points around the observation box of size "boxsize"

    Parameters
    -----
    points : numpy array of arrays
        (x,y) coordinates of the Voronoi centers
    boxsize : float
        Size of a side of the observation box (assumed square-shaped)

    Returns
    -----
    extended_points : numpy array of arrays
        (x,y) coordinates of the Voronoi centers and their copy
    """
    # Empty point array to store the original points and three copies
    extended_points = np.zeros((len(points)*4,2))

    # Copy the original values
    for index, val in enumerate(points):
        extended_points[index][0] = points[index][0]
        extended_points[index][1] = points[index][1]

    # Translate along the y-axis
    for index, val in enumerate(points):
        extended_points[index+len(points)][0] = points[index][0]
        extended_points[index+len(points)][1] = points[index][1] + boxsize
        if(extended_points[index+len(points)][1] > boxsize):
            extended_points[index+len(points)][1] = extended_points[index+len(points)][1] - boxsize * 2.0

    # Translate along the x-axis
    for index, val in enumerate(points):
        extended_points[index+len(points)*2][0] = points[index][0] + boxsize
        extended_points[index+len(points)*2][1] = points[index][1]
        if(extended_points[index+len(points)*2][0] > boxsize):
            extended_points[index+len(points)*2][0] = extended_points[index+len(points)*2][0] - boxsize * 2.0

    # Translate along the diagonal
    for index, val in enumerate(points):
        extended_points[index+len(points)*3][0] = points[index][0] + boxsize
        extended_points[index+len(points)*3][1] = points[index][1] + boxsize
        if(extended_points[index+len(points)*3][0] > boxsize):
            extended_points[index+len(points)*3][0] = extended_points[index+len(points)*3][0] - boxsize * 2.0
        if(extended_points[index+len(points)*3][1] > boxsize):
            extended_points[index+len(points)*3][1] = extended_points[index+len(points)*3][1] - boxsize * 2.0

    return extended_points



# def generate_periodic_voronoi_cells_2(points, boxsize):
#     """
#     Copy the initialized points around the observation box of size "boxsize", over the 8 nearest regions

#     Parameters
#     points : numpy array of arrays
#         (x,y) coordinates of the Voronoi centers
#     boxsize : float
#         Size of a side of the observation box (assumed square-shaped)
#     -----
#     Returns
#     extended_points : numpy array of arrays
#         (x,y) coordinates of the Voronoi centers and their copy
#     """
#     # Empty point array to store the original points and three copies
#     extended_points = np.zeros((len(points)*9,2))

#     # Copy the original values
#     for index, val in enumerate(points):
#         extended_points[index][0] = points[index][0]
#         extended_points[index][1] = points[index][1]

#     # Translate along the y-axis
#     for index, val in enumerate(points):
#         extended_points[index+len(points)][0] = points[index][0]
#         extended_points[index+len(points)][1] = points[index][1] + boxsize
#         extended_points[index+len(points)*2][0] = points[index][0]
#         extended_points[index+len(points)*2][1] = points[index][1] - boxsize

#     # Translate along the x-axis
#     for index, val in enumerate(points):
#         extended_points[index+len(points)*3][0] = points[index][0] + boxsize
#         extended_points[index+len(points)*3][1] = points[index][1]
#         extended_points[index+len(points)*4][0] = points[index][0] - boxsize
#         extended_points[index+len(points)*4][1] = points[index][1]

#     # Translate along the diagonal
#     for index, val in enumerate(points):
#         extended_points[index+len(points)*5][0] = points[index][0] + boxsize
#         extended_points[index+len(points)*5][1] = points[index][1] + boxsize

#         extended_points[index+len(points)*6][0] = points[index][0] - boxsize
#         extended_points[index+len(points)*6][1] = points[index][1] - boxsize

#         extended_points[index+len(points)*7][0] = points[index][0] + boxsize
#         extended_points[index+len(points)*7][1] = points[index][1] - boxsize

#         extended_points[index+len(points)*8][0] = points[index][0] - boxsize
#         extended_points[index+len(points)*8][1] = points[index][1] + boxsize

#     return extended_points

def generate_periodic_voronoi_cells_2( double [:,:] points, double boxsize):
    """
    Copy the initialized points around the observation box of size "boxsize", over the 8 nearest regions
    C-like version (250+ times faster than the pure python)

    Parameters
    -----
    points : numpy array of arrays
        (x,y) coordinates of the Voronoi centers
    boxsize : float
        Size of a side of the observation box (assumed square-shaped)

    Returns
    -----
    extended_points : numpy array of arrays
        (x,y) coordinates of the Voronoi centers and their copy
    """

    #Define all the variables
    cdef Py_ssize_t index
    cdef Py_ssize_t len_points = len(points)
    DTYPE=np.float64
    # Empty point array to store the original points and three copies
    extended_points_np = np.zeros((len(points)*9,2), dtype=DTYPE)
    cdef double [:,:] extended_points = extended_points_np

    index = 0
    # Copy the original values
    for index in range(len_points):
        extended_points[index,0] = points[index,0]
        extended_points[index,1] = points[index,1]

        # Translate along the y-axis
    for index in range(len_points):
        extended_points[index+len_points,0] = points[index,0]
        extended_points[index+len_points,1] = points[index,1] + boxsize
        extended_points[index+len_points*2,0] = points[index,0]
        extended_points[index+len_points*2,1] = points[index,1] - boxsize

    # Translate along the x-axis
    for index in range(len_points):
        extended_points[index+len_points*3,0] = points[index,0] + boxsize
        extended_points[index+len_points*3,1] = points[index,1]
        extended_points[index+len_points*4,0] = points[index,0] - boxsize
        extended_points[index+len_points*4,1] = points[index,1]

    # Translate along the diagonal
    for index in range(len_points):
        extended_points[index+len_points*5,0] = points[index,0] + boxsize
        extended_points[index+len_points*5,1] = points[index,1] + boxsize

        extended_points[index+len_points*6,0] = points[index,0] - boxsize
        extended_points[index+len_points*6,1] = points[index,1] - boxsize

        extended_points[index+len_points*7,0] = points[index,0] + boxsize
        extended_points[index+len_points*7,1] = points[index,1] - boxsize

        extended_points[index+len_points*8,0] = points[index,0] - boxsize
        extended_points[index+len_points*8,1] = points[index,1] + boxsize

    return extended_points

def generate_hexagonal_seed(point_number, boxsize, lattice_noise=0):
    """
    Create seed points (Voronoi center) on a controlled or "noisy" (with a bit of noise) hexagonal lattice, and applies periodic boundary conditions.

    Parameters
    -----
    point_number : int
        Number of seed point to create
    boxsize : float
        Length of the axis for observation box (assumed square for now)
    lattice_noise : float
        Intensity of the noise applied to the lattice seed points

    Returns
    -----
    generated_points : ndarray of ndarray
        Array of seed points on a lattice (of coordinates x,y)
    generated_angles : ndarray
        Array of randomly drawn angles in radians
    generated_polarities : ndarray of double
        Array of polarity vectors (cos theta_i, sin theta_i), initialized with random initial values
    (in the future maybe): flag
    """
    # FIXME Cannot yet handle non integer steps
    cpt = 0
    generated_points = np.zeros([point_number,2])
    step = np.sqrt(boxsize**2/point_number)

    for line_number,y in enumerate(np.arange(-boxsize/2.0,boxsize/2.0,step)):
        if(line_number % 2 == 0):
            for x in np.arange(-boxsize/2.0,boxsize/2.0,step):
                generated_points[cpt][0] = x + np.random.random() * lattice_noise
                generated_points[cpt][1] = y + np.random.random() * lattice_noise
                cpt = cpt + 1
        else:
            for x in np.arange( (-boxsize/2.0)+step/2,boxsize/2.0,step):
                generated_points[cpt][0] = x + np.random.random() * lattice_noise
                generated_points[cpt][1] = y + np.random.random() * lattice_noise
                cpt = cpt + 1

    return generated_points


if __name__ == '__main__':

    # For debugging purposes, generate a known seed
    np.random.seed(42)

    boxsize = 10

    # # Deterministic case
    # test_points = np.array([[0, 0], [0, 1],[0.5,1],[1,0.5] ,[0, 2], [1, 0], [1, 1], [1, 2],[2, 0], [2, 1], [2, 2]])
    # vor_test = Voronoi(test_points)

    # regions, vertices = voronoi_finite_polygons_2d(vor_test,boxsize*2.0)

    # plot_polygons(vor_test, regions, vertices)


    # Random case
#    points_f = np.random.rand(10,2)*boxsize
    # Hexagonal case
    points_f = generate_hexagonal_seed(100, boxsize, lattice_noise=0)
    # Periodicity
    points_f = generate_periodic_voronoi_cells_2(points_f, boxsize)
    vor = Voronoi(points_f)
    regions, vertices = voronoi_finite_polygons_2d(vor,boxsize*2.0)
#     for index, element in enumerate(get_polygons_properties_2d(regions, vertices)):
#    print(element)
    plot_polygons(vor, regions, vertices)


    # Check if ridge_points and ridges_vertices are still usable

    # ridges_pts = vor.ridge_points
    # print("Ridges points")
    # plt.plot(ridges_pts[:,0], ridges_pts[:,1], "r*")


#    for elem in ridges_pts:
#        print(elem)
#        plt.plot(elem, "r-")

    # DEBUG : get the ridges_vertices of the Voronoi object
    # ridges_vertices = vor.ridge_vertices
    # for elem in ridges_vertices:
    #     plt.plot(elem[0], elem[1], "bo")
#    for ind, val in enumerate(ridges_vertices):
#        plt.plot(val[0],val[1],"r*")
#        plt.plot(val)



    print("-"*50)

    # DEBUG : Print the different regions
    # print("Regions : ")
    # for index,region in enumerate(regions):
    #     print("Index : {0}, Region : {1}".format(index,region))


    print("-"*50)

#    print("Vertices : ")
#    for index, vertex in enumerate(vertices):
#         print("Index : {0}, Vertices: {1}".format(index,vertex))

    shrd_1 = get_shared_vertices(regions, 3)


    #### Plotting test for Random case

    for index_pair in vor.ridge_vertices:
        #Critical for a nice graph
        if index_pair[0] >= 0 and index_pair[1] >= 0:
            v0 = vor.vertices[index_pair[0]]
            v1 = vor.vertices[index_pair[1]]
                # Draw a line from v0 to v1.
            plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)


#    get_cell_energy(4,7)

    # Search the nearest neighbors of a point and plot the two closest
    # Actually not needed since vertices are sorted counter-clockwise. !!!

    # DEBUG
#    NN =  get_nearest_neighbors(regions[4], 2, vertices)
#    reg = regions[4]
#    print("Reg : {0}".format(reg))
#    print("NN : {0}".format(NN))
#    print("vertices[reg] : {0}".format(vertices[reg]))
#    print("vertices[reg[NN[0]]] :  {0}".format(vertices[reg[NN[0]]]))
#
#    print("-"*50)
#    print("Point of interest" )
#    pt1_x, pt1_y = vertices[reg[NN[1]]]
#    pt2_x, pt2_y = vertices[reg[NN[2]]]
#    plt.plot(pt1_x,pt1_y, "r*", mew=10, ms=10)
#    plt.plot(pt2_x, pt2_y,"b+", mew=10, ms=10)

    # Figure parameters
    plt.xlim(-boxsize/2.0, boxsize/2.0)
    plt.ylim(-boxsize/2.0, boxsize/2.0)
    plt.title("Deterministic test model. {0} regions".format(len(regions)))
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig('Hexagonal_Tesselation_100_seed_boxsize_10_boxsize_only.png')
    plt.show()
