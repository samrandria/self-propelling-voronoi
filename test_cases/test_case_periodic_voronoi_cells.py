"""
Test case : periodic voronoi CELLS and not only centers
"""

import numpy as np
from numpy import pi, random
import matplotlib.pyplot as plt
from voronoi_tesselation import get_polygons_properties_2d, voronoi_finite_polygons_2d
from scipy.spatial import voronoi_plot_2d, Voronoi, ConvexHull
from scipy.integrate import odeint
import random
from voronoi_tesselation import get_shared_vertices
from voronoi_tesselation import find_vertex_index_in_region
from flocking_confluent_marchetti import initialization



def generate_periodic_voronoi_cells(points, boxsize):
    """
    Copy the initialized points around the observation box of size "boxsize" 

    Parameters:
    points : numpy array of arrays
        (x,y) coordinates of the Voronoi centers
    boxsize : float
        Size of a side of the observation box (assumed square-shaped)
    -----
    Returns:
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

def plot_polygons(vor, regions, vertices):
    """
    Graphical function to plot colored polygons within a box

    Note that while the Voronoi regions are finite, they may still go beyond the box 
    Parameters:
    vor : A Voronoi object
    regions : list of lists
        Indices of vertices in each revised Voronoi regions.
    vertices : list of lists
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end. 
    """
    polygons_hulls = []
    index = 0
    for region in regions:
        polygon = vertices[region]
        # plt.fill(*zip(*polygon), alpha=0.4) # Color the polygons
        polygons_hulls.append(ConvexHull(polygon)) # Add the area to a list 
        plt.plot(polygon[:,0], polygon[:,1], '--k')
        ## Uncomment the lines below for a nicer display in an area of interest
        # plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1) # Forces display in a box
        # plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1) # Forces display in a box
        # print('Polygon {0}, Perimeter {1:.2f}, Area {2:.2f} '.format(index, polygons_hulls[index].area, polygons_hulls[index].volume))
        index += 1



if __name__ == '__main__':
    np.random.seed(42)
    point_number = 40
    boxsize = 5

    # points, angles, polarities = initialization(point_number,boxsize)

    # extended_points = generate_periodic_voronoi_cells(points,boxsize)

    test_points = np.array([[-0.811301, -0.862376], [-0.991442, 1.18615], [-1.04563, 2.42307], [1.32604, -0.347973], [1.20227, 0.375367], [0.09987, -1.0113], [0.654955, -2.30193], [-2.0921, -1.55172], [-1.74523, 0.645301], [-1.78317, -0.338932], [-0.108708, 0.98573], [2.3802, -0.715592], [-0.254544, -2.17942], [1.05956, 1.42854], [1.58066, 2.25834], [2.43478, -2.41764], [-0.721674, -0.00432083], [0.233984, 0.0381089], [-1.81503, 1.76315], [0.107944, 1.84848], [2.15288, 0.479513], [-1.25771, -1.56388], [2.26115, 1.53255], [0.847283, -1.342924], [1.77745, -1.65821  ]])

    extended_points = generate_periodic_voronoi_cells(test_points,boxsize)


    vor = Voronoi(extended_points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # vor_ex = voronoi_finite_polygons_2d(vor)

    plot_polygons(vor,regions,vertices)
    plt.xlim(-2.0*boxsize,2.0*boxsize)
    plt.ylim(-2.0*boxsize,2.0*boxsize)
    plt.plot(test_points[:,0], test_points[:,1], 'r+')

    # Draw the box contour
    contour = [[-2.5,-2.5],[2.5,-2.5],[2.5,2.5],[-2.5,2.5]]
    plt.plot( [-2.5,-2.5],[2.5,-2.5],'-b')
    plt.plot([2.5,-2.5],[2.5,2.5] ,'-b')
    plt.plot([2.5,2.5],[-2.5,2.5],'-b')
    plt.plot([-2.5,2.5],[-2.5,-2.5] , '-b' )

    # plt.plot(points[:,0], points[:,1], 'r+')
    # plt.plot(extended_points[len(points):2*len(points)-1,0], extended_points[len(points):2*len(points)-1,1], 'bo')
    # plt.plot(extended_points[2*len(points):3*len(points)-1,0], extended_points[2*len(points):3*len(points)-1,1], 'go')
    # plt.plot(extended_points[3*len(points):4*len(points)-1,0], extended_points[3*len(points):4*len(points)-1,1], 'yo')
    plt.show()