"""
Test of quiver plot
"""


from flocking_confluent_marchetti import initialization
import numpy as np
from numpy import pi, random
import matplotlib.pyplot as plt
import matplotlib
from voronoi_tesselation import get_polygons_properties_2d, voronoi_finite_polygons_2d 
from scipy.spatial import voronoi_plot_2d, Voronoi, ConvexHull
from scipy.integrate import odeint
import random
from voronoi_tesselation import get_shared_vertices
from voronoi_tesselation import find_vertex_index_in_region

# Test
from matplotlib.collections import PatchCollection


def plot_polygons_v2(vor, regions, vertices):
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
        plt.fill(*zip(*polygon), alpha=0.4) # Color the polygons
        polygons_hulls.append(ConvexHull(polygon)) # Add the area to a list 
        plt.plot(polygon[:,0], polygon[:,1], '--')
        ## Uncomment the lines below for a nicer display in an area of interest
        plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1) # Forces display in a box
        plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1) # Forces display in a box
        # print('Polygon {0}, Perimeter {1:.2f}, Area {2:.2f} '.format(index, polygons_hulls[index].area, polygons_hulls[index].volume))
        index += 1


if __name__ == '__main__':

    point_number = 100
    boxsize = 10
    points, angles, polarities = initialization(point_number, boxsize)

    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # Random values for the velocities
    velocities = np.random.rand(100,2)* 5 - 2.5 

    plot_polygons_v2(vor,regions,vertices)
    # Plot the seeds
    plt.plot(points[:,0], points[:,1], '+k')
    (vor,regions,vertices)
    
    plt.quiver(points[:,0], points[:,1], velocities[:,0], velocities[:,1])
    plt.title("Quiver plot with random velocities")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()