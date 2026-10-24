""" 
Test case : map colors for polygons
Build a vector of orientation angles.
Transcribe this value into 
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
import matplotlib.colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

# Modifiy locally the plot_polygons function if needed
def plot_polygons(vor, regions, vertices, angles):
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
    # Get a colormap
    cmap = plt.get_cmap('hsv')
    # Assign a color to the angles
    # colors = cmap(angles)

    # norm_angles = matplotlib.colors.Normalize(vmin=0,vmax=2.0*np.pi)
    patches = []

    for region, angle in zip(regions, angles):
        polygon = vertices[region]

        # plt.fill(*zip(*polygon), alpha=0.4) # Color the polygons
        plt.fill(*zip(*polygon), facecolor=cmap(angle)) # Color the polygons
        plt.plot(polygon[:,0], polygon[:,1], '--k')
        ## Uncomment the lines below for a nicer display in an area of interest
        plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1) # Forces display in a box
        plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1) # Forces display in a box

        # Cast polygon into a matplotlib Polygon object
        polygon = Polygon(polygon)
        patches.append(polygon)
    
    # collection = PatchCollection(patches)
    # ax = plt.gca()
    # matplotlib.colorbar.ColorbarBase(ax)
    # ax.add_collection(collection)
    # collection.set_color(colors)
    # ax.autoscale_view()
    # fig = plt.gcf()
    # plt.colorbar()
if __name__ == '__main__':

    # Seed the random generator for reproducibility
    np.random.seed(42)

    point_number = 100
    boxsize = 10

    # Initialization
    points, angles, polarities = initialization(point_number, boxsize)

    # Generate an arbitrary angle vector
    test_angles = np.linspace(np.pi/2, np.pi, len(angles))
    test_angles = np.ones(len(angles))*np.pi


    # Build the Voronoi tesselation
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    plot_polygons(vor, regions, vertices, test_angles)
    plt.show()