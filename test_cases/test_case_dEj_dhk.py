"""
Test case for the function dEj_dhk
"""

import math
import numpy as np
from numpy import pi, random
import matplotlib.pyplot as plt
from voronoi_tesselation import get_polygons_properties_2d, voronoi_finite_polygons_2d, plot_polygons
from scipy.spatial import voronoi_plot_2d, Voronoi, ConvexHull
from scipy.integrate import odeint
import random
from voronoi_tesselation import get_shared_vertices
from voronoi_tesselation import find_vertex_index_in_region


# Simulation box parameters
N = 100
L = 10
steps = 10
time = np.linspace(0, 10, steps) # Creates an array of 'steps' evenly spaced points


# Target parameters
P_0 = 1.0 # TO CHANGE 
A_0 = 1.0 
p_0 = P_0 / (A_0)**0.5
A_bar = float(L)**2/float(N) 

# Cell parameters
K_A = 1.0 # TO CHANGE
K_P = 1.0 # TO CHANGE
v_0 = 1.0
mu = 1.0
f_0 = v_0/mu 

def get_force_on_cell(regions, index, vertices, list_perimeter, list_area):
    """
    Compute the total force on cell "index" using equations (A5) (A7) (A8) (A9) BI et al. 2016 

    Parameters :
        regions : list of lists
            Indices of vertices in each revised Voronoi regions.
        index : int
            Index of the i-th region 
        vertices : list of lists
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        list_perimeter : list of float
            Perimeters of every region in regions obtained with get_polygon_properties
        list_area : list of float
            Area of every region in regions obtained with get_polygon_properties

    ----------
    Returns:
        force_cell : numpy array float
            [x,y] components of the force exerted on the cell "index"
    """

    # The formula for component mu (mu being x or y) goes like this
    # Let nu be x or y 
    # dE_j/dr_imu = sum over all coordinates nu, one coordinate at a time of
    # (sum over the k-adjacent cells of (dE_j/d_hknu) * dhk/drimu)


    # First determine the k-cells that are adjacent to the i-th cell
    adjacent_regions, shared_vertices = get_shared_vertices(regions, index)

    # DEBUG
    # print("Adjacent regions (to region {0}) : {1}".format(index,adjacent_regions))
    # print("Shared vertices : {0}".format(shared_vertices))

    # Calculate the dEj/dhknu term

    # Need the index j of the cell being treated
    # The relative position of the vertex k being treated
    # k -> the k nearest/adjacent neighbors

    dEj_hk = [] # Empty list to store the pair of energy [dEj_hkx, dEj_hky]


    for counter, j in enumerate(adjacent_regions):
        # Fetch area and perimeter of the adjacent regions
        A_j = list_area[j]
        P_j = list_perimeter[j]

        # DEBUG
        # print("A_{0} : {1}".format(j, A_j))
        # print("P_{0} : {1}".format(j, P_j))
        
        # Determine the neighbors vertices h_k+1 (h_kp1) and h_k-1 (h_km1)
        # The neighbors of the shared vertex h_k (between cells i and j), belonging to cell j

        for k in shared_vertices[counter]:
            # Goes through the list of shared vertices
            relative_index, found = find_vertex_index_in_region(regions, j, k)
            # Get the (x,y) coordinates for the vertices h_k, h_km1, h_kp1

            # DEBUG
            # print("Relative Index of {0} in {1} : {2}".format(k,regions[j],relative_index))

            # For security, could run a test on found value, but normally it should be alright
            # Obviously, I will get proven wrong.

            # h_k
            vertex_index_h_k = regions[j][relative_index]
            h_k = vertices[vertex_index_h_k]
            h_kx = h_k[0] # x coordinate, for easier mnemonic
            h_ky = h_k[1] # y coordinate, for easier mnemonic

            # h_km1
            vertex_index_h_km1 = regions[j][relative_index-1] # If relative_index-1 == -1, python will loop to the end of the list
            h_km1 = vertices[vertex_index_h_km1]
            h_km1x = h_km1[0] # x coordinate, for easier mnemonic
            h_km1y = h_km1[1] # y coordinate, for easier mnemonic

            # h_kp1
            # Need a supplementary test to avoid going out of bounds for the array
            if(relative_index+1 < len(regions[j])):
                # Normal case
                vertex_index_h_kp1 = regions[j][relative_index+1]
                h_kp1 = vertices[vertex_index_h_kp1]
                h_kp1x = h_kp1[0] # x coordinate, for easier mnemonic
                h_kp1y = h_kp1[1] # y coordinate, for easier mnemonic
            else:
                # Out of bounds case, roll back to index 0
                vertex_index_h_kp1 = regions[j][0]
                h_kp1 = vertices[vertex_index_h_kp1]
                h_kp1x = h_kp1[0] # x coordinate, for easier mnemonic
                h_kp1y = h_kp1[1] # y coordinate, for easier mnemonic

            # DEBUG
            # print("h_k = {0}".format(h_k))
            # print("h_km1 = {0}".format(h_km1))
            # print("h_kp1 = {0}".format(h_kp1))

            # DEBUG
            # print("Norm of h_km1 - h_k = {0}".format(np.linalg.norm(h_km1 - h_k)))
            # print("Norm of h_kp1 - h_k = {0}".format(np.linalg.norm(h_kp1 - h_k)))

            # Disgustingly long
            dEj_hkx =( K_A * (A_j-A_0) * (h_kp1y - h_km1y)
                      + 2 * K_P * (P_j - P_0) * ( (h_kx - h_km1x)/(np.linalg.norm(h_km1 - h_k))
                                                + (h_kx - h_kp1x)/(np.linalg.norm(h_kp1 - h_k)) ) )

            dEj_hky =( K_A * (A_j-A_0) * (h_kp1x - h_km1x)
                      + 2 * K_P * (P_j - P_0) * ( (h_ky - h_km1y)/(np.linalg.norm(h_km1 - h_k))
                                                + (h_ky - h_kp1y)/(np.linalg.norm(h_kp1 - h_k)) ) )

            dEj_hk.append([dEj_hkx, dEj_hky])

            # DEBUG
            # print("dEj_hkx = {0}".format(dEj_hkx))
            # print("dEj_hky = {0}".format(dEj_hky))
            # print("-"*50)

    dEj_hk = np.array(dEj_hk)

    # DEBUG
    # print("The END")
    # print("*"*100)
    # print("dEj_hk = {0}".format(dEj_hk))

    ###################################
    ### End of the dEj_dhk part


def polygon_perimeter_test(region, vertices):
    """
    Brute function to compute the perimeter of a polygon. 
    """
    perimeter = 0
    for i in range(len(region)):
        idx = region[i]
        idxm1 = region[i-1]
        perimeter += np.linalg.norm(vertices[idx]-vertices[idxm1])
    
    return perimeter
    
if __name__ == '__main__':

    h1 = [3, 2.5]
    h2 = [3.5, 4]
    h3 = [3, 5.5]
    h4 = [5.5, 3.5]
    h5 = [4.5, 1]
    h6 = [6, 2]
    h7 = [1.5, 4.5]
    h8 = [5, 6]
    h9 = [6, 4.5]
    h10 = [7.5, 5]
    h11 = [8, 3]
    h12 = [8.5, 3.5]
    h13 = [1.5, 3]

    vertices = [h1, h2, h7, h12, h5, h10, h3, h11, h13, h6, h8, h4, h9]
    vertices = np.array(vertices)

    # regions -> [ [h2, h3, h7, h13, h1], [h2,h1,h5,h6,h4], [h4,h6,h11,h12,h10,h9], [h2,h4,h9,h8,h3] ]
    regions = [ [1,6,2,8,0], [1,0,4,9,11], [11,9,7,3,5,12], [1,11,12,10,6] ]
    regions = np.array(regions)

    indices, perimeters, areas = get_polygons_properties_2d(regions, vertices)

    # for perimeter in perimeters:
    #     print("Perimeter : {0}".format(perimeter))
    #     # Results should be approximately 8.046, 9.147, 9.026, 8.625
    
    # for area in areas:
    #     print("Area : {0}".format(area))
    #      # Results should be approximately 4.125, 5.625, 5.375, 4.875
    
    # # Manual computation of perimeter
    # perim0 = polygon_perimeter_test(regions[0],vertices)
    # perim1 = polygon_perimeter_test(regions[1],vertices)
    # perim2 = polygon_perimeter_test(regions[2],vertices)
    # perim3 = polygon_perimeter_test(regions[3],vertices)
    # print("perim2 = {0}".format(perim0))
    # print("perim2 = {0}".format(perim1))
    # print("perim2 = {0}".format(perim2))
    # print("perim2 = {0}".format(perim3))

    get_force_on_cell(regions, 0, vertices, perimeters, areas)