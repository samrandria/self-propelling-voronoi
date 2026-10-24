'''
Reproduce the model from Flocking Transition in Confluent Tissue
Giavazzi, Marchetti et al.
arXiv:1706.01113v1
Note : float in python are double in C.
'''

####### Variable meaning ######


"""
r_i     : position of the cell
E       : Effective tissue energy
A_i     : Cross sectional area of the ith cell
P_i     : Perimeter of the ith cell
K_A     : Area stiffness
K_P     : Perimeter stiffness
A_0     : Area target value
P_0     : Perimeter target value
N       : Number of cells
boxsize : Side size of the simulation box (so area is boxsize**2) (In paper this is L)
A_bar   : Average cell area = L**2/N (considering a confluent case)
p_0     : Target shape index = P_0/(A_0**0.5)
fi_s    : Self-propulsive force of the ith cell such that f_i_s = f_0 * n_i
f_0      : Magnitude of the propulsive force = v_0/mu
v_0      : Cell motility
mu      : Mobility
n_i     : Direction of self-propulsive force = (cos theta_i, sin theta_i)
F_i     : Force arising from tissue energy (-grad_r_i E_i)
eta_i   : White noise
D_r     : Rotational diffusion rate
J       : Alignment due to neighbours (s)
tau     : Alignment rate = J**-1
unit_l  : Unit length = A_0**0.5
unit_t  : Unit time = 1/(mu*K_A*A_0)
MSD     : Mean Square Displacement = (1/N) * sum_i( (rprime_i(t) - rprime_i(0))**2)
rprime  : = r_i - r_CM
rCM     : Reference frame of the center of mass = (1/N) * sum_i(r_i)
q       : Cellular shape index = (1/N) sum_i(Pi/(A_i**0.5)
phi     : Vicsek order parameter = (1/N) < abs( sum_i(v_i/norm(v_i)) )>


delta_t = Simulation step in the Euler method
steps = Number of steps for simulation


"""



import numpy as np
from numpy import pi, random
import matplotlib
# Uncomment when using the server
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from voronoi_tesselation import get_polygons_properties_2d, voronoi_finite_polygons_2d, plot_polygons
from scipy.spatial import voronoi_plot_2d, Voronoi, ConvexHull
from scipy.integrate import odeint
import random
from voronoi_tesselation import get_shared_vertices
from voronoi_tesselation import find_vertex_index_in_region
from voronoi_tesselation import plot_contour
from voronoi_tesselation import generate_periodic_voronoi_cells_2
from voronoi_tesselation import generate_hexagonal_seed
import fit_ellipses
# from voronoi_tesselation import generate_periodic_voronoi_cells, generate_periodic_voronoi_cells_2
# Library for gif making
import imageio

# Library for argument passing and shell command calling
import os
import sys
import subprocess
import errno
# Writing in the CSV format
import csv

# Timer
from datetime import datetime

# DEBUG force logger
force_logger = []

# Global path for logging all the necessary files
# The path created in the main after passing the parameters will prefix this path
# TODO : Need to displace it in the main probably ! Or change it for the mkdir !
GLOBAL_LOG_PATH = 'logdirectory/'


def create_csv_log_file(filename, parameter_to_log):
    """
    Create a new file with the csv format if it does not already exist. Append if it already exists.

    Parameters
    filename: String
        Name of the file to create/to append to.
    parameter_to_log: Whatever
        What shall be written in the file.
    -----
    Returns
    success : boolean
        True if the file has been properly opened/written and closed. False otherwise.
    """
    # path = './simulation_test_trash/'
    path = GLOBAL_LOG_PATH
    # Convert the parameter into string just in case
    file_log_name = str(filename)
    file_log_name = os.path.join(path, file_log_name)
    try:
        with open(file_log_name,'a',newline='') as csvfile:
            file_log_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # for row in file:
            file_log_writer.writerow(parameter_to_log)
    except IOError:
        print("Could not open file {}. Will return False".format(file_log_name))
        return False

    return True

cdef inline double compute_cell_energy(double perimeter, double area, double K_A, double K_P, double A_0, double P_0 ):
    """
    Compute the cell energy of the identifier-th cell following Giavazzi Eq(1)
    Cython typed and inlined.

    Might need to rewrite the tissue energy computation as a single comprehension for speeding up
    Parameters
    perimeter : float
        Perimeter of the Voronoi cell
    area : float
        Area of the Voronoi cell
    K_A : float
        Area "stiffness"
    K_P : float
        Perimeter "stiffness"
    A_0 : float
        Target value for area
    P_0 : float
        Target value for perimeter
    -----------

    Returns
    energy : float
        Energy of the cell.
    -----------
    """

    # energy = K_A*(area - A_0)**2 + K_P*(perimeter - P_0)**2

    return (K_A*(area - A_0)) * (K_A*(area - A_0)) + (K_P*(perimeter - P_0)) * (K_P*(perimeter - P_0))

def compute_every_cell_energy(double[:] perimeters, double[:] areas, double K_A, double K_P, double A_0, double P_0):
    """
    Compute the tissue energy arising from every cell using Giavazzi Eq(A3)
    Cython/C-like

    Parameters
    -----
    perimeters : array of double
            Contains the perimeters of every cell in the tissue
    areas : array of double
        Contains the areas of every cell in the tissue
    K_A : float
        Area "stiffness"
    K_P : float
        Perimeter "stiffness"
    A_0 : float
        Target value for area
    P_0 : float
        Target value for perimeter

    Returns
    -----
    tissue_energy : double
            Sum of every cell energy
    """
    cdef double tissue_energy = 0.0
    cdef Py_ssize_t index
    cdef int length = len(perimeters)

    for index in range(length):
        tissue_energy = tissue_energy + compute_cell_energy(perimeters[index], areas[index], K_A, K_P, A_0, P_0 )

    return tissue_energy

# def compute_force_on_cell(regions, index, vertices, list_perimeter, list_area):
#     """
#     Compute the total force on cell "index" using equations (A5) (A7) (A8) (A9) BI et al. 2016

#     Parameters :
#     regions : list of lists
#         Indices of vertices in each revised Voronoi regions.
#     vertices : list of lists
#         Coordinates for revised Voronoi vertices. Same as coordinates
#         of input vertices, with 'points at infinity' appended to the
#         end.
#     list_perimeter : list of float
#         Perimeters of every region in regions obtained with get_polygon_properties
#     list_area : list of float
#         Area of every region in regions obtained with get_polygon_properties

#     ----------
#     Returns
#         force_cell : numpy array float
#             [x,y] components of the force exerted on the cell "index"
#     """

#     # The formula for component mu (mu being x or y) goes like this
#     # Let nu be x or y
#     # dE_j/dr_imu = sum over all coordinates nu, one coordinate at a time of
#     # (sum over the k-adjacent cells of (dE_j/d_hknu) * dhk/drimu)


#     # First determine the k-cells that are adjacent to the i-th cell
#     adjacent_regions, shared_vertices = get_shared_vertices(regions, index)
#     # Calculate the dEj/dhknu term

#     # Need the index j of the cell being treated
#     # The relative position of the vertex k being tr
#     # k -> the k nearest/adjacent neighbors

#     dEj_hk = [] # Empty list to store the pair of energy [dEj_hkx, dEj_hky]


#     for counter, j in adjacent_regions:
#         # Fetch area and perimeter of the adjacent regions
#         A_j = list_area[j]
#         P_j = list_perimeter[j]

#         # Determine the neighbors vertices h_k+1 (h_kp1) and h_k-1 (h_km1)
#         # The neighbors of the shared vertex h_k (between cells i and j), belonging to cell j

#         for k in shared_vertices[counter]:
#             # Goes through the list of shared vertices
#             relative_index, found =  find_vertex_index_in_region(regions, j, k)
#             # Get the (x,y) coordinates for the vertices h_k, h_km1, h_kp1

#             # h_k
#             vertex_index_h_k = regions[j][relative_index]
#             h_k = vertices[vertex_index_h_k]
#             h_kx = h_k[0] # x coordinate, for easier mnemonic
#             h_ky = h_k[1] # y coordinate, for easier mnemonic

#             # h_km1
#             vertex_index_h_km1 = regions[j][relative_index-1] # If relative_index-1 == -1, python will loop to the end of the list
#             h_km1 = vertices[vertex_index_h_km1]
#             h_km1x = h_km1[0] # x coordinate, for easier mnemonic
#             h_km1y = h_km1[1] # y coordinate, for easier mnemonic

#             # h_kp1
#             # Need a supplementary test to avoid going out of bounds for the array
#             if(relative_index+1 < len(regions[j])):
#                 # Normal case
#                 vertex_index_h_kp1 = regions[j][relative_index+1]
#                 h_kp1 = vertices[vertex_index_h_kp1]
#                 h_kp1x = h_kp1[0] # x coordinate, for easier mnemonic
#                 h_kp1y = h_kp1[1] # y coordinate, for easier mnemonic
#             else:
#                 # Out of bounds case, roll back to index 0
#                 vertex_index_h_kp1 = regions[j][0]
#                 h_kp1 = vertices[vertex_index_h_kp1]
#                 h_kp1x = h_kp1[0] # x coordinate, for easier mnemonic
#                 h_kp1y = h_kp1[1] # y coordinate, for easier mnemonic

#             # DEBUG
#             print("h_k = {0}".format(h_k))
#             print("h_km1 = {0}".format(h_km1))
#             print("h_kp1 = {0}".format(h_kp1))

#             # Disgustingly long
#             dEj_hkx =( K_A * (A_j-A_0) * (h_kp1y - h_km1y)
#                       + 2 * K_P * (P_j - P_0) * ( (h_kx - h_km1x)/(np.linalg.norm(h_km1 - h_k))
#                                                 + (h_kx - h_kp1x)/(np.linalg.norm(h_kp1 - h_k)) ) )

#             dEj_hky =( K_A * (A_j-A_0) * (h_kp1x - h_km1x)
#                       + 2 * K_P * (P_j - P_0) * ( (h_ky - h_km1y)/(np.linalg.norm(h_km1 - h_k))
#                                                 + (h_ky - h_kp1y)/(np.linalg.norm(h_kp1 - h_k)) ) )

#             #
#             dEj_hk.append([dEj_hkx, dEj_hky])


def compute_every_force_on_cell_euler(points,regions,vertices, delta_r, K_A, K_P, A_0, P_0):
    """
    Compute the total force exerted on a cell F = -dE/dr using "Euler's method"
    Seed center points are successively moved in the forward and backward x and y directions by a quantity
    delta_r, and the modified tissue energy is computed for each of these movements
    dE/dr -> (E(x+delta,y)-E(x,y))/delta + (E(x,y+delta)-E(x,y))/delta
    WARNING: Putting a delta too high may raise a KeyError in conjunction with the voronoi_finite_polygons_2d method

    Parameters
    ------
    points : 2D array of (double)
        The seed points (centers) of Voronoi tesselation.
    regions : list of lists
        Indices of vertices in each revised Voronoi regions.
    vertices : list of lists
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    delta_r : float
        Step of integration taken equal for delta_x and delta_y
    K_A : float
        Area "stiffness"
    K_P : float
        Perimeter "stiffness"
    A_0 : float
        Target value for area
    P_0 : float
        Target value for perimeter


    Returns
    ------
    force : 2D ndarray
        The i-th row contains the x and y component for the force applied to the i-th cell with the i-th center (seed)
    """




    # Compute the tissue energy prior any displacement :
    indx, perimeters, areas = get_polygons_properties_2d(regions,vertices)
    energy_old = compute_every_cell_energy(perimeters, areas, K_A, K_P, A_0, P_0)

    # diff_energy_x = E(x+delta_x,y) - E(x,y)
    # diff_energy_y = E(x,y+delta_y) - E(x,y)
    # The arrays indices correspond to the POINTS' indices
    array_diff_energy_x = np.zeros(len(points))
    array_diff_energy_y = np.zeros(len(points))

    pnt_number = len(points)

    for index, point in enumerate(points):

        ## X DISPLACEMENT FORWARD
        # Save the value of the point about to be displaced
        point_save_x = point[0]
        # Displace the x coordinate of a point forward by delta_r_x (delta_r)
        points[index][0] = point[0] + delta_r

        # DEBUG : Add periodic boundary generation HERE
        extended_points = generate_periodic_voronoi_cells_2(points,boxsize)

        # Recompute the tesselation
        vor = Voronoi(extended_points)
        # Reconstruct finite regions
        regions_modified_x, vertices_modified_x = voronoi_finite_polygons_2d(vor, point_number=pnt_number)
        # regions_modified_x, vertices_modified_x = voronoi_finite_polygons_2d(vor, boxsize*2.0)
        # indices_modified_x, perimeters_modified_x, areas_modified_x = get_polygons_properties_2d(regions_modified_x, vertices_modified_x)
        indices_modified_x, perimeters_modified_x, areas_modified_x = get_polygons_properties_2d(regions_modified_x[0:pnt_number], vertices_modified_x)

        # Compute the new tissue energy (after a delta_r_x displacement of point[index]) and save it
        energy_modified_x_forward = compute_every_cell_energy(perimeters_modified_x, areas_modified_x, K_A, K_P, A_0, P_0)

        # Replace the displaced point at its original position
        # FIXME : Check if the point_save is distinct from points[index][0]
        points[index][0] = point_save_x

        ## X DISPLACEMENT BACKWARD
        # Save the value of the point about to be displaced
        point_save_x = point[0]
        # Displace the x coordinate of a point backward by delta_r_x (delta_r)
        points[index][0] = point[0] - delta_r

        # DEBUG : Add periodic boundary generation HERE
        extended_points = generate_periodic_voronoi_cells_2(points,boxsize)

        # Recompute the tesselation
        vor = Voronoi(extended_points)
        # Reconstruct finite regions
        regions_modified_x, vertices_modified_x = voronoi_finite_polygons_2d(vor, point_number=pnt_number)
        # regions_modified_x, vertices_modified_x = voronoi_finite_polygons_2d(vor, boxsize*2.0)
        # indices_modified_x, perimeters_modified_x, areas_modified_x = get_polygons_properties_2d(regions_modified_x, vertices_modified_x)
        indices_modified_x, perimeters_modified_x, areas_modified_x = get_polygons_properties_2d(regions_modified_x[0:pnt_number], vertices_modified_x)

        # Compute the new tissue energy (after a delta_r_x displacement of point[index]) and save it
        energy_modified_x_backward = compute_every_cell_energy(perimeters_modified_x, areas_modified_x, K_A, K_P, A_0, P_0)

        # Replace the displaced point at its original position
        points[index][0] = point_save_x


        ## X DERIVATIVE COMPUTATION
        array_diff_energy_x[index] = energy_modified_x_forward - energy_modified_x_backward
        # array_diff_energy_x[index] = energy_modified_x_forward - energy_old



        #############################

        ## Y DISPLACEMENT FORWARD
        # Do the same thing for the y coordinate

        # Save the value of the point about to be displaced
        point_save_y = point[1]
        # Displace the y coordinate of a point forward by delta_r_y (delta_r)
        points[index][1] = point[1] + delta_r

        # DEBUG : Add periodic boundary generation HERE
        extended_points = generate_periodic_voronoi_cells_2(points,boxsize)

        # Recompute the tesselation
        vor = Voronoi(extended_points)
        # Reconstruct finite regions
        regions_modified_y, vertices_modified_y = voronoi_finite_polygons_2d(vor, point_number=pnt_number)
        # regions_modified_y, vertices_modified_y = voronoi_finite_polygons_2d(vor, boxsize*2.0)
        indices_modified_y, perimeters_modified_y, areas_modified_y = get_polygons_properties_2d(regions_modified_y[0:pnt_number], vertices_modified_y)

        # Compute the new tissue energy (after a delta_r_y displacement of point[index]) and save it
        energy_modified_y_forward = compute_every_cell_energy(perimeters_modified_y, areas_modified_y, K_A, K_P, A_0, P_0)
        # Replace the displaced point at its original position
        points[index][1] = point_save_y

        ## Y DISPLACEMENT BACKWARD
        # Do the same thing for the y coordinate

        # Save the value of the point about to be displaced
        point_save_y = point[1]
        # Displace the y coordinate of a point backward by delta_r_y (delta_r)
        points[index][1] = point[1] - delta_r

        # DEBUG : Add periodic boundary generation HERE
        extended_points = generate_periodic_voronoi_cells_2(points,boxsize)

        # Recompute the tesselation
        vor = Voronoi(extended_points)
        # Reconstruct finite regions
        regions_modified_y, vertices_modified_y = voronoi_finite_polygons_2d(vor, point_number=pnt_number)
        # regions_modified_y, vertices_modified_y = voronoi_finite_polygons_2d(vor, boxsize*2.0)
        indices_modified_y, perimeters_modified_y, areas_modified_y = get_polygons_properties_2d(regions_modified_y[0:pnt_number], vertices_modified_y)

        # Compute the new tissue energy (after a delta_r_y displacement of point[index]) and save it
        energy_modified_y_backward = compute_every_cell_energy(perimeters_modified_y, areas_modified_y, K_A, K_P, A_0, P_0)
        # Replace the displaced point at its original position
        points[index][1] = point_save_y

        ## Y DERIVATIVE COMPUTATION
        array_diff_energy_y[index] = energy_modified_y_forward - energy_modified_y_backward


        ### Debug: uncomment if need to register the modifications to the areas and perimeter
        ### during the spatial derivative part.
        # Log the following parameters in the appropriate files
        # One written row : $(point_numbers) areas/perimeters after the displacement of one seedpoint by delta_r_r
        # create_csv_log_file("areas_log_modified_x.csv", areas_modified_x)
        # create_csv_log_file("areas_log_modified_y.csv", areas_modified_y )
        # create_csv_log_file("perimeters_log_modified_x.csv", perimeters_modified_x)
        # create_csv_log_file("perimeters_log_modified_y.csv", perimeters_modified_y)



    # ON CHERCHE LES ENERGIES DERIVEES PAR RAPPORT AU CENTRE R_I
    dE_drix = array_diff_energy_x/(2*delta_r)
    dE_driy = array_diff_energy_y/(2*delta_r)
    # dE_drix = array_diff_energy_x/delta_r
    # dE_driy = array_diff_energy_y/delta_r

    # Write into a log file
    create_dE_drixy_csv_log_file(dE_drix,dE_driy)

    # 2D array to store the x and y components
    # force = np.array([-dE_drix, -dE_driy])
    ## Instead let us store like [ [f_0x, f_0y], [f_1x,f_1y]... [f_nx,f_ny]]

    forces = np.zeros([len(points),2])
    forces[:,0] = dE_drix
    forces[:,1] = dE_driy

    # DEBUG
    forces_norm = np.linalg.norm(forces, axis=1)
    # Append into the force logger
    force_logger.append(forces_norm)

    return forces


def create_dE_drixy_csv_log_file(dE_drix,dE_driy):
    """
    Write the tissue energies after each "spatial derivation" i.e. displacement of a seed point by delta_x then by delta_y (both equal to delta_r)

    Parameters
    ------
    dE_drix: numpy array
        Difference between the tissue energy after a delta displacement along the x-axis and before the displacement, divided by delta
    dE_driy: numpy array
        Difference between the tissue energy after a delta displacement along the y-axis and before the displacement, divided by delta

    Returns
    ------
    Nothing
    """
    # path = './simulation_test_trash/'
    path = GLOBAL_LOG_PATH

    energies_log_name = "dE_drixy_log.csv"
    energies_log_name = os.path.join(path, energies_log_name)
    with open(energies_log_name,'a',newline='') as csvfile:
        energies_log_name= csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in zip(dE_drix,dE_driy):
            energies_log_name.writerow(row)





def generate_periodic_boundaries(val, box):
    """
    Generates periodic boundaries by translating values to fit inside the observation box.

    Inspired by Hiraiwa Tetsuya. Modified because of performance issues.
    Parameters
        val: double, coordinate
        box: size of one side of the observation box
    -----
    Returns
        translated_value: double, coordinate after translation

    Note: in CCM.C (program by T.H), this function is called at initialization and [after Runge-Kutta but before polarity time evolution]
    """

    ## Just like in a videogame, when you cross a side of the screen, you appear back on the other side

    # while(val > box/2.0):
    #     # val = val - box
    #     # The above option is correct but time consuming
    #     val = val % box
    # while( val <= -box/2.0):
    #     # val = val + box
    #     # The above option is correct but time consuming
    #     val = val % box

    while(val < -box/2.0 or val > box/2.0):
        if(val > box/2.0 and val < box):
            val = (box % val) * (-1.0)
        elif(val < -box/2.0 and val > -box):
            val = (-box % val) * (-1.0)
        elif(val >= box):
            val = val % box
        elif(val <= box):
            val = val % -box
    return val


# def generate_periodic_boundaries_copy_seed(points, boxsize)

def initialization(point_number, boxsize):
    """
    Create an array of random seed points (Voronoi center), and applies periodic boundary conditions.
    TO DO: initialize polarity, angles, momentum, a flag indicating at which stage we are currently in.

    Parameters
    -----
    point_number : int
        Number of seed point to create
    boxsize : float
        Length of the axis for observation box (assumed square for now)

    Returns
    -----
    generated_points : ndarray of ndarray
        Array of random seed points (of coordinates x,y)
    generated_angles : ndarray
        Array of randomly drawn angles in radians
    generated_polarities : ndarray of double
        Array of polarity vectors (cos theta_i, sin theta_i), initialized with random initial values
    (in the future maybe): flag
    """
    # XXX : DEBUG
    generated_points = np.random.rand(point_number, 2) * boxsize - (boxsize * 0.5 * 0.5)
    # XXX : With a Poisson distribution
    # generated_points = np.random.poisson(lam=1.0,size=(point_number, 2)) * boxsize - (boxsize * 0.5 * 0.5)
    # generated_points = (2* np.random.rand(point_number, 2) - 1) * boxsize * 0.5
    # DEBUG
    # generated_points = generated_points * 10
    # print(generated_points)
    # print('-'*10)

    # Extremely important : places everyone in the observation box
    # TODO : Maybe find a better name for the function
    for coord in generated_points:
        coord[0] = generate_periodic_boundaries(coord[0], boxsize) # x coordinate
        coord[1] = generate_periodic_boundaries(coord[1], boxsize) # y coordinate
        # print(coord[0], coord[1])



    # ADD ANGLE INITIALISATION, POLARITY, NOISE, ETC.
    # Angle initialisation for every seed point (initialisation is done in radians)
    generated_angles = np.random.random(len(generated_points))* 2.0 * np.pi
    # Generate the associated polarity vectors
    generated_polarities = generate_every_polarity_vector(generated_angles)

    return generated_points, generated_angles, generated_polarities

def lattice_initialization(point_number, boxsize, lattice_noise=0):
    """
    Create seed points (Voronoi center) on a controlled or "noisy" (with a bit of noise) lattice, and applies periodic boundary conditions.
    TO DO: initialize polarity, angles, momentum, a flag indicating at which stage we are currently in.

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
    # FIXME Step should be computed and not hard coded
    generated_points = [[x + np.random.random() * lattice_noise, y + np.random.random() * lattice_noise] for x in range(int(-boxsize/2), int(boxsize/2),1) for y in range(int(-boxsize/2), int(boxsize/2),1)]


    # Extremely important : places everyone in the observation box
    # TODO : Maybe find a better name for the function
    for coord in generated_points:
        coord[0] = generate_periodic_boundaries(coord[0], boxsize) # x coordinate
        coord[1] = generate_periodic_boundaries(coord[1], boxsize) # y coordinate
        # print(coord[0], coord[1])


    generated_points = np.array(generated_points)
    # ADD ANGLE INITIALISATION, POLARITY, NOISE, ETC.
    # Angle initialisation for every seed point (initialisation is done in radians)
    generated_angles = np.random.random(len(generated_points))* 2.0 * np.pi
    # Generate the associated polarity vectors
    generated_polarities = generate_every_polarity_vector(generated_angles)

    return generated_points, generated_angles, generated_polarities


def hexagonal_lattice_initialization(point_number, boxsize, lattice_noise=0):
    """
    Create seed points (Voronoi center) on an triangular pattern (to give hexagonal cells) and applies periodic boundary conditions.
    TO DO: initialize polarity, angles, momentum, a flag indicating at which stage we are currently in.

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
    # FIXME Step should be computed and not hard coded
    generated_points = generate_hexagonal_seed(point_number, boxsize, lattice_noise=0)



    # Extremely important : places everyone in the observation box
    # TODO : Maybe find a better name for the function
    for coord in generated_points:
        coord[0] = generate_periodic_boundaries(coord[0], boxsize) # x coordinate
        coord[1] = generate_periodic_boundaries(coord[1], boxsize) # y coordinate
        # print(coord[0], coord[1])


    # ADD ANGLE INITIALISATION, POLARITY, NOISE, ETC.
    # Angle initialisation for every seed point (initialisation is done in radians)
    generated_angles = np.random.random(len(generated_points))* 2.0 * np.pi
    # Generate the associated polarity vectors
    generated_polarities = generate_every_polarity_vector(generated_angles)

    return generated_points, generated_angles, generated_polarities


def compute_cell_motion(seed_point, force, polarity, v_0, mu, delta_t, observation_boxsize):
    """
    Computes the equation of motion (Giavazzi (2)) using Euler's method for an individual cell

    Note : Handles periodic boundaries
    Parameters
    seed_point : numpy array
        Center of a Voronoi region of (coordinates (x,y))
    force : numpy array
        Force exerted on the region of center seed_point. (x,y) components
    polarity : numpy array
        Polarity vector for the region of center seed_point with. (x,y) components
    v_0 : double
        Speed of the cell (assumed constant)
    mu : double
        Mobility of the cell (assumed constant)
    delta_t : double
        Time integration step
    observation_boxsize : double
        Size of the observation box for boundary conditions
    -----
    Returns
    seed_point_next : numpy array
        (x,y) coordinates of the seed point
    velocity_vector : numpy array
        (x,y) components of the velocity vector associated to a moving seed point
    """

    # # Compute the next x coordinate
    # seed_point_next_x = delta_t * (mu * force[0] + v_0 * polarity[0]) + seed_point[0]
    # # Periodic boundary in x !
    # seed_point_next_x = generate_periodic_boundaries(seed_point_next_x, observation_boxsize)

    # # Compute the next y coordinate
    # seed_point_next_y = delta_t * (mu * force[1] + v_0 * polarity[1]) + seed_point[1]
    # # Periodic boundary in y !
    # seed_point_next_y = generate_periodic_boundaries(seed_point_next_y, observation_boxsize)

    # # Regroup the x and y coordinate in a numpy array
    # seed_point_next = np.array([seed_point_next_x, seed_point_next_y])

    # # Compute a velocity vector like (point_next - point_before)/delta_t
    # # Nicely done with numpy arrays
    # # DANGER : add an epsilon to avoid dividing by 0 when delta_t is zero during tests
    # epsilon = 0.001
    # velocity_vector = (seed_point_next - seed_point)/(delta_t + epsilon)


    ##### Compute first the velocity vector and then the periodic boundary
    # Compute the next x coordinate
    seed_point_next_x = delta_t * (mu * force[0] + v_0 * polarity[0]) + seed_point[0]

    # Compute the next y coordinate
    seed_point_next_y = delta_t * (mu * force[1] + v_0 * polarity[1]) + seed_point[1]

    # Regroup the x and y coordinate in a numpy array for velocity computation
    seed_point_next = np.array([seed_point_next_x, seed_point_next_y])

    # Compute a velocity vector like (point_next - point_before)/delta_t
    # Nicely done with numpy arrays
    velocity_vector = (seed_point_next - seed_point)/(delta_t)


    # Periodic boundary in x !
    seed_point_next_x = generate_periodic_boundaries(seed_point_next_x, observation_boxsize)
    # Periodic boundary in y !
    seed_point_next_y = generate_periodic_boundaries(seed_point_next_y, observation_boxsize)

    # Re-Regroup the x and y coordinate in a numpy array
    seed_point_next = np.array([seed_point_next_x, seed_point_next_y])


    return seed_point_next, velocity_vector


def compute_every_cell_motion(seed_points, forces, polarities, v_0, mu, delta_t, observation_boxsize):
    """
    Compute the next position for every seed points using Giavazzi (2)
    Generalizes compute_cell_motion() to every seed point.

    Parameters
    seed_point : numpy array of arrays
        Array of centers of Voronoi regions of (coordinates (x,y))
    force : numpy array of arrays
        Array of forces exerted on the region of center seed_point. (x,y) components
    polarity : numpy array of arrays
        Array of polarity vectors for the region of center seed_point. (x,y) components
    v_0 : double
        Speed of the cell (assumed constant)
    mu : double
        Mobility of the cell (assumed constant)
    delta_t : double
        Time integration step
    observation_boxsize : double
        Size of the observation box for boundary conditions
    -----
    Returns
    seed_points_next : numpy array of arrays
        Array of (x,y) coordinates of the seed points
    velocities_vector : numpy array
        Array of (x,y) components of the velocity vectors associated to moving seed points
    """

    seed_points_next = []
    velocities_vector = []

    # DEBUG
    if (len(seed_points) == len(forces) and len(forces)== len(polarities)):
        # print("TTUTUTUTUTUTUTURUU")
        pass
    else:
        print("ARGH")
        print(len(seed_points))
        print(len(forces))
        print(len(polarities))
    # TODO : Improve this part with numpy arrays please.
    for seed_point, force, polarity in zip(seed_points,forces,polarities):
        seed_point_next, velocity_vector = compute_cell_motion(seed_point, force, polarity, v_0, mu, delta_t, observation_boxsize)
        seed_points_next.append(seed_point_next)
        velocities_vector.append(velocity_vector)

    # Convert into a numpy array
    seed_points_next = np.array(seed_points_next)
    velocities_vector = np.array(velocities_vector)

    # Log the x and y components of the velocity vector
    create_velocity_csv_log_file(velocities_vector)


    return seed_points_next, velocities_vector


def create_velocity_csv_log_file(velocities_vector):
    """
    Log the x and y components of the velocity vector

    Parameters
    -----
    velocities_vector : numpy array
        Array of x and y coordinates of the velocity vector

    Returns
    ------
    Nothing
    """
    # Create a specific function to avoid having multiple path
    # path = "./simulation_test_trash/"
    path = GLOBAL_LOG_PATH
    velocities_vector_name = "velocities_log.csv"
    velocities_vector_name = os.path.join(path, velocities_vector_name)
    with open(velocities_vector_name,'a',newline='') as csvfile:
        velocities_vector_name= csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in velocities_vector:
            velocities_vector_name.writerow(row)




def gaussian_white_noise_process(mean, std, sample_number):
    """
    Draws random samples from a gaussian distribution

    Parameters
    mean : double
        Mean of the Gaussian Distribution
    std : double
        Standard deviation of the white noise process
    sample_number : int
        Number of samples to generate
    -----
    Returns
    white_noise_samples : numpy array
        Array of random white noise samples (double)
    """

    white_noise_samples = np.random.normal(mean, std, sample_number)
    return white_noise_samples


# def compute_random_rotational_diffusion_euler(angle, random_process, delta_t):
#     """
#     Compute the next value of an angle for the polarity vector using Giavazzi (3) and Euler's method

#     Parameters
#     angle : double
#         Theta value in radian for the polarity vector.
#     random_process : double
#         Random value of a white noise process
#     delta_t : double
#         Time integration step
#     -----
#     Returns
#     angle_next : numpy array
#         Theta value in radian for the next timestep
#     """

#     angle_next = delta_t * random_process + angle

#     # Need to map over [0;2pi[
#     # while(angle_next >= 2*np.pi):
#     #     angle_next = angle_next - 2*np.pi
#     # while(angle_next < 0):
#     #     angle_next = angle_next + 2*np.pi


#     # Modulo 2 pi to normalize
#     angle_next = angle_next % (2*np.pi)





#     return angle_next



# def compute_every_random_rotational_diffusion_euler(angles, angular_noise, delta_t):
#     """
#     Compute the values at next timestep for every angle assuming random rotational diffusion (Giavazzi (3))
#     and using Euler's method

#     Parameters
#     angles: numpy array
#         Theta values in radian for the polarity vector
#     angular_noise : double
#         D_r term, determines the memory of stochastic noise in the system
#     delta_t : double
#         Time integration step
#     -----
#     Returns
#     angles_next : numpy array
#         Theta values in radian for the next timestep
#     """

#     # Generate an array of samples from a white noise process of zero mean and var = 2*D_r
#     random_process_samples = gaussian_white_noise_process(0, angular_noise*2, len(angles))

#     angles_next = []

#     for angle, random_sample in zip(angles, random_process_samples):
#         angle_next = compute_random_rotational_diffusion_euler(angle, random_sample, delta_t)
#         # Store the newly computed angle in a list
#         angles_next.append(angle_next)

#     # Convert to numpy array
#     angles_next = np.array(angles_next)
#     return angles_next

def compute_every_random_rotational_diffusion_euler(angles, angular_noise,delta_t):
    """
    Compute the values at next timestep for every angle assuming random rotational diffusion (Giavazzi (3))
    and using Euler's method

    Parameters
    angles: numpy array
        Theta values in radian for the polarity vector
    angular_noise : double
        D_r term, determines the memory of stochastic noise in the system
    delta_t : double
        Time integration step
    -----
    Returns
    angles_next : numpy array
        Theta values in radian for the next timestep
    """

    random_process_samples = gaussian_white_noise_process(0, angular_noise*2, len(angles))
    angles_next = delta_t * random_process_samples + angles

    # Modulo 2 pi to normalize
    angles_next = np.mod(angles_next,2*np.pi)

    return angles_next


def compute_angle_alignment_long_axis(polarity_angles, ellipses_major_axis_angles, delta_t, angular_noise, interaction_strength=1):
    """
    Compute the values at next timestep for every angle assuming random rotational diffusion (Giavazzi (3))
    and using Euler's method

    Parameters
    -----
    polarity_angles: numpy array
        Theta values in radians for the polarity vector
    ellipses_major_axis_angles : numpy array
        Array of angle values between the x-axis and the fitted ellipses major axes
    interaction_strength : double
        Prefactor describing the tendency for the polarity angle and the major axis angle to align
    angular_noise : double
        D_r term, determines the memory of stochastic noise in the system
    delta_t : double
        Time integration step

    Returns
    -----
    angles_next : numpy array
        Theta values in radian for the next timestep
    """
    # Draw random values for the noise
    random_process_samples = gaussian_white_noise_process(0, angular_noise*2, len(polarity_angles))
    # XXX : uncomment if you would like Bias = angular noise
#    random_process_samples = angular_noise
    angles_next = delta_t * (interaction_strength * np.sin(polarity_angles - ellipses_major_axis_angles) + random_process_samples) + polarity_angles

    # Normalize the angles values in case of future utilisation by a non numpy function
    angles_next = np.mod(angles_next,2*np.pi)

    return angles_next


def generate_every_polarity_vector(angles):
    """
    Generate the polarity vectors [cos(angle),sin(angle)] components for every point

    Parameter:
    angles : numpy array
        Polarity angles in radian
    -----
    Returns
    polarities_vector : numpy array of numpy array
        Array of [cos(angle), sin(angle)] components for every point
    """

    polarities_vector = np.zeros([len(angles),2])
    polarities_x = np.cos(angles)
    polarities_y = np.sin(angles)

    polarities_vector[:,0] = polarities_x[:]
    polarities_vector[:,1] = polarities_y[:]
    return polarities_vector


def compute_next_system_step(points, regions, vertices, angles, polarities, boxsize, angular_noise, delta_t, delta_r, K_A, K_P, A_0, P_0):
    """
    Calculate the next step of the system based on the current state.

    Parameters
    -----
    points : numpy array of arrays
        Current (x,y) coordinates of the seed points (Voronoi centers)
    regions: numpy array of arrays
        Finite Voronoi regions computed by voronoi_finite_polygons_2d
    vertices : numpy array of arrays
        Array ox (x,y) coordinates for the vertices of the finite Voronoi regions
    angles : numpy array
        Polarity angles for each seed/region, in radians (mapped over 2pi)
    polarities : numpy array of arrays
        Polarity vector [cos(angle);sin(angle)] for each seed/region
    boxsize : double
        Size of the observation box
    angular_noise : double
        D_r parameter to tune the intensity of the random rotational diffusion
    delta_t : double
        Value of the time step for the Euler integration of the equation of motion
    delta_r : double
        Value of the spatial step for the "Euler" integration of the force.
    K_A : float
        Area "stiffness"
    K_P : float
        Perimeter "stiffness"
    A_0 : float
        Target value for area
    P_0 : float
        Target value for perimeter

    Returns
    -----
    points_next : numpy array of arrays
        Next values of (x,y) coordinates for the seed points of the Voronoi regions
    angles_next : numpy array
        Next values of polarity angles in radians
    polarities_next : numpy array of arrays
        Next values of (x,y) components of polarity vectors
    velocities_vector : numpy array
        Array of (x,y) components of the velocity vectors associated to moving seed points
    ellipses_major_axis_angles_polarities : numpy array
        [cos(psi), sin(psi)] with between the major axes of fitted ellipses and the x-axis
    """

    # Compute force for every cell
    force_now = compute_every_force_on_cell_euler(points, regions, vertices, delta_r, K_A, K_P, A_0, P_0)
    # Compute motion for every cell
    points_next, velocities_vector = compute_every_cell_motion(points, force_now, polarities, v_0, mu, delta_t, boxsize)
    # Possibly raise some exceptions ?

    # XXX : Compute a fitted ellipse for every cell
    semi_minor_axes, semi_major_axes, ellipses_major_axis_angles = fit_ellipses.fit_ellipses_to_cells(regions, vertices,plot_on=False)
    # XXX : Compute the next angle after applying random rotational diffusion
    # angles_next = compute_every_random_rotational_diffusion_euler(angles, angular_noise, delta_t)

    # XXX : Use an alignment rule to align the polarity vector and the fitted ellipse major axis.
    angles_next = compute_angle_alignment_long_axis(angles, ellipses_major_axis_angles, delta_t, angular_noise, interaction_strength=1)
    # Compute the new polarity vector
    polarities_next = generate_every_polarity_vector(angles_next)

    ellipses_major_axis_angles_polarities = np.zeros([len(ellipses_major_axis_angles),2])
    ellipses_major_axis_angles_polarities[:,0] = np.cos(ellipses_major_axis_angles)
    ellipses_major_axis_angles_polarities[:,1] = np.sin(ellipses_major_axis_angles)

    return points_next, angles_next, polarities_next, velocities_vector, ellipses_major_axis_angles_polarities


def create_force_csv_log_file(force_logger):
    """
    Writing the force logger into a file

    Parameters
    -----
    force_logger: numpy array
        Contains the norm of the force exerted on each cell, for every iteration

    Returns
    ------
    Nothing
    """
    force_log_name = "force_log.csv"
    force_log_name = os.path.join(GLOBAL_LOG_PATH, force_log_name)
    with open(force_log_name,'w',newline='') as csvfile:
        force_log_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in force_logger:
            force_log_writer.writerow(row)


def create_tissue_energy_csv_log_file(tissue_energy_vector):
    """
    Log the tissue energy for every iteration. NOTE : THIS IS NOT ACTUALLY USING THE CSV WRITER FUNCTION

    Parameters
    -----
    tissue_energy_vector : numpy array
        The i-th element is the tissue energy at the i-th iteration

    Returns
    ------
    Nothing
    """
    # Create a specific function to avoid having multiple path
    # path = "./simulation_test_trash/"
    path = GLOBAL_LOG_PATH
    tissue_energy_vector_name = "tissue_energy_log.csv"
    tissue_energy_vector_name = os.path.join(path, tissue_energy_vector_name)
    with open(tissue_energy_vector_name,'a',newline='') as logfile:
        for energy in tissue_energy_vector:
            logfile.write("{}\n".format(energy))


# def load_computation_state()


# if __name__ == '__main__':
def main_spv():

    # Start timer
    startTime = datetime.now()

    # Number of step between two plots
    step_between_plots = 10

    # Initialize global variables
    global point_number
    global boxsize
    global iteration_number
    global delta_t
    global delta_r
    # Target parameters


    global p_0
    global A_bar


    # Cell parameters


    global v_0
    global mu
    global f_0
    global angular_noise


    cdef double K_A
    cdef double K_P
    cdef double A_0
    cdef double P_0

    #### DEFAULT parameters when no arguments are passed when running the script
    # Simulation box parameters
    DEFAULT_point_number = 100
    DEFAULT_boxsize = 10
    DEFAULT_iteration_number = 1
    DEFAULT_delta_t = 0.001
    # DEFAULT_delta_t = 0.001
    # DEFAULT_delta_r = 1
    DEFAULT_delta_r = DEFAULT_boxsize/1000.0
    # Target parameters
    DEFAULT_P_0 = 2.0* np.sqrt(np.pi)
    DEFAULT_A_0 = 1.0
    DEFAULT_p_0 = DEFAULT_P_0 / (DEFAULT_A_0)**0.5 # Might be confusing with P_0 ?
    # Should do a try except ZeroDivisionError to avoid division problem
    DEFAULT_A_bar = float(DEFAULT_boxsize)**2/float(DEFAULT_point_number)

    # Cell parameters
    DEFAULT_K_A = 1.0 # TO CHANGE
    DEFAULT_K_P = 1.0 # TO CHANGE
    DEFAULT_v_0 = 0.0 # TO CHANGE
    DEFAULT_mu = 1.0
    DEFAULT_f_0 = DEFAULT_v_0/DEFAULT_mu
    DEFAULT_angular_noise = 0.0

    # Default parameter flag
    DEFAULT_param_flag = False

    #### Test if arguments are passed when running the script
    if not len(sys.argv) > 1:
        print("Load defaut arguments")
        DEFAULT_param_flag = True
        # If no arguments have been passed, load the default values
        # Simulation box parameters
        point_number = DEFAULT_point_number
        boxsize = DEFAULT_boxsize
        iteration_number =  DEFAULT_iteration_number
        delta_t = DEFAULT_delta_t
        delta_r = DEFAULT_delta_r
        # Target parameters
        P_0 = DEFAULT_P_0
        A_0 = DEFAULT_A_0
        p_0 = DEFAULT_p_0
        A_bar = DEFAULT_A_bar

        # Cell parameters
        K_A = DEFAULT_K_A
        K_P = DEFAULT_K_P
        v_0 = DEFAULT_v_0
        mu = DEFAULT_mu
        f_0 = DEFAULT_f_0
        angular_noise = DEFAULT_angular_noise

        print(point_number,boxsize,iteration_number,delta_t,delta_r,P_0,A_0,p_0,A_bar,K_A,v_0,K_P,mu,f_0,angular_noise)


    else:
        ### Encapsulate into a try block to avoid crashing everything
        print("Loading passed arguments")
        try:
            # Arguments have been passed
            DEFAULT_param_flag = False
            # Simulation box parameters
            point_number = int(sys.argv[1])
            boxsize = float(sys.argv[2])
            iteration_number = int(sys.argv[3])
            delta_t = float(sys.argv[4])
            delta_r = float(sys.argv[5])
            # Target parameters
            P_0 = float(sys.argv[6])
            A_0 = float(sys.argv[7])
            p_0 = P_0 / (A_0)**0.5 # Might be confusing with P_0 ?
            A_bar = (boxsize)**2/point_number


            # Cell parameters
            K_A = float(sys.argv[8])
            K_P = float(sys.argv[9])
            v_0 = float(sys.argv[10])
            mu = float(sys.argv[11])
            f_0 = v_0/mu
            angular_noise = float(sys.argv[12])
            print(point_number,boxsize,iteration_number,delta_t,delta_r,P_0,A_0,p_0,A_bar,K_A,v_0,K_P,mu,f_0,angular_noise)



        except IndexError:
            # Probably many more possible errors
            print("Not enough parameters, loading default values")
            DEFAULT_param_flag = True
            # Simulation box parameters
            point_number = DEFAULT_point_number
            boxsize = DEFAULT_boxsize
            iteration_number =  DEFAULT_iteration_number
            delta_t = DEFAULT_delta_t
            delta_r = DEFAULT_delta_r
            # Target parameters
            P_0 = DEFAULT_P_0
            A_0 = DEFAULT_A_0
            p_0 = DEFAULT_p_0
            A_bar = DEFAULT_A_bar

            # Cell parameters
            K_A = DEFAULT_K_A
            K_P = DEFAULT_K_P
            v_0 = DEFAULT_v_0
            mu = DEFAULT_mu
            f_0 = DEFAULT_f_0
            angular_noise = DEFAULT_angular_noise

            print(point_number,boxsize,iteration_number,delta_t,delta_r,P_0,A_0,p_0,A_bar,K_A,v_0,K_P,mu,f_0,angular_noise)




    print(point_number,boxsize,iteration_number,delta_t,delta_r,P_0,A_0,p_0,A_bar,K_A,v_0,K_P,mu,f_0,angular_noise)
    print(('-'*50))


    # Path for image and logfile saving
    path = "./point_number_{}__boxsize_{}__iteration_number_{}/delta_t_{}__delta_r_{}/P_0_{:.3f}__A_0_{}__p_0_{:.3f}__A_bar_{:.3f}__K_A_{}__v_0_{}__K_P_{}__mu_{}__f_0_{}__angular_noise_{}/".format(point_number,boxsize,iteration_number,delta_t,delta_r,P_0,A_0,p_0,A_bar,K_A,v_0,K_P,mu,f_0,angular_noise)

    # XXX Bricolage
    # Modifify the GLOBAL_LOG_PATH to prefix it with the newly created path
    global GLOBAL_LOG_PATH
    GLOBAL_LOG_PATH = os.path.join(path, GLOBAL_LOG_PATH)

    # path = './simulation_test_trash/'
    ## Create a directory using the parameters, only if it doesn't already exist
    # TODO Plan a contingency to avoid overwriting data ?
    try:
        os.makedirs(GLOBAL_LOG_PATH)
        print(" Directory creation successfully completed")
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error during mkdir")
            raise


    filenames = [] # Empty list storing filenames
    # HACK
    tissue_energy_vector = [] # Empty list to store the tissue energy at every iteration
    # Create a logfile to register the status ?
    logfilename = "logfile.txt"
    logfilename = os.path.join(GLOBAL_LOG_PATH, logfilename)

    # Write if default or loaded arguments were used
    with open(logfilename,'a') as logfile:
        if(DEFAULT_param_flag == False):
            logfile.write("Loaded passed arguments \n")
        else:
            logfile.write("Loaded default arguments \n")
        logfile.write("point_number = {}\n".format(point_number))
        logfile.write("boxsize = {}\n".format(boxsize))
        logfile.write("iteration_number = {}\n".format(iteration_number))
        logfile.write("delta_t = {}\n".format(delta_t))
        logfile.write("delta_r = {}\n".format(delta_r))
        logfile.write("P_0 = {}\n".format(P_0))
        logfile.write("A_0 = {}\n".format(A_0))
        logfile.write("p_0 = {}\n".format(p_0))
        logfile.write("A_bar = {}\n".format(A_bar))
        logfile.write("K_A = {}\n".format(K_A))
        logfile.write("v_0 = {}\n".format(v_0))
        logfile.write("K_P = {}\n".format(K_P))
        logfile.write("mu = {}\n".format(mu))
        logfile.write("f_0 = {}\n".format(f_0))
        logfile.write("angular_noise = {}\n".format(angular_noise))


    # XXX Known random seed
    random_seed = np.random.randint(0,100)
#    random_seed = 3

    with open(logfilename,'a') as logfile:
        logfile.write("random_seed = {} \n".format(random_seed))
    np.random.seed(random_seed)
    ###### Random case ######
    # points_f = np.random.rand(15,2)*boxsize
    ### Build the first Voronoi tesselation
    # XXX Disable random initialization
#    points, angles, polarities  = initialization(point_number,boxsize)

    points, angles, polarities  =  lattice_initialization(point_number, boxsize, lattice_noise=0)

    ## One cell displaced
    # points[55] = points[55] + np.array([0.5,0])

    ## Hexagonal lattice generation
#    points, angles, polarities  = hexagonal_lattice_initialization(point_number,boxsize, lattice_noise=0)

    # Copy the seeds to generate a periodic tesselation around the observation box
    extended_points = generate_periodic_voronoi_cells_2(points, boxsize)
    # DEBUG
    # extended_points = generate_periodic_voronoi_cells_2(points, boxsize)

    # Compute the tesselation on the extended_points
    vor = Voronoi(extended_points)
    # vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor, point_number=point_number)

    plot_polygons(vor, regions, vertices)
    plot_contour(boxsize)

    # Limit the perimeter and area computation to the original points
    # I strongly pray that the points' indexing and the regions' indexing match and stay this way
    # If there is a bug, it's here. Sorry future me.
    # Note : still load every vertices. Probably possible to find the index of the maximum vertex in regions and load only up to
    # this vertex, but this is for another time.
    # vertices_max_index = regions[0:point_number].max()
    # load vertices[0:(vertices_max_index+1)]
    # indx, perimeters, areas = get_polygons_properties_2d(regions[0:point_number],vertices)
    # indx, perimeters, areas = get_polygons_properties_2d(regions,vertices)


    # Isolate the subset region of interest ?

    print("YOUUUU SOROOOOOO")

    for iteration in range(iteration_number):
        print("Iteration {0}".format(iteration))
        # DEBUG ? Compute with the extended points
        # points_next, angles_next, polarities_next, velocities_vector = compute_next_system_step(extended_points, regions, vertices,
        #                                                                      angles, polarities, boxsize, angular_noise, delta_t, delta_r)

        points_next, angles_next, polarities_next, velocities_vector, ellipses_major_axis_angles_polarities = compute_next_system_step(points, regions[0:point_number], vertices,
                                                                             angles[0:point_number], polarities[0:point_number], boxsize, angular_noise, delta_t, delta_r, K_A, K_P, A_0, P_0)

        # points_next, angles_next, polarities_next, velocities_vector = compute_next_system_step(points, regions, vertices,
        #                                                                      angles, polarities, boxsize, angular_noise, delta_t, delta_r)

        # Generate periodic cells
        # This will overwrite the previously computed point, and this might be a problem or not.
        extended_points_next = generate_periodic_voronoi_cells_2(points_next, boxsize)
        # DEBUG
        # extended_points_next = generate_periodic_voronoi_cells_2(points_next, boxsize)

        # Recompute the voronoi tesselation and the finite regions
        vor_next = Voronoi(extended_points_next)
        # vor_next = Voronoi(points_next)
        regions_next, vertices_next = voronoi_finite_polygons_2d(vor_next, point_number=point_number)

        ### Figure part
        if(iteration % step_between_plots == 0):
            # Create a figure for plotting
            plt.figure("Iteration {0}".format(iteration))
            plot_polygons(vor_next, regions_next, vertices_next)
            # Plot the Seeds
            plt.plot(points_next[1:,0], points_next[1:,1], "+k")
            # Tag some seeds
            plt.plot(points_next[0][0], points_next[0][1], '+r')
            plt.plot(points_next[1][0], points_next[1][1], '+b')
            plt.plot(points_next[2][0], points_next[2][1], '+g')
            plt.plot(points_next[3][0], points_next[3][1], '+y')
            # Quiver plot for velocities
            plt.quiver(points_next[:,0], points_next[:,1], velocities_vector[:,0], velocities_vector[:,1], color='black')
            # TODO : Quiver plot for fitted ellipses angles
#            plt.quiver(points_next[:,0], points_next[:,1],  ellipses_major_axis_angles_polarities[:,0], ellipses_major_axis_angles_polarities[:,1],color='green')
            # Make a nice plot
            plt.title("Iteration #{0}, delta_t = {1}".format(iteration,delta_t))
            plt.xlim(-boxsize/2.0, boxsize/2.0)
            plt.ylim(-boxsize/2.0, boxsize/2.0)

            # Save the current figure for future gif making:
            filename = 'image{0}.png'.format(iteration)
            filename = os.path.join(path, filename)
            plt.savefig(filename)
            filenames.append(filename) # Store the file name in the list

            # Once the figure is saved, close it to avoid excessive memory consumption
            plt.close()

        ### Write in the logfile
        with open(logfilename,'a') as logfile:
            logfile.write("Iteration {0}\n".format(iteration))

        create_csv_log_file("seed_point_log.csv", points_next)


        # Compute the new perimeters and areas to log them and compute the tissue energy
        indx, perimeters_next, areas_next = get_polygons_properties_2d(regions_next[0:point_number],vertices_next)

        create_csv_log_file("areas_log.csv",areas_next)
        create_csv_log_file("perimeters_log.csv",perimeters_next)
        create_csv_log_file("ellipses_major_axes_angles.csv", ellipses_major_axis_angles_polarities)

        tissue_energy_vector.append(compute_every_cell_energy(perimeters_next, areas_next, K_A, K_P, A_0, P_0))

        ### Update part
        #Update points, regions, vertices, angles, polarities with the "next" value for the following iteration
        points = points_next
        regions = regions_next
        vertices = vertices_next
        angles = angles_next
        polarities = polarities_next



    print("Sakusen Kanryou !")

    # Writing the force logger into a file
    create_force_csv_log_file(force_logger)
    # Writing the tissue energy logger
    create_tissue_energy_csv_log_file(tissue_energy_vector)
    # End timer
    endTime = datetime.now()
    # Append the time elapsed in the logfile
    with open(logfilename, "a") as logfile:
        logfile.write("Time elapsed : {}".format(endTime-startTime))


# Temporary main for debugging purposes
if __name__=='__main__':
    main_spv()
