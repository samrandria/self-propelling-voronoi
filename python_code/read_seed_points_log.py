"""
Read the seed points coordinates in a csv file with the following format:
|[x0_0 y0_0]| |[x1_0 y1_0]| ...
|[x0_1 y0_1]| |[x1_1 y1_1]| ...

"""

from pathlib import Path
import csv
import numpy as np
import os

import matplotlib.pyplot as plt

from voronoi_tesselation import get_polygons_properties_2d, voronoi_finite_polygons_2d, plot_polygons
from scipy.spatial import voronoi_plot_2d, Voronoi, ConvexHull

from voronoi_tesselation import generate_periodic_voronoi_cells_2
from my_file_tools import get_parameter_regexp

def read_seed_points(filename):
    """
    Read the coordinates in a csv file with the following format:
    |[x0_0 y0_0]| |[x1_0 y1_0]| ...
    |[x0_1 y0_1]| |[x1_1 y1_1]| ...
    And convert them into a numpy array.

    Parameters
    -----
    filename : string
        Name of the csv log file

    Returns
    -----
    filename : string
    data_read_array : numpy array
        Parsed data from the csv log file. Every row contains the position of every seed points for an iteration. Each row represents an iteration.
    """

    # String conversion to avoid surprises
    fname = str(filename)

    # Looks like :
    # Iter 0 |[x0 y0]| |[x1 y1]| ....
    # Iter 1 |[x0 y0]| |[x1 y1]| ....

    with open(fname, 'r') as fp:
        reader = csv.reader(fp, delimiter=' ', quotechar='|')
        data_read = [row for row in reader]

    # Now looks like [['[x0 y0]', '[x1 y1]', ...  ]]
    # Remove the '[ ]'
    data_read_split = [[elem[1:-1].split() for elem in row] for row in data_read ]

    # Now looks like [['x0', 'y0'], ['x1', 'y1']...]
    # Convert the strings into floats for the x,y coordinates
    data_read_float = [[[float(x), float(y)] for x,y in elem ] for elem in data_read_split]

    # Now convert into a numpy array for easier manipulation
    data_read_array = np.array(data_read_float)

    return data_read_array


def get_state_at_iteration(state_array,iteration):
    """
    Extract a "state" from a seed_point_log where all the seed coordinates are given on one line for for one iteration

    Parameters
    -----
    state_array : numpy array
        Iterations extracted from a seed_point_log.csv file
    iteration : int
        Iteration at which we want to get the state

    Returns
    -----
    state_at_iteration : numpy array
        Coordinates of seed points at the specified iteration
    """
    try:
        state_at_iteration = state_array[iteration]
    except IndexError:
        print("Index out of bounds")

    return state_at_iteration


def reconstruct_state_at_iteration(filename, iteration, boxsize):
    """
    Reads from a seed_point_log.csv file and extracts the position of every seed points at a given iteration. Generate a periodic tesselation around the observation box.

    Note : this is a convenience function for one call. If multiple calls, use read_seed_points() once,
    and then call get_state_at_iteration and periodic boundaries as needed.

    Parameters
    -----
    filename : string
        Name of the seed poing log csv file.
    iteration : int
        Desired iteration to reconstruct
    boxsize : float

    Returns
    -----
    state : numpy array
        Array of seed points coordinates at the specified iteration
    extended_points : numpy array
        state + periodic boundaric conditions around the observation box
    """

    ## FIXME : Given that we know the desired iteration, I could specifically parse ONLY the desired lines
    ## knowing the size of an iteration. This would avoid computational overhead.

    # Just in case the boxsize is passed as an integer, convert it to float
    boxsize_float = float(boxsize)
    all_states = read_seed_points(filename)
    state = get_state_at_iteration(all_states, iteration)
    extended_points = generate_periodic_voronoi_cells_2(state, boxsize_float)

    return state, extended_points

def get_and_plot_tracked_trajectory(seed_point_log, cell_to_track_number=None, print_iteration_number_step=None, iteration_start=0):
    """
    Acquire the displacement of a cell center over time from a log file and plot it.

    Parameters
    -----
    point_coordinates_over_time : numpy array
        All the coordinates of a single cell over time

    Returns
    -----

    """

    all_states = read_seed_points(seed_point_log)

    if cell_to_track_number is None:
        print("No cell to track number entered : defaulting to cell #0")
        cell_to_track_number = 0
    else:
        pass


    try:
        point_coordinates_over_time = all_states[:,cell_to_track_number]


        if print_iteration_number_step is None:
            print("No step given to print the iteration number : defaulting to 1")
            print_iteration_number_step = 1
        elif print_iteration_number_step >= len(point_coordinates_over_time):
            print("The step given exceeds the total number of iteration : defaulting to 1")
            print_iteration_number_step = 1
        else:
            pass

        fig = plt.figure("Tracked trajectory of cell")
        plt.title("Tracked trajectory of cell {}".format(cell_to_track_number))

        for i in range(iteration_start,len(point_coordinates_over_time)):
            plt.plot(point_coordinates_over_time[i,0],point_coordinates_over_time[i,1],'b+')
            if(i%print_iteration_number_step==0):
                plt.text(point_coordinates_over_time[i,0],point_coordinates_over_time[i,1], str(i), color='red', fontsize=14)
        return fig
    except IndexError:
        print("Cell requested does not exist")


def get_and_plot_tracked_trajectory_all(seed_point_log, cell_to_track_number_list=None, print_iteration_number_step=None, iteration_start=0):
    """
    Acquire the displacement of a cell center over time from a log file and plot it.

    Parameters
    -----
    point_coordinates_over_time : numpy array
        All the coordinates of a single cell over time

    Returns
    -----

    """

    all_states = read_seed_points(seed_point_log)

    if cell_to_track_number_list is None:
        print("No cell to track number entered : defaulting to cell #0")
        cell_to_track_number_list = 0
    else:
        pass

    fig = plt.figure("Tracked trajectory of cell")
    plt.title("Tracked trajectory of cell {}".format(cell_to_track_number_list))

    lstyle_list = ['b+','r+','c+','g+']

    for cell_to_track_number,lstyle in zip(cell_to_track_number_list,lstyle_list):
        try:
            point_coordinates_over_time = all_states[:,cell_to_track_number]


            if print_iteration_number_step is None:
                print("No step given to print the iteration number : defaulting to 1")
                print_iteration_number_step = 1
            elif print_iteration_number_step >= len(point_coordinates_over_time):
                print("The step given exceeds the total number of iteration : defaulting to 1")
                print_iteration_number_step = 1
            else:
                pass


            plt.plot(point_coordinates_over_time[iteration_start:,0],point_coordinates_over_time[iteration_start:,1], lstyle, label="Cell {}".format(cell_to_track_number))
            for i in range(iteration_start,len(point_coordinates_over_time)):
                if(i%print_iteration_number_step==0):
                    plt.text(point_coordinates_over_time[i,0],point_coordinates_over_time[i,1], str(i), color='black', fontsize=14)

        except IndexError:
            print("Cell requested does not exist")

    return fig


if __name__ == '__main__':
    path = "./"
    seed_point_log = "seed_point_log.csv"
    seed_point_log = os.path.join(path, seed_point_log)

    logfilename = "logfile.txt"
    logfilename = os.path.join(path, logfilename)

    boxsize = get_parameter_regexp(logfilename, "boxsize")
    # boxsize = get_parameter_regexp(logfilename, "boxsize")

    ## Manual (better way)
    # data_read = read_seed_points(seed_point_log)
    # state = get_state_at_iteration(data_read, 0)
    # Generate the corresponding periodic boundaries
    # extended_points = generate_periodic_voronoi_cells_2(state_array_subset, 10)

    ## Convenient one call only way
    state, extended_points = reconstruct_state_at_iteration(seed_point_log, 0, 10.0)
    # # Reconstruct the Voronoi tesselation as a POC
    vor = Voronoi(extended_points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Plotting part
#    plot_polygons(vor,regions,vertices)
#    plt.plot(state[:,0], state[:,1], 'k+')
#    plt.plot(state[0][0], state[0][1], '+r')
#    plt.plot(state[1][0], state[1][1], '+b')
#    plt.plot(state[2][0], state[2][1], '+g')
#    plt.plot(state[3][0], state[3][1], '+y')
#    plt.xlim(-10.0/2.0, 10.0/2.0)
#    plt.ylim(-10.0/2.0, 10.0/2.0)
#
#    plt.show()


    #### Test trajectory tracker
    cell_to_track_number = 70
    print_iteration_number_step = 100
    fig = get_and_plot_tracked_trajectory(seed_point_log, cell_to_track_number=70, print_iteration_number_step=print_iteration_number_step)
    plt.savefig("tracked_trajectory_cell_{}".format(cell_to_track_number))
    plt.show(fig)



    ### Label all points during an iteration
    iteration_to_reconstruct = 1700
    point_number = int(get_parameter_regexp(logfilename, "point_number"))

    reconstructed_state, reconstructed_extended_points = reconstruct_state_at_iteration(seed_point_log, iteration_to_reconstruct, 10)
    vor = Voronoi(reconstructed_extended_points)
    regions, vertices = voronoi_finite_polygons_2d(vor, point_number=point_number)
    plot_polygons(vor, regions, vertices)
    for i in range(point_number):
        plt.plot(reconstructed_extended_points[i,0], reconstructed_extended_points[i,1], 'k+')
        plt.text(reconstructed_extended_points[i,0], reconstructed_extended_points[i,1], str(i), color='red', fontsize=14)
    plt.xlim(-boxsize/2.0, boxsize/2.0)
    plt.ylim(-boxsize/2.0, boxsize/2.0)
    plt.show()



    ####
    cell_to_track_number_list = [95, 93, 65, 2]
    get_and_plot_tracked_trajectory_all(seed_point_log, cell_to_track_number_list, print_iteration_number_step=print_iteration_number_step, iteration_start=1500)
    plt.legend()
    cell_to_track_number_list_string = '_'.join(str(e) for e  in cell_to_track_number_list)
    plt.savefig("tracked_trajectory_cells_{}".format(cell_to_track_number_list_string))
    plt.show()
