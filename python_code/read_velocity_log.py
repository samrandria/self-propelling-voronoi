"""
Read the data from the velocity log file
"""

import csv
import numpy as np
import os
from my_file_tools import get_parameter_regexp, save_fig_as
import matplotlib.pyplot as plt

def read_velocities(filename):
    """
    Reads a file where velocities are logged like (with i : iteration, c : cell, x,y : coordinates)
    v_i_c_x  v_i_c_y
    v_i_c+1_x v_i_c+1_y 
    ...
    v_i+1_c_x  v_i+1_c_y

    Parameters
    -----
    filename : string
        name of the file to parse

    Returns 
    ------
    velocities_array : numpy array
        Array containing the velocities
    """

    filename = str(filename) 

    with open(filename,'r') as logfile:
        reader = csv.reader(logfile, delimiter=' ', quotechar='|')
        data_read = [row for row in reader]

    # Now looks like [['vx0', 'vy0'], ['vx1', 'vy1']...]
    # Convert the strings into floats for the x,y coordinates
    data_read_float = [ [float(row[0]), float(row[1])] for row in data_read ]

    # Now convert into a numpy array for easier manipulation
    data_read_array = np.array(data_read_float)

    return data_read_array



def plot_mean_velocity(velocities, point_number, iteration_number, delta_t):
    """
    Plot the mean velocity over every iteration

    Parameters
    -----
    velocities_vector : numpy array 
        Array containing the velocities of each cell for every iteration
    delta_t : float
        Time step between two iterations

    Returns
    -----
    Nothing
    """

    # Break the velocity vector into "iteration_number" pieces of "point_number" elements (with x and y components)
    velocities_vector = np.reshape(velocities, [iteration_number, point_number,2])

    # FIXME Compute the norm 
    velocities_vector_norm = np.linalg.norm(velocities_vector, axis=2)
    mean_velocity_vector = np.mean(velocities_vector_norm, axis=1)

    # Create an evenly spaced time vector
    time_vector = np.arange(0, iteration_number*delta_t, delta_t)

    # Plotting part
    fig = plt.figure("Mean velocity of the system over time")
    plt.plot(time_vector, mean_velocity_vector, 'r+')
    plt.title("Mean velocity of the system over time")
    plt.xlabel("Time")
    plt.ylabel("Velocity (a.u)")
    # plt.show()
    return fig



################
def plot_indiv_velocity(velocities, point_number, iteration_number, delta_t, cell_to_plot=0):
    """
    Plot the velocity of a single cell at every iteration

    Parameters
    -----
    velocities_vector : numpy array 
        Array containing the velocities of each cell for every iteration
    delta_t : float
        Time step between two iterations
    cell_to_plot : int (optional)
        Index of the cell whose velocity will be plotted
    Returns
    -----
    Nothing
    """
    # Break the velocity vector into "iteration_number" pieces of "point_number" elements (with x and y components)
    velocities_vector = np.reshape(velocities, [iteration_number, point_number,2])

    velocities_vector_norm = np.linalg.norm(velocities_vector, axis=2)

    # Create an evenly spaced time vector
    time_vector = np.arange(0, iteration_number*delta_t, delta_t)

    # Plotting part
    # FIXME : plot the NORM INSTEAD WTH
    fig = plt.figure("Velocity of cell {} over time".format(cell_to_plot))
    plt.plot(time_vector, velocities_vector_norm[:,cell_to_plot], 'r-')
    plt.title("Velocity of cell {} over time".format(cell_to_plot))
    plt.xlabel("Time")
    plt.ylabel("Velocity (a.u)")
    # plt.show()
    return fig


if __name__ == '__main__':
#    path = "./simulation_test_trash/"
    path = "./"
    velocities_log_filename = "velocities_log.csv"
    velocities_log_filename = os.path.join(path,velocities_log_filename)
    logfilename = "logfile.txt"
    logfilename = os.path.join(path, logfilename)
   
    # Simulation parameters
    delta_t = get_parameter_regexp(logfilename, "delta_t")
    point_number = int(get_parameter_regexp(logfilename, "point_number"))
    iteration_number = int(get_parameter_regexp(logfilename, "iteration_number"))

    velocities = read_velocities(velocities_log_filename)

    figure_mean_velocity = plot_mean_velocity(velocities, point_number, iteration_number, delta_t)
    save_fig_as(path, "mean_velocity.png")
    plt.show(figure_mean_velocity)

    # XXX : change cell to plot if needed
    cell_to_plot = 0
    figure_indiv_velocity = plot_indiv_velocity(velocities, point_number, iteration_number, delta_t, cell_to_plot=cell_to_plot)
    save_fig_as(path, "individual_velocity_cell_{}.png".format(cell_to_plot))
    plt.show(figure_indiv_velocity)
