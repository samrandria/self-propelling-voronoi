"""
Read force and energy log
"""

from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
import csv
import os
from my_file_tools import get_parameter_regexp
# FIXME : make a graphical toolbox file
import read_area_perimeter_log as rapl

def read_force_log(force_log_filename):
    """
    Get a numpy array with the norm of the force of every cell for every iteration

    Parameters
    -----
    force_log_filename : string
        Name of the file containing all the forces (assumed that path is already joined)
    
    Returns
    -----
    force_read : numpy array
        Norm of the force for each cell at every iteration    

    """
    # Open the file
    with open(force_log_filename,'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
        # We now have a list of strings like [['1.1 2.2 3.3'],['4.4 5.5 6.6'], ['7.7 8.8 9.9']]


    # Split the strings into  [['1.1', '2.2', '3.3'],['4.4','5.5', '6.6'], ['7.7', '8.8', '9.9']]
    data_read = [row[0].split() for row in data_read]

    # Convert each string element into a float
    force_read = []    
    for row in data_read:
        force_read.append([float(elem) for elem in row])

    # Now convert the list into a numpy array
    force_read = np.array(force_read)

    return force_read




def plot_mean_force(force_vector, delta_t):
    """
    Plot the mean force over every iteration

    Parameters
    -----
    force_vector : numpy array 
        Array containing the force of each cell for every iteration
    delta_t : float
        Time step between two iterations

    Returns
    -----
    fig : Figure for the plot
    """
    ########################
    # Plot the stuff
    nb_iter = len(force_vector)

    # Compute the average force at every iteration
    mean_force_vector = np.mean(force_vector, axis=1)

    # Create an evenly spaced time vector
    time_vector = np.arange(0, nb_iter*delta_t, delta_t)

    # Plotting part
    fig = plt.figure("Mean force of the system over time")
    plt.plot(time_vector, mean_force_vector, 'r+')
    plt.title("Mean force of the system over time")
    plt.xlabel("Time")
    plt.ylabel("Force norm (a.u)")
    # plt.show()
    return fig


def read_tissue_energy_log(tissue_energy_log_filename):
    """
    Get the a numpy array with the total tissue energy for every iteration
    
    Parameters
    -----
    tissue_energy_log_filename : string
    Name of the file containing all the tissue energy (assumed that path is already joined)


    Returns
    -----
    tissue_energy_read : numpy array
        Total tissue energy at every iteration    
    """

    with open(tissue_energy_log_filename, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
        # We now have a list of strings like [['1.2'],['3.4'], ['5.6']]

    # Convert each string element into a float
    tissue_energy_read = []
    for row in data_read:
        tissue_energy_read.append([float(elem) for elem in row])

    # Now convert the list into a numpy array
    tissue_energy_read = np.array(tissue_energy_read)

    return tissue_energy_read

def plot_tissue_energy_log(tissue_energy_vector, delta_t):
    """
    Plot the tissue energy over every iteration

    Parameters
    -----
    tissue_energy_vector : numpy array 
        Array containing the tissue energy of each cell for every iteration
    delta_t : float
        Time step between two iterations

    Returns
    -----
    fig : Figure for the plot
    """
    ########################
    # Plot the stuff
    nb_iter = len(tissue_energy_vector)

    # Create an evenly spaced time vector
    time_vector = np.arange(0, nb_iter*delta_t, delta_t)

    # Plotting part
    fig = plt.figure("Tissue energy of the system over time")
    plt.plot(time_vector, tissue_energy_vector, 'r+')
    plt.title("Tissue energy of the system over time")
    plt.xlabel("Time")
    plt.ylabel("Tissue energy (a.u)")
    # plt.show()
    return fig




if __name__ == '__main__':

    path = "./"
    logfilename = "logfile.txt"
    logfilename = os.path.join(path, logfilename)
    force_log_filename = "force_log.csv"
    force_log_filename = os.path.join(path, force_log_filename)
    tissue_energy_log_filename = "tissue_energy_log.csv"
    tissue_energy_log_filename = os.path.join(path, tissue_energy_log_filename)
    delta_t = get_parameter_regexp(logfilename, "delta_t")

    ## Force part
    force_vector = read_force_log(force_log_filename)
    figure_mean_force = plot_mean_force(force_vector, delta_t)
    rapl.save_fig_as(path, "mean_force.png")
    plt.show(figure_mean_force)

    ## Tissue energy part
    tissue_energy_vector = read_tissue_energy_log(tissue_energy_log_filename)
    figure_tissue_energy = plot_tissue_energy_log(tissue_energy_vector, delta_t)
    rapl.save_fig_as(path, "tissue_energy.png")
    plt.show(figure_tissue_energy)