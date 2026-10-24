"""
Read area log
"""

from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
import csv
import os
import MSD
from my_file_tools import save_fig_as, get_parameter_regexp


def read_area_log_modified_x(zone_to_print):
    # Get the path 
    path = './'
    filename = 'areas_log_modified_x.csv'
    filename = os.path.join(path, filename)

    # Open the file
    with open(filename,'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
        # We now have a list of strings like [['1.1 2.2 3.3'],['4.4 5.5 6.6'], ['7.7 8.8 9.9']]


    # Split the strings into  [['1.1', '2.2', '3.3'],['4.4','5.5', '6.6'], ['7.7', '8.8', '9.9']]
    data_read = [row[0].split() for row in data_read]

    # Convert each string element into a float
    # #Black_magic
    float_data_read = []    
    for row in data_read:
        float_data_read.append([float(elem) for elem in row])

    # Now convert the list into a numpy array
    float_data_read = np.array(float_data_read)

    # Plot the stuff
    nb_iter = range(len(float_data_read))
    for iteration in range(len(float_data_read)):
        plt.plot(nb_iter,float_data_read[:,zone_to_print], label="iteration {}, index {}".format(iteration,0))
            # for index in range(len(float_data_read[0])):
                    # plt.plot(nb_iter,float_data_read[:,index], label="iteration {}, index {}".format(iteration,index))
                    # plt.legend()

    # Display it 
    plt.title("Area after modification in x variation, Zone {}".format(zone_to_print))
    plt.xlabel("Line number")
    plt.ylabel("Area (a.u)")
    plt.show()

def read_area_log_modified_y(zone_to_print):
    # Get the path 
    path = './'
    filename = 'areas_log_modified_y.csv'
    filename = os.path.join(path, filename)

    # Open the file
    with open(filename,'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
        # We now have a list of strings like [['1.1 2.2 3.3'],['4.4 5.5 6.6'], ['7.7 8.8 9.9']]


    # Split the strings into  [['1.1', '2.2', '3.3'],['4.4','5.5', '6.6'], ['7.7', '8.8', '9.9']]
    data_read = [row[0].split() for row in data_read]

    # Convert each string element into a float
    # #Black_magic
    float_data_read = []    
    for row in data_read:
        float_data_read.append([float(elem) for elem in row])

    # Now convert the list into a numpy array
    float_data_read = np.array(float_data_read)

    # Plot the stuff
    nb_iter = range(len(float_data_read))
    for iteration in range(len(float_data_read)):
        plt.plot(nb_iter,float_data_read[:,zone_to_print], label="iteration {}, index {}".format(iteration,0))
            # for index in range(len(float_data_read[0])):
                    # plt.plot(nb_iter,float_data_read[:,index], label="iteration {}, index {}".format(iteration,index))
                    # plt.legend()

    # Display it 
    plt.title("Area after modification in y variation, Zone {}".format(zone_to_print))
    plt.xlabel("Line number")
    plt.ylabel("Area (a.u)")
    plt.show()

def read_area_log(areas_log_filename):
    """
    Get a numpy array with the areas of every cell for every iteration

    Parameters
    -----
    areas_log_filename : string
        Name of the file containing all the areas (assumed that path is already joined)
    
    Returns
    -----
    areas_read : numpy array
        Areas for each cell at every iteration    
    """
    # Open the file
    with open(areas_log_filename,'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
        # We now have a list of strings like [['1.1 2.2 3.3'],['4.4 5.5 6.6'], ['7.7 8.8 9.9']]


    # Split the strings into  [['1.1', '2.2', '3.3'],['4.4','5.5', '6.6'], ['7.7', '8.8', '9.9']]
    data_read = [row[0].split() for row in data_read]

    # Convert each string element into a float
    areas_read = []    
    for row in data_read:
        areas_read.append([float(elem) for elem in row])

    # Now convert the list into a numpy array
    areas_read = np.array(areas_read)

    return areas_read


def plot_mean_area(areas_vector, delta_t):
    """
    Plot the mean area over every iteration

    Parameters
    -----
    areas_vector : numpy array 
        Array containing the areas of each cell for every iteration
    delta_t : float
        Time step between two iterations

    Returns
    -----
    Nothing
    """
    nb_iter = len(areas_vector)

    mean_area_vector = np.mean(areas_vector, axis=1)

    # Create an evenly spaced time vector
    time_vector = np.arange(0, nb_iter*delta_t, delta_t)

    # Plotting part
    fig = plt.figure("Mean area of the system over time")
    plt.plot(time_vector, mean_area_vector, 'r+')
    plt.title("Mean area of the system over time")
    plt.xlabel("Time")
    plt.ylabel("Area (a.u)")
    # plt.show()
    return fig



################
def plot_indiv_area(areas_vector, delta_t, cell_to_plot=0):
    """
    Plot the area of a single cell at every iteration

    Parameters
    -----
    areas_vector : numpy array 
        Array containing the areas of each cell for every iteration
    delta_t : float
        Time step between two iterations
    cell_to_plot : int (optional)
        Index of the cell whose area will be plotted
    Returns
    -----
    Nothing
    """
    nb_iter = len(areas_vector)
    
    # Create an evenly spaced time vector
    time_vector = np.arange(0, nb_iter*delta_t, delta_t)

    # Plotting part
    fig = plt.figure("Area of cell {} over time".format(cell_to_plot))
    plt.plot(time_vector, areas_vector[:,cell_to_plot], 'r-')
    plt.title("Area of cell {} over time".format(cell_to_plot))
    plt.xlabel("Time")
    plt.ylabel("Area (a.u)")
    # plt.show()
    return fig

def read_perimeter_log(perimeters_log_filename):
    """
    Get a numpy array with the perimeters of every cell for every iteration

    Parameters
    -----
    perimeters_log_filename : string
        Name of the file containing all the perimeters (assumed that path is already joined)
    
    Returns
    -----
    perimeters_read : numpy array
        Perimeters for each cell at every iteration    

    """
    # Open the file
    with open(perimeters_log_filename,'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
        # We now have a list of strings like [['1.1 2.2 3.3'],['4.4 5.5 6.6'], ['7.7 8.8 9.9']]


    # Split the strings into  [['1.1', '2.2', '3.3'],['4.4','5.5', '6.6'], ['7.7', '8.8', '9.9']]
    data_read = [row[0].split() for row in data_read]

    # Convert each string element into a float
    perimeters_read = []    
    for row in data_read:
        perimeters_read.append([float(elem) for elem in row])

    # Now convert the list into a numpy array
    perimeters_read = np.array(perimeters_read)

    return perimeters_read


def plot_mean_perimeter(perimeters_vector, delta_t):
    """
    Plot the mean perimeter over every iteration

    Parameters
    -----
    perimeters_vector : numpy array 
        Array containing the perimeters of each cell for every iteration
    delta_t : float
        Time step between two iterations

    Returns
    -----
    Nothing
    """

    nb_iter = len(perimeters_vector)

    mean_perimeter_vector = np.mean(perimeters_vector, axis=1)

    # Create an evenly spaced time vector
    time_vector = np.arange(0, nb_iter*delta_t, delta_t)

    # Plotting part
    fig = plt.figure("Mean perimeter of the system over time")
    plt.plot(time_vector, mean_perimeter_vector, 'r+')
    plt.title("Mean perimeter of the system over time")
    plt.xlabel("Time")
    plt.ylabel("Perimeter (a.u)")
    # plt.show()
    return fig

def plot_indiv_parameter(perimeters_vector, delta_t, cell_to_plot=0):
    """
    Plot the perimeter of a single cell at every iteration

    Parameters
    -----
    perimeters_vector : numpy array 
        Array containing the perimeters of each cell for every iteration
    delta_t : float
        Time step between two iterations
    cell_to_plot : int (optional)
        Index of the cell whose perimeter will be plotted
    Returns
    -----
    Nothing
    """
    nb_iter = len(perimeters_vector)
    
    # Create an evenly spaced time vector
    time_vector = np.arange(0, nb_iter*delta_t, delta_t)

    fig = plt.title("Perimeter of cell {} over time".format(cell_to_plot))
    plt.plot(time_vector, perimeters_vector[:,cell_to_plot], 'r-')
    plt.title("Perimeter of cell {} over time".format(cell_to_plot))
    plt.xlabel("Time")
    plt.ylabel("Perimeter (a.u)")
    # plt.show()
    return fig



if __name__ == '__main__':
    # read_area_log_modified_x(1)
    # read_area_log_modified_y(1)
    # read_area_log_indiv(0)

    path = "./"
    logfilename = "logfile.txt"
    logfilename = os.path.join(path, logfilename)
    areas_log_filename = "areas_log.csv"
    areas_log_filename = os.path.join(path, areas_log_filename)

    perimeters_log_filename = "perimeters_log.csv"
    perimeters_log_filename = os.path.join(path, perimeters_log_filename)

    delta_t = get_parameter_regexp(logfilename, "delta_t")

    # FIXME
    cell_to_plot=0

    ## Part area
    areas_vector = read_area_log(areas_log_filename)
    figure_mean_area = plot_mean_area(areas_vector, delta_t)
    save_fig_as(path, "mean_area.png")
    plt.show(figure_mean_area)

    figure_indiv_area = plot_indiv_area(areas_vector, delta_t,cell_to_plot=cell_to_plot)
    save_fig_as(path, "indiv_area_cell_{}.png".format(cell_to_plot))
    plt.show(figure_indiv_area)
    ## Part perimeter
    perimeters_vector = read_perimeter_log(perimeters_log_filename)
    figure_mean_perimeter = plot_mean_perimeter(perimeters_vector, delta_t)
    save_fig_as(path, "mean_perimeter.png")
    plt.show(figure_mean_perimeter)
    figure_indiv_perimeter = plot_indiv_parameter(perimeters_vector, delta_t, cell_to_plot=cell_to_plot)
    save_fig_as(path,"indiv_perimeter_cell_{}.png".format(cell_to_plot))
    plt.show(figure_indiv_perimeter)
