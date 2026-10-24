"""
Mean Square Displacement calculator
If the seed points identity is conserved throughout the iterations (which is most likely the case visually, 
due to our colour trackers), then we can just parse the seed_point_log and compute difference in position from it ?).
This will have to take into account periodic boundary displacement.
"""

import os
# Regular expressions
import re

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.optimize

import read_area_perimeter_log as rapl
import read_seed_points_log as rspl
import read_velocity_log as rvl

from my_file_tools import save_fig_as, get_parameter_regexp
from pathlib import Path
from cycler import cycler

def compute_MSD_bad(states):
    """
    [DEPRECATED] Compute the Mean Square Displacement : 1/N * (sum_n->N (x_n(t) - x_n(0)**2)

    Parameters
    -----
    states : numpy array
        States extracted from a seed_point_log.csv file recording position of seed points at every iteration 

    Returns
    ----- 
    MSD : numpy array
        Mean Square Displacement of every voronoi cell over time
    """

    state_size = float(len(states[0]))

    ## Compute the center of mass at every iteration for the x and y coordinates
    x_center_mass = np.array([sum(state[:,0]) for state in states])
    x_center_mass = (1/state_size) * x_center_mass 
    y_center_mass = np.array([sum(state[:,1]) for state in states])
    y_center_mass = (1/state_size) * y_center_mass
    
    ## Compute the position in the frame of the center of mass

    # Maybe a more pythonic way ?
    x_prime_t = []
    for i in range(len(x_center_mass)):
        x_prime_t.append(states[i,:,0] - x_center_mass[i])

    # Conversion in numpy array
    x_prime_t = np.array(x_prime_t)


    # Maybe a more pythonic way ?
    y_prime_t = []
    for i in range(len(y_center_mass)):
        y_prime_t.append(states[i,:,1] - y_center_mass[i])

    # Conversion in numpy array
    y_prime_t = np.array(y_prime_t)


    # Get the points at time 0
    x_prime_0 = x_prime_t[0] 
    y_prime_0 = y_prime_t[0] 

    # Compute the MSD
    MSD_x_array = np.array([(x_prime - x_prime_0)**2 for x_prime in x_prime_t]) 
    MSD_y_array = np.array([(y_prime - y_prime_0)**2 for y_prime in y_prime_t]) 
    
    MSD_x_sum = np.sum(MSD_x_array, axis=1)
    MSD_y_sum = np.sum(MSD_y_array, axis=1)

    MSD = (MSD_x_sum + MSD_y_sum) * 1.0/len(MSD_x_sum)

    return MSD
    


def compute_MSD_velocity(logfile_name, velocity_filename, velocities, initial_positions):
    """
    [DEPRECATED] Compute the Mean Square Displacement : 1/N * (sum_n->N [delta_t * (v_n(t) - v_n(0)**2)])

    Parameters
    ----- 
    velocities : numpy array
        Velocity vectors for individual cells extracted from a velocities_log.csv file recording the velocity of each seed point at every iteration 
    initial_positions numpy array
        Reference for the (x,y) positions of the seed points (at iteration 0 or )

    Returns
    -----
    MSD : numpy array
        Mean Square Displacement of every voronoi cell over time
    """

    # TODO : check link below
    # https://stackoverflow.com/questions/31264591/mean-square-displacement-python#31266574
    # Extract relevant parameters from the logfile
    point_number = int(get_parameter_regexp(logfile_name, "point_number"))
    delta_t = float(get_parameter_regexp(logfile_name, "delta_t"))
    iteration_number = int(get_parameter_regexp(logfile_name, "iteration_number"))

    # Compute the "displacements" by multiplying the velocity vector by the time step
    displacements = velocities * delta_t

    # Create a cumulated displacement array -> Positions at time t, without the effects of periodic boundaries
    cum_array = np.zeros([point_number*iteration_number,2])
    cum_sum = np.zeros([point_number,2])

    for i in range(iteration_number):
        # Get the cumulated displacement for every cell after the i-th iteration
        cum_sum = cum_sum + displacements[i*point_number:(i+1)*point_number]
        # Insert it in the cumulated displacement array (with a little verbose formula to avoid simple reference assignement)
        cum_array[i*point_number:(i+1)*point_number] = cum_array[i*point_number:(i+1)*point_number] + cum_sum


    ## Compute the center of mass at every iteration for the x and y coordinates
    x_center_mass = np.zeros(iteration_number) 
    y_center_mass = np.zeros(iteration_number) 

    for i in range(iteration_number):
        x_center_mass[i] = np.sum(cum_array[i*point_number:(i+1)*point_number,0])
        y_center_mass[i] = np.sum(cum_array[i*point_number:(i+1)*point_number,1])

    # Average
    x_center_mass = np.mean(x_center_mass)
    y_center_mass = np.mean(y_center_mass)

    # Compute the r_prime = position - center mass
    x_prime_t = np.zeros([point_number*iteration_number])
    y_prime_t = np.zeros([point_number*iteration_number])

    for i, x_CM in enumerate(x_center_mass):
        x_prime_t[i*point_number:(i+1)*point_number] = cum_array[i*point_number:(i+1)*point_number,0] - x_CM

    for i, y_CM in enumerate(y_center_mass):
        y_prime_t[i*point_number:(i+1)*point_number] = cum_array[i*point_number:(i+1)*point_number,1] - y_CM
    
    # Substract the initial position, square the result and and take the average (again ?)
    delta_x_prime_t = np.zeros([point_number*iteration_number])
    delta_y_prime_t = np.zeros([point_number*iteration_number])

    for i in range(iteration_number):
        delta_x_prime_t[i*point_number:(i+1)*point_number] = (1.0/point_number) * (x_prime_t[i*point_number:(i+1)*point_number] - initial_positions[i,0])**2
        delta_y_prime_t[i*point_number:(i+1)*point_number] = (1.0/point_number) * (y_prime_t[i*point_number:(i+1)*point_number] - initial_positions[i,1])**2


    ## Fuse the x and y coordinates in a "MSD" vector
    MSD = delta_x_prime_t + delta_y_prime_t 

    return MSD


def compute_MSD(point_number, delta_t, iteration_number, velocities):
    """
    Basic version without computation within the center of mass frame

    Parameters
    -----
    point_number : int
        Number of cells in the system
    delta_t : float
        Timestep between two computations
    iteration_number : int
        Number of iteration of computation in the system
    velocities : numpy array
        Array containing the velocities logged for each cell at every iteration
    
    Returns
    -----
    MSD_vector : numpy array of iteration_number elements
        The i-th element is the MSD at the i-th iteration
    """
    
    # Break the velocity vector into "iteration_number" pieces of "point_number" elements (with x and y components)
    velocities_vector = np.reshape(velocities, [iteration_number, point_number,2])

    # Transform into displacement = speed * time 
    displacement_vector = velocities_vector * delta_t
    # Get the norm of the displacement in x and y
    diff_r = np.sqrt(displacement_vector[:,:,0]**2 + displacement_vector[:,:,1]**2)
    
    # Accumulate the displacements : gives the distance from "initial position" at time t
    cum_displacement_vector = np.cumsum(diff_r, axis=0)

    # Substract the initial position 
    cum_displacement_vector = cum_displacement_vector - cum_displacement_vector[0]

    # Raise to the square 
    squared_cum_displacement_vector = cum_displacement_vector**2

    # Sum the displacements of every cell within an iteration and average
    MSD_vector = np.mean(squared_cum_displacement_vector, axis=1)

    # Now each element of the vector represents the MSD at time t ?

    return MSD_vector

def compute_MSD_center_mass(point_number, delta_t, iteration_number, velocities, points_coordinates):
    """
    Version with computation within the center of mass frame

    Parameters
    -----
    point_number : int
        Number of cells in the system
    delta_t : float
        Timestep between two computations
    iteration_number : int
        Number of iteration of computation in the system
    velocities : numpy array
        Array containing the velocities logged for each cell at every iteration
    
    Returns
    -----
    MSD_vector : numpy array of iteration_number elements
        The i-th element is the MSD at the i-th iteration
    """
    
    # Break the velocity vector into "iteration_number" pieces of "point_number" elements (with x and y components)
    velocities_vector = np.reshape(velocities, [iteration_number, point_number,2])

    center_of_mass = 1.0/point_number * np.sum(points_coordinates, axis=1)
    
    coordinates_minus_center_of_mass = np.zeros([np.shape(points_coordinates)])
    for i in range(len(points_coordinates)):
        coordinates_minus_center_of_mass[i] = points_coordinates[i,:,:] - center_of_mass[i]

    ######## TODO  TO FINISH !!!!!!!!!######################

    # Transform into displacement = speed * time 
    displacement_vector = velocities_vector * delta_t
    # Get the norm of the displacement in x and y
    diff_r = np.sqrt(displacement_vector[:,:,0]**2 + displacement_vector[:,:,1]**2)
    
    # Accumulate the displacements : gives the distance from "initial position" at time t
    cum_displacement_vector = np.cumsum(diff_r, axis=0)

    # Substract the initial position 
    cum_displacement_vector = cum_displacement_vector - cum_displacement_vector[0]

    # Raise to the square 
    squared_cum_displacement_vector = cum_displacement_vector**2

    # Sum the displacements of every cell within an iteration and average
    MSD_vector = np.mean(squared_cum_displacement_vector, axis=1)

    # Now each element of the vector represents the MSD at time t ?

    return MSD_vector






def MSD_analysis(logfilename, velocities_log_filename):
    """
    Opens the necessary logs to compute the MSD

    Parameters
    -----
    logfilename : string
        Name of the logfile containing the parameters for the experiment (already joined with the path)
    velocities_log_filename : string
        Name of the file containing the log for the velocity of every cell at each iteration (already joined with the path)

    Returns
    -----
    MSD_vector : np array
        Vector with the Mean Sqaured Displacement at every iteration
    """
    # Load the simulation parameters
    point_number = int(get_parameter_regexp(logfilename, "point_number"))
    delta_t = float(get_parameter_regexp(logfilename, "delta_t"))
    iteration_number = int(get_parameter_regexp(logfilename, "iteration_number"))
    
    # Load the velocities from the log 
    velocities = rvl.read_velocities(velocities_log_filename)

    MSD_vector = compute_MSD(point_number, delta_t, iteration_number, velocities)

    return MSD_vector


def plot_MSD(MSD_vector, delta_t):
    """
    Plot the Mean Squared Displacement over every iteration

    Parameters
    -----
    MSD_vector : numpy array 
        Array containing the MSD for every iteration
    delta_t : float
        Time step between two iterations

    Returns
    -----
    fig : Figure for the plot
    """

    nb_iter = len(MSD_vector)

    ## Plot the MSD
    # Prepare a time vector to plot the MSD at every time step
    time_vector = np.arange(0, nb_iter*delta_t, delta_t)

    # Fitting
    popt_diffusive, pcov_diffusive = scipy.optimize.curve_fit(fit_diffusive, time_vector, MSD_vector)
    popt_ballistic, pcov_ballistic = scipy.optimize.curve_fit(fit_ballistic, time_vector, MSD_vector)

    fig = plt.figure("Evolution of MSD over time")
    plt.plot(time_vector,  MSD_vector, 'r+')
    plt.title("Evolution of MSD over time")
    plt.xlabel("Time")
    plt.ylabel("MSD")

    # XXX : Sometimes, the fit is not visually appropriate, especially when there are not so many values  
    plt.plot(time_vector, fit_diffusive(time_vector, *popt_diffusive), 'g-', label="Diffusive fit: a={:.3E}, b={:.3E}".format(*tuple(popt_diffusive)))
    plt.plot(time_vector, fit_ballistic(time_vector, *popt_ballistic), 'b-', label="Ballistic fit: a={:.3E}, b={:.3E}".format(*tuple(popt_ballistic)))
    
    # Print the legend
    plt.gca().legend()

    # plt.show()
    
    # XXX log of MSD


    return fig

## Try to fit a diffusive equation and a ballistic equation
def fit_diffusive(t, a, b):
    return np.sqrt(a*t) + b

def fit_ballistic(t, a, b):
    return a*t + b


def compute_orientational_orders(point_number, iteration_number, velocities):
    """
    Computes the average normalized velocity

    Parameters
    -----
    # TODO 
    
    Returns
    -----
    # TODO
    """
    # break the velocity vector into "iteration_number" pieces of "point_number" elements (with x and y components)
    # we assume that the array is regular with the same number of velocities during each iteration
    velocities_vector = np.reshape(velocities, [iteration_number, point_number,2])

     
    average_absolute_velocity = np.linalg.norm(velocities_vector, axis=2)
    average_absolute_velocity = np.mean(average_absolute_velocity, axis=1)

    orientational_orders = np.sum(velocities_vector, axis=1)
    orientational_orders = np.linalg.norm(orientational_orders, axis=1)
    orientational_orders = orientational_orders / (point_number * average_absolute_velocity[:])

    return orientational_orders

def plot_orientational_orders(orientational_orders, delta_t):
    """
    # TODO
    """
    nb_iter = len(orientational_orders)

    # Prepare a time vector to plot the orientational order at every time step
    time_vector = np.arange(0, nb_iter*delta_t, delta_t)

    fig = plt.figure("Evolution of orientational order over time")
    plt.plot(time_vector,  orientational_orders, 'g-')
    plt.title("Evolution of orientational order over time")
    plt.xlabel("Time")
    plt.ylabel(r"$\varphi$")

    return fig

def orientational_orders_analysis(logfilename, velocities_log_filename):
    """
    Opens the necessary logs to compute the orientational orders

    Parameters
    -----
    logfilename : string
        Name of the logfile containing the parameters for the experiment (already joined with the path)
    velocities_log_filename : string
        Name of the file containing the log for the velocity of every cell at each iteration (already joined with the path)

    Returns
    -----
    Orientational orders : np array
       Order parameter of the system. The closer it is to 1, the more ordered the system is. 
    """
    # Load the simulation parameters
    point_number = int(get_parameter_regexp(logfilename, "point_number"))
    delta_t = float(get_parameter_regexp(logfilename, "delta_t"))
    iteration_number = int(get_parameter_regexp(logfilename, "iteration_number"))
    
    # Load the velocities from the log 
    velocities = rvl.read_velocities(velocities_log_filename)

    orientational_orders = compute_orientational_orders(point_number, iteration_number, velocities)

    return orientational_orders

def compute_velocities_angles(point_number, iteration_number, velocities):
    """
    # TODO
    """
    # break the velocity vector into "iteration_number" pieces of "point_number" elements (with x and y components)
    # we assume that the array is regular with the same number of velocities during each iteration
    velocities_vector = np.reshape(velocities, [iteration_number, point_number,2])

    # Normalize 
    velocities_vector[:,:,0] = velocities_vector[:,:,0] / np.linalg.norm(velocities_vector, axis=2)
    velocities_vector[:,:,1] = velocities_vector[:,:,1] / np.linalg.norm(velocities_vector, axis=2)
    # Get the angle of the velocity vector
    velocities_angles = np.arccos(velocities_vector[:,:,0])

    return velocities_angles

def plot_polar_histogram(point_number, iteration_number, velocities_angles, iteration_to_plot):
    """
    Plots the distribution of velocities angles as a histogram with polar projection.
    
    Parameters
    -----
    point_number : int
        Number of seed points in the observation zone
    iteration_number : int 
        Number of iteration in the simulation
    velocities_angles : numpy array
        Array containing the orientation angles of the cells velocity vectors
    iteration_to_plot : int
        The iteration that will be plotted

    Returns
    -----
    fig : Figure
    """

    radii =  10 * np.random.rand(point_number)
    width = (2*np.pi) / point_number
    fig = plt.gcf()    
    ax = plt.gca()
    try:
        plt.hist(velocities_angles[iteration_to_plot,:], bins='auto')
        plt.title("Distribution of velocity angles {}".format(iteration_to_plot), y=1.15)
        plt.tight_layout()
    except IndexError:
        print("The iteration required {}/{} is out of range, plotting iteration 0".format(iterations_to_plot, iteration_number))
        plt.hist(velocities_angles[0,:], bins='auto')
        plt.title("Distribution of velocity angles 0 ")
#    bars = ax.bar(velocities_angles[0,:], radii, width=width)
    finally:
        return fig


def compute_nematic_scalar_quadrupole(point_number, iteration_number, fitted_ellipses_angles):
    """
    Compute the quadrupole (average of the second Legendre polynomial) as an orientational order parameter (as defined in De Gennes and Prost, The Physics of Liquid Crystals)

    Parameters
    -----


    Returns
    -----
    scalar_quadrupole: float
        Orientational order parameter describing the alignement of liquid crystals

    """


    #### How to define the local nematic director ?
    scalar_quadrupole = None

    # Local nematic director : spatial average of long ellipses axes
    local_nematic_director = np.mean(fitted_ellipses_angles, axis=1) 

    assert len(local_nematic_director) == iteration_number , "Mismatch fitted ellipses angles vs iteration number"

    # Compute the angle between the individual long axes and the local director
    # Use a transposed array (.T) for proper broadcast 
    theta = fitted_ellipses_angles.T - local_nematic_director
    # Recover the original shape
    theta = theta.T

    scalar_quadrupole = 0.5 * np.mean( 3 * np.cos(theta)**2 - 1, axis=1)

    return scalar_quadrupole

def plot_scalar_quadrupole(scalar_quadrupole, delta_t):
    """
    # TODO
    """
    
    nb_iter = len(scalar_quadrupole)

    # Prepare a time vector to plot the scalar quadrupole at every time step
    time_vector = np.arange(0, nb_iter*delta_t, delta_t)

    fig = plt.figure("Evolution of scalar quadrupole over time")
    plt.plot(time_vector,  scalar_quadrupole, 'c+')
    plt.title("Evolution of scalar quadrupole over time")
    plt.xlabel("Time")
    plt.ylabel("S")

    return fig



def get_logdirectories_list(base_path='../../'):
    """
    Quick hack to get the path to the logdirectory of simulation batch.
    """
    pth = Path(base_path)
    dir_list = [str(dir_l)+'/logdirectory' for dir_l in pth.iterdir() if pth.is_dir()]
#    print(len(dir_list))
#    for dl in dir_list:
#        print(dl)
    return dir_list

def plot_orientational_orders_all(dir_list, velocities_log_filename, logfilename, delta_t):
    """
    Plot the orientational order evolution for every v_0 value in the simulation batch.

    Parameters
    -----
    dir_list : list
        List of path strings to the logdirectories
    velocities_log_filename : str
        Name of the velocity logfile
    logfilename : str
        Name of the simulation logfile
    delta_t : float
        Timestep between two iterations

    Returns
    -----
    fig : Figure
    """
    # Prepare figure 
    fig = plt.figure("Evolution of orientational order over time")
    plt.title("Evolution of orientational order over time")
    plt.xlabel("Time")
    plt.ylabel(r"$\varphi$")

    # If we really want different symbols for every plot, go for this but honestly this look terrible
#    custom_cycler = (cycler(color=list('rgby')) + cycler(linestyle=['-', '--', ':', '-.']))
#    ax = plt.gca()
#    ax.set_prop_cycle(custom_cycler)

    for pth in dir_list:
        temp_logfilename = os.path.join(pth, logfilename)
        temp_velocities_log_filename = os.path.join(pth,velocities_log_filename)

        orientational_orders = orientational_orders_analysis(temp_logfilename, temp_velocities_log_filename)

        nb_iter = len(orientational_orders)
        # Prepare a time vector to plot the orientational order at every time step
        time_vector = np.arange(0, nb_iter*delta_t, delta_t)

        # Get v_0
        try:
            v_0_value = re.search('v_0_[0-9]+.[0-9]+', pth).group(0)
        except AttributeError:
            # The value couldn't be found
            v_0_value = 'default value'

        plt.plot(time_vector,  orientational_orders, label=v_0_value)
        
    plt.legend()
    return fig

def plot_scalar_quadrupole_all(dir_list, fitted_ellipses_angles_coord_logfilename, logfilename, delta_t):
    """
    Plot the orientational order evolution for every v_0 value in the simulation batch.

    Parameters
    -----
    dir_list : list
        List of path strings to the logdirectories
    fitted_ellipses_angles_coord_logfilename : str
        Name of the logfile with the coordinates [cos(psi), sin(psi)] of the fitted ellipses angles
    logfilename : str
        Name of the simulation logfile
    delta_t : float
        Timestep between two iterations

    Returns
    -----
    fig : Figure
    """
    # Prepare figure 
    fig = plt.figure("Evolution of scalar quadrupole over time")
    plt.title("Evolution of scalar quadrupole over time")
    plt.xlabel("Time")
    plt.ylabel(r"$S$")

    # If we really want different symbols for every plot, go for this but honestly this look terrible
#    custom_cycler = (cycler(color=list('rgby')) + cycler(linestyle=['-', '--', ':', '-.']))
#    ax = plt.gca()
#    ax.set_prop_cycle(custom_cycler)

    for pth in dir_list:
        temp_logfilename = os.path.join(pth, logfilename)
        temp_fitted_ellipses_angles_coord_logfilename = os.path.join(pth,fitted_ellipses_angles_coord_logfilename)

        fitted_ellipses_angles_coord = rspl.read_seed_points(temp_fitted_ellipses_angles_coord_logfilename)
        

        ## Extract the angles values by taking the arccos
        fitted_ellipses_angles = np.arccos(fitted_ellipses_angles_coord[:,:,0])

        scalar_quadrupole = compute_nematic_scalar_quadrupole(point_number, iteration_number, fitted_ellipses_angles)

        nb_iter = len(scalar_quadrupole)
        # Prepare a time vector to plot the orientational order at every time step
        time_vector = np.arange(0, nb_iter*delta_t, delta_t)

        # Get v_0
        try:
            v_0_value = re.search('v_0_[0-9]+.[0-9]+', pth).group(0)
        except AttributeError:
            # The value couldn't be found
            v_0_value = 'default value'

        plt.plot(time_vector, scalar_quadrupole, label=v_0_value)
        
    plt.legend()
    return fig






if __name__ == '__main__':

    path = "./"
    velocities_log_filename = "velocities_log.csv"
    logfilename = "logfile.txt"
    logfilename = os.path.join(path, logfilename)
    velocities_log_filename = os.path.join(path,velocities_log_filename)
   
    delta_t = get_parameter_regexp(logfilename, "delta_t")

    ############################
    ### MSD computations

    MSD_vector = MSD_analysis(logfilename, velocities_log_filename)
    figure_MSD = plot_MSD(MSD_vector, delta_t)
    save_fig_as(path, "MSD_plot.png")
    plt.show(figure_MSD)

    ### Compute the orientational order

    # Load the velocities from the log 
    velocities = rvl.read_velocities(velocities_log_filename)
    point_number = int(get_parameter_regexp(logfilename, "point_number"))
    delta_t = float(get_parameter_regexp(logfilename, "delta_t"))
    iteration_number = int(get_parameter_regexp(logfilename, "iteration_number"))
    


    orientational_orders = orientational_orders_analysis(logfilename, velocities_log_filename)
    figure_orientational_orders = plot_orientational_orders(orientational_orders, delta_t)
    save_fig_as(path, "orientational_orders_plot.png")
    plt.show(figure_orientational_orders)


    ### Polar histogram for angles
    velocities_angles = compute_velocities_angles(point_number, iteration_number, velocities)
    iterations_to_plot = [1, 500, 1000, 1900]
    subplots_rows = 2
    subplots_cols = 2

    pol_fig, pol_ax = plt.subplots(nrows=subplots_rows, ncols=subplots_cols, subplot_kw=dict(projection='polar'))
    iteration_cpt = 0 
    for i in range(subplots_rows):
        for j in range(subplots_cols):
            plt.sca(pol_ax[i,j])

            plot_polar_histogram(point_number, iteration_number, velocities_angles, iterations_to_plot[iteration_cpt])
            iteration_cpt += 1
    
    # Get the element of the list 
    items_list_title = ""
    for item in iterations_to_plot:
        items_list_title += "_{}".format(str(item))

    save_fig_as(path, "polar_plot_histogram_velocities_angles{}.png".format(items_list_title))
    plt.show("Polar_plot_subplots")

    #### Plot all the orientational order plots on the same plot
    dir_list = get_logdirectories_list()
    plot_orientational_orders_all(dir_list, velocities_log_filename, logfilename, delta_t)
    save_fig_as(path, "all_orientational_orders.png")
    plt.show()


    ### Scalar quadrupole
    ## Extract the fitted ellipse angles coordinates [cos(ang), sin(ang)] using the same methods as to read the seed points (same logging style)
    fitted_ellipses_angles_coord_logfilename = "ellipses_major_axes_angles.csv"
    fitted_ellipses_angles_coord_logfilename = os.path.join(path, fitted_ellipses_angles_coord_logfilename)
    fitted_ellipses_angles_coord = rspl.read_seed_points(fitted_ellipses_angles_coord_logfilename)
    

    ## Extract the angles values by taking the arccos
    fitted_ellipses_angles = np.arccos(fitted_ellipses_angles_coord[:,:,0])

    scalar_quadrupole = compute_nematic_scalar_quadrupole(point_number, iteration_number, fitted_ellipses_angles)
    plot_scalar_quadrupole(scalar_quadrupole, delta_t)
    save_fig_as(path, "scalar_quadrupole.png")
    plt.show()

    #### Plot all the scalar quadrupole plots on the same plot
    plot_scalar_quadrupole_all(dir_list, fitted_ellipses_angles_coord_logfilename, logfilename, delta_t)
    save_fig_as(path, "all_scalar_quadrupole.png")
    plt.show()

