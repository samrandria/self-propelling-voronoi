#!/usr/bin/env python3

"""
Parameters generator.
Generates a text file with the commands to insert in the job manager.
Call with python3 parameters.generator.py parameter_to_explore min_parameter max_parameter step_size
"""

import numpy as np
import os
import sys

def generate_parameters(filename, filename_prefix, parameter_to_explore, min_parameter, max_parameter, step_size):
    """
    Creates a textfile containing commands for the job manager with a parameter to explore

    Parameters
    -----
    filename : string
        Name of the file to register the commands and parameters
    filename_prefix : string
        e.g. : "/opt/anaconda3/bin/python3 flocking_confluent_marchetti.py"
    parameter_to_explore : string
        The parameter to vary, e.g. "point_number", "boxsize", "K_P"...
    min_parameter : int, float
        Start value for the exploration
    max_parameter : int, float
        End value for the exploration
    step_size : int, float
        Increment size for the exploration
    
    Returns
    -----
    Nothing
    """
    # Technically already passed as a string but let's avoid surprises
    filename = str(filename)
    param = str(parameter_to_explore)

    min_param = min_parameter
    max_param = max_parameter
    step_size_param = step_size

    # Some parameters must be cast into integer to avoid simulation crashes
    # FIXME : Actually, we don't care, proper casting is done in the main script, we write strings anyway
    if(param == 'point_number' or param == 'iteration_number'):
        min_param = int(min_param)
        max_param = int(max_param)
        step_size_param = int(step_size_param)
    else:
        min_param = float(min_param)
        max_param = float(max_param)
        step_size_param = float(step_size_param)

    # Load some base parameters for the dictionary
    point_number = 100 
    boxsize = 10
    iteration_number = 50
    delta_t = 0.001
    delta_r = boxsize/1000.0
    # Target parameters
    P_0 = 5.0* np.sqrt(np.pi)
    A_0 = 1.0 
    # Cell parameters
    K_A = 1.0
    K_P = 1.0
    v_0 = 0.0
    mu = 1.0
    angular_noise = 0.0 


    # Create a dictionary to fetch the value associated to parameter_to_explore
    # XXX : In python 3.6 Dictionary REMAINS ORDERED, this is critical for proper writing herefafter
    parameters_dictionary = {'point_number' : point_number, 'boxsize' : boxsize, 'iteration_number' : iteration_number, 
                             'delta_t' : delta_t, 'delta_r' : delta_r, 'P_0' : P_0,
                             'A_0' : A_0, 'K_A' : K_A, 'K_P' : K_P,
                             'v_0' : v_0, 'mu' : mu, 'angular_noise' : angular_noise}
    

    # Check if the parameter_to_explore is a key of the dictionary
    if(param not in parameters_dictionary):
        raise KeyError("Parameter does not exist")
    else: 
        pass

    # Test the parameter to explore and fetch the value to modify
    # Launch the loop to produce the values in the script     

    parameter_string = ""
    try:
        for value_to_modify in np.arange(min_param, max_param, step_size_param):
            # Update the value in the dictionary 
            parameters_dictionary[param] = value_to_modify 

            # Create the parameter string by concatenating all the values 
            for val in parameters_dictionary.values():
                parameter_string = parameter_string + " " + str(val)

            # Append the string to the generated command file
            write_parameters_file(filename, filename_prefix, parameter_string)
            # Reset the string
            parameter_string = ""
        
    except KeyError:
        print("Erreur")
    
    

def write_parameters_file(filename, filename_prefix, parameter_string):
    """
    Concatenates a command prefix and the string of parameters to build a full command

    Parameters
    -----
    filename : string
        Name of the file to write/append to 
    filename_prefix : string
        e.g. flocking_confluent_marchetti.py
    parameter_string : string
        String of parameter to run the main script in the "Passed arguments mode"

    Returns
    -----
    Nothing
    """
    fname = str(filename)
    # Concate the command prefix and the parameter string
    string_to_write = str(filename_prefix) + " " + str(parameter_string)

    with open(fname, 'a') as txt:
        txt.write(string_to_write)
        txt.write("\n")




if __name__ == '__main__':

    # FIXME : Hardcoded path : pas terrible.
    path = "./"

    python_path = "/opt/anaconda3/bin/python3"
    # -O flag to disable asserts
    python_flags = "-O"
    main_script_name = "main_spv_script.py" 

    filename_prefix = ' '.join((python_path, python_flags, main_script_name))
#    filename_prefix = "/opt/anaconda3/bin/python3 main_spv_script.py"


    if not len(sys.argv) > 1:
        print("Not enough arguments")
    else:
        try:
            parameter_to_explore = sys.argv[1]
            min_parameter = sys.argv[2]
            max_parameter = sys.argv[3]
            step_size= sys.argv[4]

            filename = "parameter_{}_min_{}_max_{}_step_{}.txt".format(parameter_to_explore, min_parameter, max_parameter, step_size)
            filename = os.path.join(path, filename)

            generate_parameters(filename, filename_prefix, parameter_to_explore, min_parameter, max_parameter, step_size)

        except IndexError:
            print("Not enough arguments")

    
