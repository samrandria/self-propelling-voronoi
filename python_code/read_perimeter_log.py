"""
Read area log
"""

from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
import csv
import os

def read_perimeter_log_modified_x(zone_to_print):
    # Get the path 
    path = './'
    filename = 'perimeters_log_modified_x.csv'
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
    plt.title("Perimeter after modification in x variation, Zone {}".format(zone_to_print))
    plt.xlabel("Line number")
    plt.ylabel("Perimeter (a.u)")
    plt.show()

def read_perimeter_log_modified_y(zone_to_print):
    # Get the path 
    path = './simulation_test_trash/'
    filename = 'perimeters_log_modified_y.csv'
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
    plt.title("Perimeter after modification in y variation, Zone {}".format(zone_to_print))
    plt.xlabel("Line number")
    plt.ylabel("Perimeter (a.u)")
    plt.show()

def read_perimeter_log(zone_to_print):
    # Get the path 
    path = './simulation_test_trash/'
    filename = 'perimeters_log.csv'
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
    plt.title("Perimeter of zone {} for each step of integration".format(zone_to_print))
    plt.xlabel("Integration step")
    plt.ylabel("Perimeter (a.u)")
    plt.show()



if __name__ == '__main__':
    read_perimeter_log_modified_x(1)
    read_perimeter_log_modified_y(1)
    read_perimeter_log(0)
