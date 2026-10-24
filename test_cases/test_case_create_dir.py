"""
Creating a directory while avoiding race condition 
"""

import os
import errno

if __name__ == '__main__':
    point_number = 100 
    boxsize = 10
    iteration_number = 50
    delta_t = 0.001

    path = "./point_number_{}/boxsize_{}/iteration_number_{}/delta_t_{}".format(point_number,boxsize,iteration_number,delta_t)

    try:
        os.makedirs(path)
        print("Successfully completed")
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error")
            raise
        print("Error")
            