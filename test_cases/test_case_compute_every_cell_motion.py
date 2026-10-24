"""
Test program for compute_every_cell_motion
"""

import numpy as np 
from flocking_confluent_marchetti import compute_every_cell_motion, compute_cell_motion

if __name__ == '__main__':

    seed_points = np.array([[1,2], [2,5], [3,3]])
    forces = np.array([[2,2],[1,1], [3,3]])
    polarities = np.array([[0,0], [1,1], [2,2]])

    # Generalized function
    seed_points_next = compute_every_cell_motion(seed_points, forces, polarities, 1, 2, 1, 5)

    # Repeated calls of individual function
    seed_point_next_0 = compute_cell_motion(seed_points[0], forces[0], polarities[0], 1, 2, 1, 5)
    seed_point_next_1 = compute_cell_motion(seed_points[1], forces[1], polarities[1], 1, 2, 1, 5)
    seed_point_next_2 = compute_cell_motion(seed_points[2], forces[2], polarities[2], 1, 2, 1, 5)

    print(seed_points)
    print('-'*50)
    print(seed_points_next)
    print('*'*50)
    print(seed_point_next_0)
    print(seed_point_next_1)
    print(seed_point_next_2)