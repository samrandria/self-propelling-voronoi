"""
Test for generate_periodic_boundaries
"""

import numpy as np 
from flocking_confluent_marchetti import generate_periodic_boundaries

if __name__ == '__main__':

    seed_points = np.array([[1,1], [6,3], [4,6], [10,10], [10,11]]) 

    translated_seed_points = []

    for x,y in seed_points:
        x = generate_periodic_boundaries(x,10)
        y = generate_periodic_boundaries(y,10)
        translated_seed_points.append([x,y])

    translated_seed_points = np.array(translated_seed_points)

    print(seed_points)
    print('-'*50)
    print(translated_seed_points)
