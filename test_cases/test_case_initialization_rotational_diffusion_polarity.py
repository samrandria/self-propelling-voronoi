"""
Test case for rotational diffusion computation, initialization, polarity_vector generation
"""

import numpy as np 
from flocking_confluent_marchetti import initialization, compute_every_random_rotational_diffusion_euler
from flocking_confluent_marchetti import compute_random_rotational_diffusion_euler, generate_every_polarity_vector
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(42)
    N = 10
    boxsize = 10
    delta_t = 1
    points, angles, polarities = initialization(N, boxsize)

    plt.figure("Angles")
    plt.angle_spectrum(angles)

    # If there is zero noise, the angles should be the same prior and after the call
    D_r = 0
    angles_next = compute_every_random_rotational_diffusion_euler(angles, D_r, delta_t) 
    polarities_next = generate_every_polarity_vector(angles_next)
    print(angles_next == angles)
    print(polarities_next == polarities)

    print('-'*50)
    # If there is non-zero noise, the angles will shift 
    D_r = 0.5
    angles_next = compute_every_random_rotational_diffusion_euler(angles, D_r, delta_t) 
    polarities_next = generate_every_polarity_vector(angles_next)
    print(angles_next == angles)
    print(polarities_next == polarities)

    plt.show()