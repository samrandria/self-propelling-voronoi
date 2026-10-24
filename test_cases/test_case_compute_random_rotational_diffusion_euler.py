"""
Test case compute_random_rotational_diffusion_euler
"""

import numpy as np 
from flocking_confluent_marchetti import compute_random_rotational_diffusion_euler, gaussian_white_noise_process
from flocking_confluent_marchetti import compute_every_random_rotational_diffusion_euler


if __name__ == '__main__':

    angles = np.array([0, np.pi, 2*np.pi, 3*np.pi, -np.pi, -2*np.pi, 2*np.pi+1, -np.pi/2, 10*np.pi, -50*np.pi])

    # Test with random_process = 0, delta_t = 0 
    random_process = 0
    delta_t = 0
    print("Test 1, random_process = {0}, delta_t = {1}".format(random_process,delta_t))
    # Should only remap the angles to the [0, 2pi[
    for angle in angles:
        angle_next = compute_random_rotational_diffusion_euler(angle,random_process,delta_t)
        print(angle_next)

    # NOTES : for -50*np.pi, although the angle should be remapped to exactly 0, floating point calculation gives
    # a result of ~9.23E-14 
    # In practice, delta_t and random_process should be small enough ?
    print('-'*10)

    random_process = 0
    delta_t = 1
    print("Test 2, random_process = {0}, delta_t = {1}".format(random_process,delta_t))
    # Test with random_process = 0, delta_t = 1 
    # Should only remap the angles to the [0, 2pi[
    for angle in angles:
        angle_next = compute_random_rotational_diffusion_euler(angle,random_process,delta_t)
        print(angle_next)
    print('-'*10)

    random_process = 1
    delta_t = 0
    print("Test 3, random_process = {0}, delta_t = {1}".format(random_process,delta_t))
    # Test with random_process = 1, delta_t = 0 
    # Should only remap the angles to the [0, 2pi[
    for angle in angles:
        angle_next = compute_random_rotational_diffusion_euler(angle,random_process,delta_t)
        print(angle_next)
    print('-'*10)

    random_process = 1
    delta_t = 1
    print("Test 4, random_process = {0}, delta_t = {1}".format(random_process,delta_t))
    # Test with random_process = 1, delta_t = 1 
    # Should remap the angles to the [0, 2pi[ and add 1
    for angle in angles:
        angle_next = compute_random_rotational_diffusion_euler(angle,random_process,delta_t)
        print(angle_next)
    print('-'*10)

    random_process = 1
    delta_t = 2
    print("Test 5, random_process = {0}, delta_t = {1}".format(random_process,delta_t))
    # Test with random_process = 1, delta_t = 1 
    # Should remap the angles to the [0, 2pi[ and add 1
    for angle in angles:
        angle_next = compute_random_rotational_diffusion_euler(angle,random_process,delta_t)
        print(angle_next)
    print('-'*10)


    ######
    print('*'*50)
    
    #### Test of the generalized function
    print("Test of the generalized function without angular noise")
    angles_next = compute_every_random_rotational_diffusion_euler(angles,0,1)
    print(angles_next)

    print('-'*50)
    print("Test of the generalized function with angular noise")
    angles_next = compute_every_random_rotational_diffusion_euler(angles,10,1)
    print(angles_next)