'''
Module containing the exact formula to compute energy derivatives, instead of the Euler approximation

'''

import numpy as np

def compute_vertex_coefficients(r_i, r_j, r_k):
    """
    Compute the coefficient described in [Bi et al, 2016] equation A2.

    Parameters
    -----
    r_i : numpy array
        Coordinates of a Voronoi center that define a vertex with two other centers
    r_j : numpy array
        Coordinates of a Voronoi center that define a vertex with two other centers
    r_k : numpy array
        Coordinates of a Voronoi center that define a vertex with two other centers

    Returns
    -----
    coeff_alpha : double
    coeff_beta : double
    coeff_gamma : double
    coeff_D : double
    """

    coeff_D = 2 * np.linalg.norm(np.cross( (r_i - r_j), (r_j - r_k) ) )**2

    coeff_alpha = np.linalg.norm(r_j - r_k)**2 * ( np.dot((r_i - r_j), (r_i - r_k))  )/coeff_D
    coeff_beta  = np.linalg.norm(r_i - r_k)**2 * ( np.dot((r_j - r_i), (r_j - r_k))  )/coeff_D
    coeff_gamma = np.linalg.norm(r_i - r_j)**2 * ( np.dot((r_k - r_i), (r_k - r_j))  )/coeff_D

    return coeff_alpha, coeff_beta, coeff_gamma, coeff_D


def compute_supplementary_vertex_coefficients(r_i, r_j, r_k):
    """
    Additional convenience coefficients to derive the exact energy 

    Parameters
    -----
    r_i : numpy array
        Coordinates of a Voronoi center that define a vertex with two other centers
    r_j : numpy array
        Coordinates of a Voronoi center that define a vertex with two other centers
    r_k : numpy array
        Coordinates of a Voronoi center that define a vertex with two other centers

    Returns 
    -----
    coeff_lambda : double
    coeff_kappa : double
    coeff_lambda_prime : double
    coeff_kappa_prime : double
    coeff_alpha_star : double
    coeff_beta_star : double
    coeff_gamma_star : double
    coeff_D_star : double
    """

    # For convenience
    r_ix = r_i[0]
    r_iy = r_i[1]
    r_jx = r_j[0]
    r_jy = r_j[1]
    r_kx = r_k[0]
    r_ky = r_k[1]

    coeff_lambda = r_jy - r_ky
    coeff_lambda_prime = r_jx - r_kx

    coeff_kappa = (r_iy - r_jy) * (r_jx - r_kx)
    coeff_kappa_prime = (r_ix - r_jx) * (r_jy - r_ky)

    coeff_alpha_star = np.dot( (r_i - r_j), (r_i - r_k) ) 
    coeff_beta_star  = np.dot( (r_j - r_k), (r_j - r_i) ) 
    coeff_gamma_star = np.dot( (r_k - r_i), (r_k - r_j) ) 

    return coeff_lambda, coeff_kappa, coeff_lambda_prime, coeff_kappa_prime, coeff_alpha_star, coeff_beta_star, coeff_gamma_star



def compute_coefficient_derivatives(r_i, r_j, r_k):
    """
    Additional convenience coefficients to derive the exact energy 

    Parameters
    -----
    r_i : numpy array
        Coordinates of a Voronoi center that define a vertex with two other centers
    r_j : numpy array
        Coordinates of a Voronoi center that define a vertex with two other centers
    r_k : numpy array
        Coordinates of a Voronoi center that define a vertex with two other centers

    Returns 
    -----
    dalpha_drix : double
    dalpha_driy : double
    dbeta_drix : double
    dbeta_driy : double
    dgamma_drix : double
    dgamma_driy : double
    """

    # For convenience
    r_ix = r_i[0]
    r_iy = r_i[1]
    r_jx = r_j[0]
    r_jy = r_j[1]
    r_kx = r_k[0]
    r_ky = r_k[1]

    # Load the coefficients
    coeff_alpha, coeff_beta, coeff_gamma, coeff_D = compute_vertex_coefficients(r_i, r_j, r_k)
    coeff_lambda, coeff_kappa, coeff_lambda_prime, coeff_kappa_prime, coeff_alpha_star, coeff_beta_star, coeff_gamma_star = compute_supplementary_vertex_coefficients(r_i, r_j, r_k)

    # Compute the derivative
    dalpha_drix = (np.linalg.norm(r_j - r_k)**2 / coeff_D) * ( 2*r_ix - r_kx - r_jx - 4*coeff_alpha_star*coeff_lambda * (coeff_lambda * (r_ix - r_jx) - coeff_kappa) )
    dalpha_driy = (np.linalg.norm(r_j - r_k)**2 / coeff_D) * ( 2*r_iy - r_ky - r_jy - 4*coeff_alpha_star*coeff_lambda_prime * (coeff_lambda_prime * (r_iy - r_jy) - coeff_kappa_prime) )

    dbeta_drix =  (1.0/coeff_D) * ( 2*(r_ix - r_kx) * coeff_beta_star + np.linalg.norm(r_i - r_k)**2 * (r_kx - r_jx) - 4*coeff_beta*coeff_lambda * (coeff_lambda * (r_ix - r_jx) - coeff_kappa) )
    dbeta_driy =  (1.0/coeff_D) * ( 2*(r_iy - r_ky) * coeff_beta_star + np.linalg.norm(r_i - r_k)**2 * (r_ky - r_jy) - 4*coeff_beta*coeff_lambda_prime * (coeff_lambda_prime * (r_iy - r_jy) - coeff_kappa_prime) )

    dgamma_drix = (1.0/coeff_D) * ( 2*(r_ix - r_jx) * coeff_gamma_star + np.linalg.norm(r_i - r_j)**2 * (r_jx - r_kx) - 4*coeff_gamma*coeff_lambda * (coeff_lambda * (r_ix - r_jx) - coeff_kappa) )  
    dgamma_driy = (1.0/coeff_D) * ( 2*(r_iy - r_jy) * coeff_gamma_star + np.linalg.norm(r_i - r_j)**2 * (r_jy - r_ky) - 4*coeff_gamma*coeff_lambda_prime * (coeff_lambda_prime * (r_iy - r_jy) - coeff_kappa_prime) )  


    return dalpha_drix, dalpha_driy, dbeta_drix, dbeta_driy, dgamma_drix, dgamma_driy
