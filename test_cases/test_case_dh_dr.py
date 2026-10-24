"""
Energy derivations implementations
Should be as agnostic of the other data structures as possible.
A vertex h is defined by a trio of Voronoi centers r_i, r_j, r_k such that

h = alpha * r_i + beta * r_j + gamma * r_k

This program implements the derivations for dh_dr
"""
import numpy as np

def coeff_lambda(r_j, r_k):
    """
    Convenience function for the computation of dalpha_drix
    Parameters:
    r_j, r_k : numpy arrays of shape [x,y]. 
        Coordinates of two vertices 
    ----- 
    Returns:
    lambd: double, a prefactor
    """
    r_jy = r_j[1]
    r_ky = r_k[1]
    lambd = (r_jy - r_ky) 
    return lambd

def coeff_lambda_prime(r_j, r_k):
    """
    Convenience function for the computation of dalpha_drix
    Parameters:
    r_j, r_k : numpy arrays of shape [x,y]. 
        Coordinates of two vertices 
    ----- 
    Returns:
    lambda_prime : double, a prefactor
    """
    r_jx = r_j[0]
    r_kx = r_k[0]
    lambda_prime = (r_jx - r_kx) 
    return lambda_prime

def coeff_kappa(r_i,r_j,r_k):
    """
    Convenience function for the computation of dalpha_drix
    Parameters:
    r_i, r_j, r_k : numpy arrays of shape [x,y]
        Coordinates of three vertices
    ----- 
    Returns:
    kappa : double, a prefactor
    """
    r_ix = r_i[0]
    r_jx = r_j[0]
    r_kx = r_k[0]

    r_iy = r_i[1]
    r_jy = r_j[1]
    r_ky = r_k[1]

    kappa = (r_iy - r_jy) * (r_jx - r_kx)
    return kappa

def coeff_kappa_prime(r_i,r_j,r_k):
    """
    Convenience function for the computation of dalpha_drix
    Parameters:
    r_i, r_j, r_k : numpy arrays of shape [x,y]
        Coordinates of three vertices
    ----- 
    Returns:
    kappa_prime : double, a prefactor
    """
    r_ix = r_i[0]
    r_jx = r_j[0]
    r_kx = r_k[0]

    r_iy = r_i[1]
    r_jy = r_j[1]
    r_ky = r_k[1]

    kappa_prime = (r_ix - r_jx) * (r_jy - r_ky)
    return kappa_prime


def coeff_D(r_i, r_j, r_k):
    """
    Convenience function for the computation of dalpha_drix
    Parameters:
    r_i, r_j, r_k : numpy arrays of shape [x,y]
        Coordinates of three vertices
    ----- 
    Returns:
    D : double, a prefactor
    """

    cross_product = np.cross((r_i - r_j), (r_j-r_k))
    # Can compute the norm of the nd array
    D = 2 * np.linalg.norm(cross_product)**2

    return D


def derive_dalpha_drix(r_i,r_j,r_k):
    """
    Derivation of coefficient alpha w.r.t. r_ix
    Parameters:
    r_i, r_j, r_k : numpy arrays of shape [x,y]
        Coordinates of three vertices
    ----- 
    Returns:
    dalpha_drix : double
        The derivative of coefficent alpha w.r.t the x coordinate of center r_i (r_ix) 
    """
    r_ix = r_i[0]
    r_jx = r_j[0]
    r_kx = r_k[0]

    r_iy = r_i[1]
    r_jy = r_j[1]
    r_ky = r_k[1]

    lambd = coeff_lambda(r_j, r_k)
    kappa = coeff_kappa(r_i,r_j,r_k)

    alpha_star = np.dot((r_i - r_j), (r_i - r_k))

    # Square norm of (r_j - r_k) divided by D
    prefactor = np.linalg.norm(r_j - r_k)**2/coeff_D(r_i, r_j, r_k)

    dalpha_drix = prefactor * (2*r_ix - r_kx - r_jx - 4*alpha_star*lambd * (lambd*(r_ix - r_jx) - kappa))
    
    return dalpha_drix


def derive_dbeta_drix(r_i,r_j,r_k):
    """
    Derivation of coefficient beta w.r.t. r_ix
    Parameters:
    r_i, r_j, r_k : numpy arrays of shape [x,y]
        Coordinates of three vertices
    ----- 
    Returns:
    dbeta_drix : double
        The derivative of coefficent beta w.r.t the x coordinate of center r_i (r_ix) 
    """
    r_ix = r_i[0]
    r_jx = r_j[0]
    r_kx = r_k[0]

    r_iy = r_i[1]
    r_jy = r_j[1]
    r_ky = r_k[1]

    lambd = coeff_lambda(r_j, r_k)
    kappa = coeff_kappa(r_i,r_j,r_k)

    # prefactor here is 1/D
    prefactor = 1.0/coeff_D(r_i, r_j, r_k)
    beta = np.linalg.norm(r_i - r_k)**2 * np.dot(r_j - r_i, r_j - r_k) * prefactor
    beta_star = np.dot((r_j - r_k), (r_j - r_i))

    # print("D : {0}".format(1/prefactor))
    # print("Beta : {0}".format(beta))
    # print("Beta_star : {0}".format(beta_star))

    dbeta_drix = prefactor * (2*(r_ix - r_kx)*beta_star + (np.linalg.norm(r_i - r_k)**2) * (r_kx - r_jx)
                                - 4*beta*lambd * (lambd* (r_ix - r_jx) - kappa)) 
    
    return dbeta_drix

def derive_dgamma_drix(r_i,r_j,r_k):
    """
    Derivation of coefficient gamma w.r.t. r_ix
    Same as derive_dbeta_drix but uses a circular permutation between j and k (except for D and its derivative
    terms)
    Parameters:
    r_i, r_j, r_k : numpy arrays of shape [x,y]
        Coordinates of three vertices
    ----- 
    Returns:
    dgamma_drix : double
        The derivative of coefficent gamma w.r.t the x coordinate of center r_i (r_ix) 
    """

    r_ix = r_i[0]
    r_jx = r_j[0]
    r_kx = r_k[0]

    r_iy = r_i[1]
    r_jy = r_j[1]
    r_ky = r_k[1]

    lambd = coeff_lambda(r_j, r_k)
    kappa = coeff_kappa(r_i,r_j,r_k)

    # print("Lambda : {0}".format(lambd))
    # print("Kappa : {0}".format(kappa))

    # prefactor here is 1/D
    prefactor = 1.0/coeff_D(r_i, r_j, r_k)
    gamma = np.linalg.norm(r_i - r_j)**2 * np.dot(r_k - r_i, r_k - r_j) * prefactor
    gamma_star = np.dot((r_k - r_i), (r_k - r_j))

    # print("D : {0}".format(1/prefactor))
    # print("Gamma : {0}".format(gamma))
    # print("Gamma_star : {0}".format(gamma_star))

    dgamma_drix = prefactor * (2*(r_ix - r_jx)*gamma_star + (np.linalg.norm(r_i - r_j)**2) * (r_jx - r_kx)
                                - 4*gamma*lambd * (lambd* (r_ix - r_jx) - kappa)) 
    
    return dgamma_drix






if __name__ == '__main__':

    r_i = np.array([3,5])
    r_j = np.array([2.5,1])
    r_k = np.array([4,6])
    
    print(np.linalg.norm(r_j - r_k)**2)
    print(np.linalg.norm(r_i - r_k)**2)
    print(np.linalg.norm(r_i - r_j)**2)

    print(coeff_lambda(r_j,r_k))
    print(coeff_lambda_prime(r_j,r_k))
    print(coeff_kappa(r_i,r_j,r_k))
    print(coeff_kappa_prime(r_i,r_j,r_k))
    print(coeff_D(r_i,r_j,r_k))


    alpha_star = np.dot((r_i - r_j), (r_i - r_k))
    print(alpha_star)
    print(derive_dalpha_drix(r_i,r_j,r_k))
    
    print("-"*50)

    print(derive_dbeta_drix(r_i,r_j,r_k))

    print("-"*50)

    print(derive_dgamma_drix(r_i,r_j,r_k))