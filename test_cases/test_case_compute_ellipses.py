"""
Compute the parameters of the ellipses associated to a Voronoi tesselation
Deprecated
"""

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from voronoi_tesselation import get_polygons_properties_2d, voronoi_finite_polygons_2d, voronoi_plot_2d
from voronoi_tesselation import generate_periodic_voronoi_cells_2 
from voronoi_tesselation import plot_polygons
import numpy as np

# XXX : to add
from matplotlib.patches import Ellipse

def compute_inertia_tensors(regions, vertices):
    """
    Compute the second moment of area for every cell (simple polygons)
    
    Parameters:
    -----
    regions : numpy array of arrays
        Array of arrays pointing to the "vertices" array, determining a cell
    vertices : numpy array
        Coordinates of cell vertices
    
    Returns:
    -----
    inertia_tensors : numpy array 
        Array of 2X2 tensors for every cell
    """
    inertia_tensors = np.zeros([len(regions), 2, 2])
    for index, region in enumerate(regions):
        for i in range(len(vertices[region])-1):
            x_i = vertices[region, 0][i]
            x_i_p1 = vertices[region, 0][i+1]           
            y_i = vertices[region, 1][i]
            y_i_p1 = vertices[region, 1][i+1]           
            # Ixx
            inertia_tensors[index, 0, 0] = inertia_tensors[index, 0, 0] + ( 
                (y_i*y_i + y_i*y_i_p1 + y_i_p1 * y_i_p1) * (x_i*y_i_p1 - x_i_p1*y_i) )   
            # Iyy
            inertia_tensors[index, 1, 1] = inertia_tensors[index, 1, 1] + ( 
                (x_i*x_i + x_i*x_i_p1 + x_i_p1 * x_i_p1) * (x_i*y_i_p1 - x_i_p1*y_i) )              
            # Ixy 
            inertia_tensors[index, 0, 1] = inertia_tensors[index, 0, 1] + (
                (x_i*y_i_p1 + 2*x_i*y_i + 2*x_i_p1*y_i_p1 + x_i_p1*y_i) * (x_i*y_i_p1 - x_i_p1*y_i) )
     
    # Do the divisions once at the end
    inertia_tensors[:, 0, 0] = inertia_tensors[:, 0, 0] * (1.0/12.0)
    inertia_tensors[:, 1, 1] = inertia_tensors[:, 1, 1] * (1.0/12.0)   
    inertia_tensors[:, 0, 1] = inertia_tensors[:, 0, 1] * (1.0/24.0)   
    # Copy the values from Ixy in Iyx in one go (fast with numpy)
    inertia_tensors[:, 1, 0] = inertia_tensors[:, 0, 1]
    # If the tensor is always Ixy = Iyx and if huge memory/computation time constraints, we can forgo the copy. 
    
    # Ensure that the Ixx and Iyy are positive
    inertia_tensors[:, 0, 0] = np.abs(inertia_tensors[:, 0, 0])
    inertia_tensors[:, 1, 1] = np.abs(inertia_tensors[:, 1, 1])   
    return inertia_tensors 


def compute_semi_major_minor_axes(inertia_tensors):
    """
    Computes the semi major and minor axes of every ellipse associated to the cells, given the cells' inertia tensors.
    
    Parameter:
    -----
    inertia_tensors : numpy array
        Array of 2X2 tensors for every cell
    
    Returns:
    -----
    semi_major_axes : numpy array
        Semi minor axes of associated ellipses. Usually denoted by an 'a' in formulao
    semi_minor_axes : numpy array
        Semi minor axes of associated ellipses. Usually denoted by a 'b' in formulae
    """
   
    tensors_number = len(inertia_tensors)
    # Initialisation
    semi_major_axes = np.zeros(tensors_number)   
    semi_minor_axes = np.zeros(tensors_number)
    
    for index, tensor in enumerate(inertia_tensors):
        Ixx = tensor[0,0]
        Iyy = tensor[1,1]
#        print("Ixx = {} , Iyy = {}".format(Ixx, Iyy))
        semi_major_axes[index] = (Ixx**-1 * Iyy**3 * (np.pi/4)**-2)**(1/8)
        semi_minor_axes[index] = (Iyy**-1 * Ixx**3 * (np.pi/4)**-2)**(1/8)
   
    return semi_major_axes, semi_minor_axes


def compute_angles_major_axis_positive_x_axis(inertia_tensors):
    """
    Compute the angles between the major axis of the ellipse and the positive x-axis of the plane
    
    Parameters :
    -----
    inertia_tensors : numpy array
        Array of 2X2 tensors for every cell
    
    Returns :
    -----
    angles_alpha : numpy array
        Array of angles between the major axis and the positive x-axis
    """
    tensors_numbers = len(inertia_tensors)
    angles_alpha = np.zeros([tensors_numbers])
    
    for index, tensor in enumerate(inertia_tensors):
        angles_alpha[index] = np.arctan(-(tensor[0,0] - tensor[1,1])/(-2* tensor[0,1]))
        
    angles_alpha = 0.5 * angles_alpha
    
    return angles_alpha


def compute_ellipse_parameters_full(regions, vertices):
    """
    Compute the center of mass coordinates, the semi-major and semi-minor axes of the ellipse
    and the angle between the major axis and te positive x-axis.
    
    Parameters:
    -----
    regions : numpy array of arrays
        Array of arrays pointing to the "vertices" array, determining a cell
    vertices : numpy array
        Coordinates of cell vertices
    
    Returns:
    -----
    semi_major_axis : float
        Semi major axis of the fitted ellipse
    semi_minor_axis : float
        Semi minor axis of the fitted ellipse   
    angle_alpha : float
        Angle between the major axis of the fitted ellipse and the positive x-axis
    inertia_tensors : numpy array
        Array of inertia tensor for each cell (polygon) 
    """
    inertia_tensors = compute_inertia_tensors(regions, vertices)
    semi_major_axes, semi_minor_axes = compute_semi_major_minor_axes(inertia_tensors)
    angles_alpha = compute_angles_major_axis_positive_x_axis(inertia_tensors)
    
    return semi_major_axes, semi_minor_axes, angles_alpha, inertia_tensors 

def generate_periodic_boundaries(val, box):
    """
    Generates periodic boundaries by translating values to fit inside the observation box.
    
    Inspired by Hiraiwa Tetsuya. Modified because of performance issues. 
    Parameters:
        val: double, coordinate
        box: size of one side of the observation box
    -----
    Returns:
        translated_value: double, coordinate after translation
    
    Note: in CCM.C (program by T.H), this function is called at initialization and [after Runge-Kutta but before polarity time evolution]
    """

    ## Just like in a videogame, when you cross a side of the screen, you appear back on the other side

    # while(val > box/2.0):
    #     # val = val - box
    #     # The above option is correct but time consuming
    #     val = val % box
    # while( val <= -box/2.0):
    #     # val = val + box
    #     # The above option is correct but time consuming
    #     val = val % box

    while(val < -box/2.0 or val > box/2.0):
        if(val > box/2.0 and val < box):
            val = (box % val) * (-1.0)
        elif(val < -box/2.0 and val > -box):
            val = (-box % val) * (-1.0)
        elif(val >= box):
            val = val % box
        elif(val <= box):
            val = val % -box
    return val

def initialization(point_number, boxsize):
    """
    Create an array of random seed points (Voronoi center), and applies periodic boundary conditions.
    TO DO: initialize polarity, angles, momentum, a flag indicating at which stage we are currently in.
    
    Parameters
    -----
    point_number : int
        Number of seed point to create
    boxsize : float 
        Length of the axis for observation box (assumed square for now)

    Returns
    -----
    generated_points : ndarray of ndarray
        Array of random seed points (of coordinates x,y)
    generated_angles : ndarray
        Array of randomly drawn angles in radians
    generated_polarities : ndarray of double
        Array of polarity vectors (cos theta_i, sin theta_i), initialized with random initial values 
    (in the future maybe): flag
    """
    # XXX : DEBUG
    generated_points = np.random.rand(point_number, 2) * boxsize - (boxsize * 0.5 * 0.5)
        #### I do not see the point of doing this in the initialization part ####
    for coord in generated_points:
        coord[0] = generate_periodic_boundaries(coord[0], boxsize) # x coordinate
        coord[1] = generate_periodic_boundaries(coord[1], boxsize) # y coordinate
        # print(coord[0], coord[1])
    return generated_points

def plot_ellipses(points, semi_major_axes, semi_minor_axes, angles_alpha):
    """
    Plot the ellipses associated to the Voronoi cells

    Parameters :
    -----
    points : #TODO
        Cell centers
    semi_major_axes :
        #TODO
    semi_minor_axes :
        #TODO
    angles_alpha : 
        #TODO

    Returns :
    -----
    #TODO
    """
#    plt.figure()
    ax = plt.gca()
    number_to_plot = 40

    # Convert angles alpha to degrees for plotting purposes
    angles_alpha_deg = np.degrees(angles_alpha)
    point = points[number_to_plot] 
    semi_major_axis = semi_major_axes[number_to_plot]
    semi_minor_axis = semi_minor_axes[number_to_plot]
    angle_alpha = -angles_alpha_deg[number_to_plot]
    print("ANGLE ALPHA RAD= {}".format(angle_alpha))
    print("ANGLE ALPHA DEG= {}".format(np.degrees(angle_alpha)))
#    angle_alpha = 180 - 130
    ellipse = Ellipse(xy=(point[0],point[1]), width=semi_minor_axis, height=semi_major_axis, angle=180+angle_alpha)
    ax.add_patch(ellipse)
    plt.plot(point[0],point[1],'k+')
        
#    for point, semi_major_axis, semi_minor_axis, angle_alpha in zip(points, semi_major_axes, semi_major_axes, angles_alpha):
#        plt.plot(point[0],point[1],'g+') 
#        ellipse = Ellipse(xy=(point[0],point[1]), width=semi_minor_axis, height=semi_major_axis, angle=angle_alpha)
#        ax.add_patch(ellipse)
#    for point in points:
#        print(point)
    

if __name__=='__main__':
    # Generate a random tesselation
    boxsize = 10
    point_number = 100
    np.random.seed(55)
#    points = np.random.rand(point_number,2) * boxsize - (boxsize * 0.5 * 0.5)
#    points = generate_periodic_voronoi_cells_2(points, boxsize)
    points = initialization(point_number, boxsize)
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor, boxsize)

    # Restricted to the observation box
    semi_major_axes, semi_minor_axes, angles_alpha, inertia_tensors = compute_ellipse_parameters_full(regions[:point_number], vertices)
    plot_polygons(vor, regions, vertices)
#    for index_pair in vor.ridge_vertices:
#        #Critical for a nice graph
#        if index_pair[0] >= 0 and index_pair[1] >= 0:
#            v0 = vor.vertices[index_pair[0]]
#            v1 = vor.vertices[index_pair[1]]
#                # Draw a line from v0 to v1.
#            plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)  

#    print("Semi-major axes = \n{}".format(semi_major_axes))  
    # Figure parameters
    plt.xlim(-boxsize/2.0, boxsize/2.0)
    plt.ylim(-boxsize/2.0, boxsize/2.0)
    plt.title("Deterministic test model. {0} regions".format(len(regions)))
    plt.xlabel("x")
    plt.ylabel("y")

    print("Points vus par plot_ellipses :")
    plot_ellipses(points, semi_major_axes, semi_minor_axes, angles_alpha)
#    print("-"*20) 
#    print("Vrais points length = {}".format(len(points[:point_number])))
    for index, point in enumerate(points[0:100]):
        plt.text(point[0],point[1],str(index) ,color="red", fontsize=12)
    plt.show() 

#    plt.xlim(-boxsize/2.0, boxsize/2.0)
#    plt.ylim(-boxsize/2.0, boxsize/2.0)
#    plt.title("Deterministic test model. {0} regions".format(len(regions)))
#    plt.xlabel("x")
#    plt.ylabel("y")
#    plt.show() 

