"""
Compute the parameters of the ellipses associated to a Voronoi tesselation
"""
# import pyximport ; pyximport.install()
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from voronoi_tesselation import get_polygons_properties_2d, voronoi_finite_polygons_2d, voronoi_plot_2d
from voronoi_tesselation import generate_periodic_voronoi_cells_2 
from voronoi_tesselation import plot_polygons
import numpy as np

import plot_ellipses_test as pet
# XXX : to add
from matplotlib.patches import Ellipse


def compute_moment_tensors(regions, vertices, points, areas):
    """
    Compute a moment tensor for every cell (simple polygons, 0th, 1st and 2nd moments)
    Based on Mulchrone and Choudhury, J. of Structural Geology 26 (2004)
    
    Parameters:
    -----
    regions : numpy array 
        Array of arrays pointing to the "vertices" array, determining a cell
    vertices : numpy array
        Coordinates of cell vertices
    points : numpy array
        Coordinates of Voronoi centers
    areas : numpy array
        Areas of Voronoi cells
    Returns:
    -----
    moment_tensors : numpy array 
        Array of 3X3 tensors for every cell
    """
    moment_tensors = np.zeros([len(regions), 3, 3])
    centroid_x = points[:,0] 
    centroid_y = points[:,1]

    # DEBUG
    # print("centroid_x : \n{}".format(centroid_x))
    # print("centroid_y : \n{}".format(centroid_y))
    tmp = 0
    # Mnemonic for easier manipulation
    for index, region in enumerate(regions):
        # Load the numpy arrays
        x_i = vertices[region, 0]
        y_i = vertices[region, 1]

        # use np.roll to have a "+1" array that wraps at the end
        x_i_p1 = np.roll(x_i, -1) 
        y_i_p1 = np.roll(y_i, -1) 

        # XXX DEBUG
#        if(index == 3):
#            print('/'*50)
#            print("TIME OUT")
#            print("x_i = \n {}".format(x_i))
#            print('-'*20)
#            print("y_i = \n {}".format(y_i))
#            print('-'*20)
#            print("x_i_p1 = \n {}".format(x_i_p1))
#            print('-'*20)
#            print("y_i_p1 = \n {}".format(y_i_p1))
#            print('-'*20)
#            print('/'*50)
#            print("TIME IN")

        # cu : cube ; sq : square
        x_i_cu = x_i * x_i * x_i
        x_i_sq = x_i * x_i 
        x_i_p1_cu = x_i_p1 * x_i_p1 * x_i_p1
        x_i_p1_sq = x_i_p1 * x_i_p1 
        y_i_cu = y_i * y_i * y_i
        y_i_sq = y_i * y_i 
        y_i_p1_cu = y_i_p1 * y_i_p1 * y_i_p1
        y_i_p1_sq = y_i_p1 * y_i_p1
    
        # Need to find a way to sum all the points' coordinates within a region 
        # m_01 
        moment_tensors[index, 0, 1] = centroid_y[index] * areas[index]
        
        # m_10
        moment_tensors[index, 1, 0] = centroid_x[index] * areas[index]   
#        print("m10 = {}".format(moment_tensors[index, 1, 0]))
        tmp += moment_tensors[index, 1, 0]
#        print("tmp = {}".format(tmp))
        
        # m_02
        moment_tensors[index, 0, 2] = np.sum( 
            (x_i - x_i_p1) * (y_i_cu + y_i_sq * y_i_p1 + y_i * y_i_p1_sq + y_i_p1_cu), axis=0)
        
        # m_20
        moment_tensors[index, 2, 0] = np.sum( 
            (x_i_cu + x_i_sq * x_i_p1 + x_i * x_i_p1_sq + x_i_p1_cu ) * (y_i - y_i_p1), axis=0)
        
        # m_11
        moment_tensors[index, 1, 1] = np.sum( 
        (x_i - x_i_p1) * (x_i * (3*y_i_sq + 2*y_i*y_i_p1 + y_i_p1_sq) 
                            + x_i_p1 * (y_i_sq + 2*y_i*y_i_p1 + 3*y_i_p1_sq)), axis=0)
    
    
    # m_00 = area (assumed already computed with Quick Hull)
    # Not computed
    # m_12
    # Not computed
    # m_21
    # Not computed
    # m_22
    # Not computed
    
    # Do the divisions once at the end
    moment_tensors[:, 1, 1] = moment_tensors[:, 1, 1] * (1.0/24.0)                  
    moment_tensors[:, 0, 2] = moment_tensors[:, 0, 2] * (1.0/12.0)   
    moment_tensors[:, 2, 0] = moment_tensors[:, 2, 0] * (-1.0/12.0)   
    
    # Ensure that the Ixx and Iyy are positive
#    moment_tensors[:, 0, 0] = np.abs(moment_tensors[:, 0, 0])
#    moment_tensors[:, 1, 1] = np.abs(moment_tensors[:, 1, 1])   
    return moment_tensors 

def compute_ellipse_parameters(central_moment_tensors):
    """
    Compute the fitted ellipses for every cell using the central moment tensors
    Based on Mulchrone and Choudhury, J. of Structural Geology 26 (2004)
    
    Parameters:
    -----
    central_moment_tensors : numpy array 
        Array of 3X3 central moment tensors for every cell 
   
    Returns:
    -----
    semi_minor_axes : numpy array
        Semi minor axes of the fitted ellipse   
    semi_major_axes : numpy array
        Semi major axes of the fitted ellipse
    angles_alpha : numpy array
        Angles (in radian) between the major axis of the fitted ellipse and the positive x-axis
    """
   
    length_central_moment_tensors = len(central_moment_tensors)
    semi_minor_axes = np.zeros(length_central_moment_tensors)    
    semi_major_axes = np.zeros(length_central_moment_tensors)
    angles_alpha = np.zeros(length_central_moment_tensors)
    delta = np.zeros(length_central_moment_tensors)
   
    u_11 = central_moment_tensors[:,1,1]
    u_02 = central_moment_tensors[:,0,0]
    u_20 = central_moment_tensors[:,2,0]
    delta = (4 * u_11 * u_11 + (u_20 - u_02)*(u_20 - u_02) )**(0.5)
    
    angles_alpha = 2 * u_11 / (u_20 - u_02) 
    semi_minor_axes = 2 * (u_20 + u_02 - delta) / u_11
    semi_major_axes = 2 * (u_20 + u_02 + delta) / u_11

    print('x'*20)    
    print('Delta : {}'.format(delta))
    print('x'*20)
    print("Semi minor before sqrt : \n {}".format(semi_minor_axes))
    print('x'*20)
    print("Semi major before sqrt : \n {}".format(semi_major_axes))

    # XXX : see with absolute values 
#    semi_minor_axes = np.absolute(semi_minor_axes)
#    semi_major_axes = np.absolute(semi_major_axes)

    print('x'*20)
    print("Semi minor after abs : \n {}".format(semi_minor_axes))
    print('x'*20)
    print("Semi major after abs : \n {}".format(semi_major_axes))


    # Separated the operations for clarity
    angles_alpha = 0.5 * np.arctan(angles_alpha)
    semi_minor_axes = np.sqrt(semi_minor_axes)
    semi_major_axes = np.sqrt(semi_major_axes)

    print('x'*20)
    print("Angles after abs : \n {}".format(angles_alpha))
    print('x'*20)
    print("Semi minor after sqrt : \n {}".format(semi_minor_axes))
    print('x'*20)
    print("Semi major after sqrt : \n {}".format(semi_major_axes))
 
    return semi_minor_axes, semi_major_axes, angles_alpha


def compute_central_moment_tensors(moment_tensors, points, areas):
    """
    Compute a central moment tensor (u) for every cell (simple polygons, 0th, 1st and 2nd moments)
    Based on Mulchrone and Choudhury, J. of Structural Geology 26 (2004)
    
    Parameters:
    -----
    moment_tensors : numpy array 
        Array of 3X3 tensors for every cell
    points : numpy array
        Coordinates of Voronoi centers
    areas : numpy array
        Areas of Voronoi cells
   
    Returns:
    -----
    central_moment_tensors : numpy array 
        Array of 3X3 central moment tensors for every cell
    """
    
    ### Only need u_11, u_02 and u_20 to compute ellipses' parameters
    central_moment_tensors = np.zeros([len(moment_tensors), 3, 3])
    
    central_moment_tensors[:, 1, 1] = moment_tensors[:, 1, 1] - points[:, 0] * points[:, 1] * areas[:]
    central_moment_tensors[:, 0, 2] = moment_tensors[:, 0, 2] - points[:, 1] * points[:, 1] * areas[:]
    central_moment_tensors[:, 2, 0] = moment_tensors[:, 2, 0] - points[:, 0] * points[:, 0] * areas[:]
    
    return central_moment_tensors



##########################################################

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
    angles_alpha : numpy array
        Values in radians

    Returns :
    -----
    #TODO
    """
#    plt.figure()
    ax = plt.gca()
    # XXX
    number_to_plot = 3

    # Convert angles alpha to degrees for plotting purposes
    angles_alpha_deg = np.degrees(angles_alpha)
    point = points[number_to_plot] 
    semi_major_axis = semi_major_axes[number_to_plot]
    semi_minor_axis = semi_minor_axes[number_to_plot]
    print("semi_major_axis = {}".format(semi_major_axis))
    print("semi_minor_axis = {}".format(semi_minor_axis))

    angle_alpha = -angles_alpha_deg[number_to_plot]
    print("ANGLE ALPHA DEG= {}".format(angle_alpha))
#    angle_alpha = 180 - 130
    ellipse = Ellipse(xy=(point[0],point[1]), width=semi_minor_axis, height=semi_major_axis, angle=180+angle_alpha)
    ax.add_patch(ellipse)
    plt.plot(point[0],point[1],'ko')
        
#    for point, semi_major_axis, semi_minor_axis, angle_alpha in zip(points, semi_major_axes, semi_major_axes, angles_alpha):
#        plt.plot(point[0],point[1],'g+') 
#        ellipse = Ellipse(xy=(point[0],point[1]), width=semi_minor_axis, height=semi_major_axis, angle=angle_alpha)
#        ax.add_patch(ellipse)
#    for point in points:
#        print(point)
    

def print_debug(ellipse_to_plot, points, areas, moment_tensors, central_moment_tensors, semi_major_axes, semi_minor_axes ):

    print("-"* 40)
    print("START PRINT DEBUG")
    print("Center of region[{}]: {}".format(ellipse_to_plot, points[ellipse_to_plot]))
    print('='*10)
    print("Vertices of the polygon in region[{}]: \n {}".format(ellipse_to_plot, vertices[regions[ellipse_to_plot]]))
    print('='*10)
    print("Area of region[{}]: {}".format(ellipse_to_plot, areas[ellipse_to_plot]))
    print('='*10)
    print("Moment tensor of region[{}]: \n {}".format(ellipse_to_plot, moment_tensors[ellipse_to_plot]))
    print('='*10)
    print("Central moment tensor of region[{}]: \n {}".format(ellipse_to_plot, central_moment_tensors[ellipse_to_plot]))
    print('='*10)
    print("Semi-major axis of region[{}]: {}".format(ellipse_to_plot, semi_major_axes[ellipse_to_plot]))
    print('='*10)
    print("Semi-minor axis of region[{}]: {}".format(ellipse_to_plot, semi_minor_axes[ellipse_to_plot]))
    print('='*10)

    for index, vertex in enumerate(vertices[regions[ellipse_to_plot]]):
        plt.plot(vertex[0],vertex[1], 'kx')
        plt.text(vertex[0],vertex[1],str(index) ,color="black", fontsize=12)


    print("END PRINT DEBUG")
    print("-"* 40)

if __name__=='__main__':
#    import sys
#    sys.settrace()
    # Generate a random tesselation
    boxsize = 10*10
    point_number = 100
    np.random.seed(32)

    points = initialization(point_number, boxsize)
    # points = np.array([[0, 0], [0, 1],[0.9,1],[1,0.5] ,[0, 2], [1, 5], [1, 1], [1, 2],[2, 0], [2, 1], [2, 2]])   
#    points = np.array([[0, 0], [0, 1],[0.5,1],[1,0.5] ,[0, 2], [1, 0], [1, 1], [1, 2],[2, 0], [2, 1], [2, 2]])*10
    # points = np.asarray(generate_periodic_voronoi_cells_2(points,boxsize))
#    print("TYPEEEEEE : {}".format(type(points)))

    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor, boxsize)
    index, perimeters, areas = get_polygons_properties_2d(regions, vertices)
    # for index, region in enumerate(regions):
    #     print("Vertices : \n {}".format(vertices[region]))
    #     print('-'*10)
    #     print("Perimeters : \n {}".format(perimeters[index]))
    #     print('-'*10)
    #     print("Areas : \n {}".format(areas[index]))
    #     print('*'*20)




    ### Test equality between functions
    # print(compute_moment_tensors(regions, vertices, points, areas)==compute_moment_tensors(regions, vertices, points, areas))
    # print('-'*40)
    # moment_tensors = compute_moment_tensors(regions, vertices, points, areas)

    #XXX DEBUG
#    id_region = 3
#    print("points[{}] = {}".format(id_region, points[id_region]))
#    print("vertices[regions[{}]] = \n {}".format(id_region, vertices[regions[id_region]]))
#    print("Moment tensor : \n {}".format(moment_tensors[id_region]))

# START

    moment_tensors = compute_moment_tensors(regions, vertices, points, areas)
    # XXX : Debug, test values for the central_moment_tensors
    central_moment_tensors = compute_central_moment_tensors(moment_tensors, points, areas)
    # central_moment_tensors = np.zeros([len(points),3,3])
    # central_moment_tensors[:,1,1] = 1
    # central_moment_tensors[:,0,2] = 5
    # central_moment_tensors[:,2,0] = 7


#     # Restricted to the observation box
    semi_minor_axes, semi_major_axes, angles_alpha = compute_ellipse_parameters(central_moment_tensors)
    plot_polygons(vor, regions, vertices)

    # print('-'*10)
    # print('MOMENT TENSORS :') 
    # print(moment_tensors)
    # print('-'*10)
    # print('-'*10)
    # print('CENTRAL MOMENT TENSORS :') 
    # print(central_moment_tensors)
    # print('-'*10)


#    print("Semi-major axes = \n{}".format(semi_major_axes))  
    # Figure parameters
    plt.xlim(-boxsize/2.0, boxsize/2.0)
    plt.ylim(-boxsize/2.0, boxsize/2.0)
    plt.title("Deterministic test model. {0} regions".format(len(regions)))
    plt.xlabel("x")
    plt.ylabel("y")

    # XXX Try with prefactors in front of the ellipses
    # semi_minor_axes = semi_minor_axes * 0.5
    # semi_major_axes = semi_major_axes * 0.5

    print("Points vus par plot_ellipses :")
#    plot_ellipses(points, 2*semi_major_axes, 2*semi_minor_axes, angles_alpha)
 #    print("-"*20) 
 #    print("Vrais points length = {}".format(len(points[:point_number])))
    for index, point in enumerate(points[0:point_number]):
        plt.plot(point[0],point[1], 'k+')
        plt.text(point[0],point[1],str(index) ,color="red", fontsize=12)
    # plt.show() 
    
    ##### XXX Start debug 
    ellipse_to_plot = 3
#    print_debug(ellipse_to_plot, points, areas, moment_tensors, central_moment_tensors, semi_major_axes, semi_minor_axes)
    ax_test = plt.gca()
#    pet.plot_ellipse(semimaj=semi_major_axes[ellipse_to_plot], semimin=semi_minor_axes[ellipse_to_plot], phi=np.radians(angles_alpha[ellipse_to_plot]),
#                                x_cent=points[ellipse_to_plot,0], y_cent=points[ellipse_to_plot, 1], ax=ax_test)
    plt.show()                    
# STOP



#    plt.xlim(-boxsize/2.0, boxsize/2.0)
#    plt.ylim(-boxsize/2.0, boxsize/2.0)
#    plt.title("Deterministic test model. {0} regions".format(len(regions)))
#    plt.xlabel("x")
#    plt.ylabel("y")
#    plt.show() 

