
# coding: utf-8

# In[1]:


### IMPORTS
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from voronoi_tesselation import get_polygons_properties_2d, voronoi_finite_polygons_2d, voronoi_plot_2d
from voronoi_tesselation import generate_periodic_voronoi_cells_2
from voronoi_tesselation import plot_polygons
import numpy as np
import plot_ellipses_test as pet
from matplotlib.patches import Ellipse
from ellipse_fitting_polygon import initialization


# In[2]:


def compute_moment_tensors_full(regions, vertices, points, areas):
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
    #centroid_x = points[:,0] 
    #centroid_y = points[:,1]

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
        # m_00 = area (assumed already computed with Quick Hull)
        moment_tensors[index, 0, 0] = np.sum((x_i + x_i_p1) * (y_i - y_i_p1), axis=0)
        # m_01 
#        moment_tensors[index, 0, 1] = centroid_y[index] * areas[index]
        moment_tensors[index, 0, 1] = np.sum((x_i - x_i_p1) * (y_i_sq + y_i * y_i_p1 + y_i_p1_sq) , axis=0)
        # m_10
#        moment_tensors[index, 1, 0] = centroid_x[index] * areas[index]   
        moment_tensors[index, 1, 0] = np.sum( (x_i_sq + x_i * x_i_p1 + x_i_p1_sq) * (y_i - y_i_p1), axis=0)
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
 
    

    # m_12
    # Not computed
    # m_21
    # Not computed
    # m_22
    # Not computed
    
    # Do the divisions once at the end
    moment_tensors[:, 0, 0] = moment_tensors[:, 0, 0] * (-1.0/2.0)                     
    moment_tensors[:, 0, 1] = moment_tensors[:, 0, 1] * (1.0/6.0)
    moment_tensors[:, 1, 0] = moment_tensors[:, 1, 0] * (-1.0/6.0)
    moment_tensors[:, 1, 1] = moment_tensors[:, 1, 1] * (1.0/24.0)                  
    moment_tensors[:, 0, 2] = moment_tensors[:, 0, 2] * (1.0/12.0)   
    moment_tensors[:, 2, 0] = moment_tensors[:, 2, 0] * (-1.0/12.0)   
#    moment_tensors[:, 2, 0] = moment_tensors[:, 2, 0] * (1.0/12.0)      
    # Ensure that the Ixx and Iyy are positive
#    moment_tensors[:, 0, 0] = np.abs(moment_tensors[:, 0, 0])
#    moment_tensors[:, 1, 1] = np.abs(moment_tensors[:, 1, 1])   
    return moment_tensors 


# In[3]:


def compute_central_moment_tensors_full(moment_tensors, points, areas):
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
    
#    central_moment_tensors[:, 1, 1] = moment_tensors[:, 1, 1] - points[:, 0] * points[:, 1] * areas[:]
#    central_moment_tensors[:, 0, 2] = moment_tensors[:, 0, 2] - points[:, 1] * points[:, 1] * areas[:]
#    central_moment_tensors[:, 2, 0] = moment_tensors[:, 2, 0] - points[:, 0] * points[:, 0] * areas[:]
    
    central_moment_tensors[:, 1, 1] = moment_tensors[:, 1, 1] - moment_tensors[:, 1, 0] * moment_tensors[:, 0, 1] / moment_tensors[:, 0, 0]
    central_moment_tensors[:, 0, 2] = moment_tensors[:, 0, 2] - moment_tensors[:, 0, 1] * moment_tensors[:, 0, 1] / moment_tensors[:, 0, 0]
    central_moment_tensors[:, 2, 0] = moment_tensors[:, 2, 0] - moment_tensors[:, 1, 0] * moment_tensors[:, 1, 0] / moment_tensors[:, 0, 0]
    
    
    return central_moment_tensors


# In[4]:


def compute_ellipse_parameters(central_moment_tensors, no_abs_denominator=False):
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
    u_02 = central_moment_tensors[:,0,2]
    u_20 = central_moment_tensors[:,2,0]
    delta = (4 * u_11 * u_11 + (u_20 - u_02)*(u_20 - u_02) )**(0.5)
    
    ### XXX : DEBUG
    if(no_abs_denominator):
        angles_alpha = 2 * u_11 / (u_20 - u_02) 
    else:
        angles_alpha = 2 * u_11 / np.abs(u_20 - u_02) 
        
    semi_minor_axes = 2 * (u_20 + u_02 - delta) / u_11
    semi_major_axes = 2 * (u_20 + u_02 + delta) / u_11

    print('x'*20)    
    print('Delta : {}'.format(delta))
    print('x'*20)
    print("Semi minor before sqrt : \n {}".format(semi_minor_axes))
    print('x'*20)
    print("Semi major before sqrt : \n {}".format(semi_major_axes))

    # XXX : see with absolute values 
    semi_minor_axes = np.absolute(semi_minor_axes)
    semi_major_axes = np.absolute(semi_major_axes)

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


# In[5]:


def rotation_matrix(angle):
    """
    Rotation matrix to apply to polygons 

    Parameters:
    -----
    angle : double
        Angle of rotation  in radians.
    """
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rot_mat


# In[6]:


def ellipse_Hiraiwa(regions, vertices):
    len_regions = len(regions)
    moment_tensors = np.zeros([len_regions, 3, 3])
    ## Compute the second moment of inertia
    for index, region in enumerate(regions):
        # Load the numpy arrays
        x_i = vertices[region, 0]
        y_i = vertices[region, 1]

        # use np.roll to have a "+1" array that wraps at the end
        x_i_p1 = np.roll(x_i, -1) 
        y_i_p1 = np.roll(y_i, -1) 

        # cu : cube ; sq : square
#        x_i_cu = x_i * x_i * x_i
#        x_i_sq = x_i * x_i 
#        x_i_p1_cu = x_i_p1 * x_i_p1 * x_i_p1
#        x_i_p1_sq = x_i_p1 * x_i_p1 
#        y_i_cu = y_i * y_i * y_i
#        y_i_sq = y_i * y_i 
#        y_i_p1_cu = y_i_p1 * y_i_p1 * y_i_p1
#        y_i_p1_sq = y_i_p1 * y_i_p1

        # Need to find a way to sum all the points' coordinates within a region 
        # m_00 = area (assumed already computed with Quick Hull)
        moment_tensors[index, 1, 1] = np.sum((x_i*y_i_p1-x_i_p1*y_i)*(y_i*y_i+y_i*y_i_p1+y_i_p1*y_i_p1)/12.0,axis=0)
        moment_tensors[index, 1, 2] = np.sum(-(x_i*y_i_p1-x_i_p1*y_i)*(x_i*(2*y_i+y_i_p1) + x_i_p1*(y_i+2*y_i_p1))/24.0, axis=0) 
        moment_tensors[index, 2, 2] = np.sum((x_i*y_i_p1-x_i_p1*y_i)*(x_i*x_i+x_i*x_i_p1+x_i_p1*x_i_p1)/12.0, axis=0)
        
    ## Compute the ellipse parameters 
    semi_minor_axes = np.zeros(len_regions)    
    semi_major_axes = np.zeros(len_regions)
    angles_alpha = np.zeros(len_regions)
    deformation_degree = np.zeros(len_regions)
    I_11 = moment_tensors[:, 1, 1]
    I_12 = moment_tensors[:, 1, 2]
    I_22 = moment_tensors[:, 2, 2]
    
    ## Degree of deformation
    deformation_degree = np.sqrt((I_11-I_22)*(I_11-I_22) + 4.0*I_12*I_12)  
    
    ## Angle
   
    angles_alpha = -0.5 * np.arctan(-2.0*I_12 / (I_22-I_11))
    
    print("I_11 = \n {}".format(I_11))
    print("I_22 = \n {}".format(I_22))
    print("I_12 = \n {}".format(I_12))
    return deformation_degree, angles_alpha


# In[7]:


def plot_ellipse(semi_major_axes, semi_minor_axes, centroids, angles_alpha, color='green', plotting_range=None):
    """
    Assumes that we possess every parameter, this is purely a plotting function.
    Note for Ellipse: 
    width = total length (diameter) of horizontal axis
    height = total length (diameter) of vertical axis
    angle = rotation ink
    
    Parameters:
    -----
    semi_major_axes: numpy array
        Semi major axes of the fitted ellipses
    semi_minor_axes: numpy array
        Semi minor axes of the fitted ellipses
    angles_alpha:
        Values in radians
    
    Returns:
    -----
    # TODO: what are the best practices for figures
    """
    nb_ellipses = len(centroids)
    angles_alpha_deg = np.degrees(angles_alpha)
    # Get the current axis to add the ellipse patches
    ax = plt.gca()
    
    if plotting_range is None:
        plotting_range = nb_ellipses
    else:
        pass
    for index in range(plotting_range):
        ellipse = Ellipse(xy=(centroids[index, 0], centroids[index, 1]), height=2*semi_minor_axes[index]
                          , width=2*semi_major_axes[index], angle = angles_alpha_deg[index])
#                          , width=2*semi_major_axes[index], angle = 45)
        ellipse.set_facecolor(str(color))
        ellipse.set_alpha(0.3)
        ellipse.set_linestyle('--')
        ax.add_patch(ellipse)
        plt.plot(centroids[index, 0], centroids[index, 1], 'ko')
        plt.text(centroids[index, 0], centroids[index, 1], str(index), color='red', fontsize=12)


# In[8]:


def create_polygons():
    """
    Handmade polygons for testing purposes
    """
    ### Create some hand made polygons by specifying their vertices (will have to compute their centroid)
    poly_1 = np.array([[1,1], [3,1], [8,6], [8,9], [5,8]])
    poly_2 = np.array([[2,2], [7,2], [7,7], [2,7]])
    poly_6 = np.array([[4,1], [7,2], [6,10], [4,7], [2,5]])
    poly_4 = np.array([[8,3], [10,3], [10,6], [4,9], [1,8], [2,6]])
    poly_5 = np.array([[3,1], [6,1], [7,4], [6,7], [3,7], [2,4]])
    poly_3 = np.array([[3,0], [6,1], [9,1], [11,2], [13,5], [11,7], [8,6], [7,7], [5,4], [3,5], [2,2]])

    ### Translate them for plotting purposes
    poly_2 = poly_2 + 10
    poly_3 = poly_3 + 20
    poly_4 = poly_4 + 30
    poly_5 = poly_5 + 40
    poly_6 = poly_6 + 50

    ### Bring them together
    poly_all = np.concatenate((poly_1, poly_2))
    poly_all = np.concatenate((poly_all, poly_3))
    poly_all = np.concatenate((poly_all, poly_4))
    poly_all = np.concatenate((poly_all, poly_5))
    poly_all = np.concatenate((poly_all, poly_6))

    ### Define the corresponding regions
    region_1 = [x for x in range(len(poly_1))]
    region_2 = [x+len(region_1) for x in range(len(poly_2))]
    region_3 = [x+len(region_1+region_2) for x in range(len(poly_3))]
    region_4 = [x+len(region_1+region_2+region_3) for x in range(len(poly_4))]
    region_5 = [x+len(region_1+region_2+region_3+region_4) for x in range(len(poly_5))]
    region_6 = [x+len(region_1+region_2+region_3+region_4+region_5) for x in range(len(poly_6))]

    region_all = []
    region_all +=[region_1]
    region_all +=[region_2]
    region_all +=[region_3]
    region_all +=[region_4]
    region_all +=[region_5]
    region_all +=[region_6]
    
    return poly_all, region_all


# In[9]:


### Computation part
def fit_ellipses_to_cells(regions, vertices, points=None, areas=None, debug=False):
    """
    Compute the "best fit" ellipse to Voronoi cells (or any polygons) using their moment tensors and their central moment tensors. 

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
    debug : boolean
        When true, compute angles between the major axis and the x-axis with and without an absolute value on the denominator.

    Returns:
    -----
    # TODO : a figure maybe ?
    """
    
    moment_tensors = compute_moment_tensors_full(regions, vertices, points, areas)
    central_moment_tensors = compute_central_moment_tensors_full(moment_tensors, points, areas)
    
    if(debug):
        semi_minor_axes, semi_major_axes, angles_alpha_abs = compute_ellipse_parameters(central_moment_tensors, no_abs_denominator=False)
        semi_minor_axes, semi_major_axes, angles_alpha_no_abs = compute_ellipse_parameters(central_moment_tensors, no_abs_denominator=True)
    else:
        semi_minor_axes, semi_major_axes, angles_alpha = compute_ellipse_parameters(central_moment_tensors, no_abs_denominator=False)
#    # Debug
#    print("DEBUUUUUUUUUG")
#    for index, cmt in enumerate(central_moment_tensors):
#        print("cmt[{}] = \n {}".format(index,cmt))
    
    
    centroids = np.zeros([len(semi_major_axes), 2])
    
    if points is None:
        # Assuming we don't have access to the Voronoi centers
        centroids[:,0] = moment_tensors[:, 1, 0] / moment_tensors[:, 0, 0]
        centroids[:,1] = moment_tensors[:, 0, 1] / moment_tensors[:, 0, 0]
    else:
        centroids[:,0] = points[:,0]
        centroids[:,1] = points[:,1]   
        
    if(debug):
        plot_ellipse(semi_major_axes, semi_minor_axes, centroids, angles_alpha_abs, color='blue')
        plot_ellipse(semi_major_axes, semi_minor_axes, centroids, angles_alpha_no_abs, color='green')
    else:
        plot_ellipse(semi_major_axes, semi_minor_axes, centroids, angles_alpha, color='red')
#    plt.show()


# In[10]:


def plot_full_polygons(regions, vertices):
    """
    Draws the polygon entirely by circling through every point

    Parameters:
    -----
    regions : numpy array 
        Array of arrays pointing to the "vertices" array, determining a cell
    vertices : numpy array
        Coordinates of cell vertices

    Returns:
    -----
    Nothing
    """
    for region in regions:
        tmp_poly = np.zeros([len(vertices[region])+1,2])
        tmp_poly[0:len(vertices[region])] = vertices[region][:]
        # Rollback
        tmp_poly[-1] = vertices[region][0]
        plt.plot(tmp_poly[:,0], tmp_poly[:,1])


# In[11]:

if __name__ == '__main__':

    poly_all, regions_all = create_polygons()

    # Draw the (incomplete) polygons
    #for r in regions_all:
    #    plt.plot(poly_all[r,0], poly_all[r,1])
    plot_full_polygons(regions_all, poly_all)    
    fit_ellipses_to_cells(regions_all, poly_all)
    plt.show()


    # In[20]:


    #%matplotlib notebook
    ### Try different rotations
    poly_all, regions_all = create_polygons()
    range_to_study = np.arange(0.0, 1.25, 0.13)
    for rotation_factor in range_to_study:
        # Rotate the polygons
        angle_to_rotate = np.pi * rotation_factor
        poly_rot_all = poly_all
        for index, point in enumerate(poly_rot_all):
            poly_rot_all[index] = np.matmul(rotation_matrix(angle_to_rotate), point)

        # Draw the (incomplete) polygons
        plt.figure()
        plt.title("Rotation angle {} * $\pi$. Blue : with abs, Green : without abs".format(rotation_factor))
        plot_full_polygons(regions_all, poly_all)
        fit_ellipses_to_cells(regions_all, poly_rot_all, debug=True)
        plt.axis('scaled')
#        plt.savefig("poly_rot_pi_x_{}_with_and_without_abs.png".format(rotation_factor))
        
        


    # In[13]:


    ### Shift every polygon with a different angle
    poly_all, regions_all = create_polygons()
    #np.random.seed(10)
    rotation_factor = np.random.rand(len(regions_all))
    curr_pos = 0
    plt.figure()
    for index,rot_factor in enumerate(rotation_factor):
        angle_to_rotate = np.pi * rot_factor
        for i, point in enumerate(poly_rot_all[regions_all[index]]):
            # Careful, this assumes that vertices and regions are ordered 
            poly_rot_all[curr_pos + i] = np.matmul(rotation_matrix(angle_to_rotate), point)
            
        curr_pos += len(regions_all[index])
            
    #    plt.savefig("poly_rot_pi_x_{}.png".format(rotation_factor))

    fit_ellipses_to_cells(regions_all, poly_rot_all)
    plot_full_polygons(regions_all, poly_rot_all)
    plt.title("Rotation angle {} * $\pi$".format(rotation_factor))
    plt.show()


    # In[15]:


    boxsize = 100
    #np.random.seed(29308)
    np.random.seed(4876)
    points = initialization(50,boxsize)
    points = np.asarray(generate_periodic_voronoi_cells_2(points,boxsize))

    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor, boxsize)
    index, perimeters, areas = get_polygons_properties_2d(regions, vertices)

    plot_full_polygons(regions,vertices)
    plt.xlim(-boxsize/2.0, boxsize/2.0)
    plt.ylim(-boxsize/2.0, boxsize/2.0)

    fit_ellipses_to_cells(regions, vertices, debug=True)
#    plt.savefig('test_debug.png')
    plt.show()

    # In[ ]:




