"""
Test case to generate colormaps
"""

import matplotlib.pyplot as plt 
import numpy as np 

if __name__ == '__main__':

    data = np.random.random((50,50)) * 2 * np.pi

    x = np.linspace(0,100)
    y = np.linspace(0,100)

    # Load the desired colormap
    # cmap = plt.get_cmap('viridis_r')

    # Normalize the data
    plt.pcolormesh(x,y,data, vmin=0,vmax=2*np.pi,cmap='viridis')


    # Display the colorbar in legend
    # pi_map = np.linspace(0, 2*np.pi) 
    # plt.colorbar(ticks=pi_map)
    plt.colorbar()
    plt.show()