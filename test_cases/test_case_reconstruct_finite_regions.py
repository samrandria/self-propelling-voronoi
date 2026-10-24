"""
Reconstruct finite regions
"""

from scipy.spatial import Voronoi
import numpy as np


if __name__ == '__main__':

    points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
    vor = Voronoi(points)

    print("Ridge_point :")
    print(vor.ridge_points)
    print('-'*10) 
    print("Ridge_vertices :")
    print(vor.ridge_vertices)
    print('o'*10)
    
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

