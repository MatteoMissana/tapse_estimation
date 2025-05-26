import cupy as cp
import numpy as np
import torch
from numpy.polynomial import polynomial as poly

def tric_apex_distance_calculation(free_wall, septum, apex):
    """
    Calculate the distance from the apex to the free wall and septum.
    also calculates the diameter of the right ventricle, and the area of the triangle defined by those three points.
    the output arrays are of the same lenght as the input, with the measure calculated for each time interval"""

    midpoint = (free_wall + septum) / 2

    # Compute distances (Euclidean norm, along last axis)
    dist_0 = np.linalg.norm(free_wall - apex, ord=2, axis=-1)
    dist_1 = np.linalg.norm(septum - apex, ord=2, axis=-1)
    dist_2 = np.linalg.norm(midpoint - apex, ord=2, axis=-1)


    diameter = np.linalg.norm(free_wall - septum, ord=2, axis=-1)

    semiperimeter = (dist_0 + dist_1 + diameter) / 2
    # Area using Heron's formula
    area = np.sqrt(
        semiperimeter
        * (semiperimeter - dist_0)
        * (semiperimeter - dist_1)
        * (semiperimeter - diameter)
    )

    diast_area = area.max()
    syst_area = area.min()

    # calculate rvfac surrogate
    rvfac = (diast_area - syst_area) / diast_area * 100



    

    rvldfw = dist_0.max()
    rvldsep = dist_1.max()
    rvlsfw = dist_0.min()
    rvlssep = dist_1.min()
    rvldmid = dist_2.max()
    rvlsmid = dist_2.min()

    rvlsffw = (rvldfw - rvlsfw)/ rvldfw * 100
    rvlsfsep = (rvldsep - rvlssep)/ rvldsep * 100
    rvlsfmid = (rvldmid - rvlsmid)/ rvldmid * 100
    rvlsfglobal = ((rvldfw+rvldsep)-(rvlsfw+rvlssep))/(rvldfw+rvldsep) * 100

    tadd = diameter.max() 
    tasd = diameter.min()



    return rvfac, diast_area, syst_area, rvldfw, rvldsep, rvlsfw, rvlssep, rvldmid, rvlsmid, tadd, tasd, rvlsffw, rvlsfsep, rvlsfmid, rvlsfglobal

def tapse_calculation(
    coordinates_septum: np.ndarray,
    coordinates_fw: np.ndarray,
    direction: np.ndarray,
    tapse_calc = 'distance'
):
    """
    Calculate TAPSE (Tricuspid Annular Plane Systolic Excursion) from the coordinates of the septum and free wall.
    The coordinates should be in the format (x, y) and the direction should be a unit vector.
    The pixelsize is a list containing the pixel size in mm for each dimension.

    based on the parameter "tapse_calc", the function will calculate the tapse in two different ways:
    - 'distance': just calculates the maximum distance between the points in the septum and free wall for the time you provide
    - 'projection': projects the points in the direction of the vector and calculates the distance between the maximum and minimum projection for both septum and free wall, then averages them
    """

    if tapse_calc not in ['distance', 'projection']:
        raise ValueError("tapse_calc must be either 'distance' or 'projection'")
    elif tapse_calc == 'projection' and direction is None:
        raise ValueError("If tapse_calc is 'projection', direction must be provided")
    
    elif tapse_calc == 'distance':
        diff_septum = coordinates_septum[:, np.newaxis, :] - coordinates_septum[np.newaxis, :, :]  # forma (n, n, 2)
        dist_septum = np.linalg.norm(diff_septum, axis=-1)  # forma (n, n)
        tapse_septum = dist_septum.max()

        diff_fw = coordinates_fw[:, np.newaxis, :] - coordinates_fw[np.newaxis, :, :]  # forma (n, n, 2)
        dist_fw = np.linalg.norm(diff_fw, axis=-1)  # forma (n, n)
        tapse_fw = dist_fw.max()
        tapse = (tapse_septum + tapse_fw) / 2

    elif tapse_calc == 'projection':
        projection_septum = coordinates_septum @ direction
        projection_fw = coordinates_fw @ direction
        tapse_septum = projection_septum.max() - projection_septum.min()
        tapse_fw = projection_fw.max() - projection_fw.min()
        tapse = (tapse_septum + tapse_fw) / 2

    return tapse_septum, tapse_fw, tapse

def find_parallel_direction(points):
    """
    Find the direction of the vector that is parallel to the array.
    The array should be a 2D numpy array with shape (n, 2).
    """
    if len(points) < 2:
        raise ValueError("Array must contain at least two points.")
    
    # Fit a linear polynomial: returns [intercept, slope]
    b, m = poly.polyfit(points[0], points[1], 1)

    # Create direction vector using the slope
    v = np.array([1, m])  # dx = 1, dy = m

    # Normalize to get the versor
    versor = v / np.linalg.norm(v)
    print("versor", versor)
    
    return versor