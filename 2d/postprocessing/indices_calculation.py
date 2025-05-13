import cupy as cp
import numpy as np
import torch
from numpy.polynomial import polynomial as poly

def tric_apex_distance_calculation(free_wall, septum, apex):

    # middle point (no scaling here, as in original)
    middle = (free_wall + septum) / 2

    # Compute distances (Euclidean norm, along last axis)
    dist_0 = np.linalg.norm(free_wall - apex, ord=2, axis=-1)
    dist_1 = np.linalg.norm(septum - apex, ord=2, axis=-1)
    dist_2 = np.linalg.norm(middle - apex, ord=2, axis=-1)
    diameter = np.linalg.norm(free_wall - septum, ord=2, axis=-1)

    # Average distance
    dist = (dist_0 + dist_1 + dist_2) / 3

    return dist, diameter


def tapse_calculation(
    coordinates_septum: np.ndarray,
    coordinates_fw: np.ndarray,
    direction: np.ndarray,
):
    """
    Calculate TAPSE (Tricuspid Annular Plane Systolic Excursion) from the coordinates of the septum and free wall.
    The coordinates should be in the format (x, y) and the direction should be a unit vector.
    The pixelsize is a list containing the pixel size in mm for each dimension.
    """

    projection_septum = coordinates_septum @ direction
    projection_fw = coordinates_fw @ direction
    tapse_septum = projection_septum.max() - projection_septum.min()
    tapse_fw = projection_fw.max() - projection_fw.min()
    tapse = (tapse_septum + tapse_fw) / 2

    return tapse

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