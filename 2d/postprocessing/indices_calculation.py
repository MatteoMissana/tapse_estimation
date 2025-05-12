import cupy as cp
import numpy as np
import torch
from numpy.polynomial import polynomial as poly

def tric_apex_distance_calculation(free_wall, septum, apex, pixelsize):
    middle = (free_wall + septum)/2

    dist_0 = torch.linalg.vector_norm(free_wall - apex, ord=2)
    dist_1 = torch.linalg.vector_norm(septum - apex, ord=2)
    dist_2 = torch.linalg.vector_norm(middle - apex, ord=2)

    diameter = torch.linalg.vector_norm(free_wall - septum, ord=2)

    dist = (dist_0 + dist_1 + dist_2) / 3
    dist = dist.cpu().numpy()
    diameter = diameter.cpu().numpy() * pixelsize[0] #TODO: change this to adapt to the case where pixelsize has 2 different values
    return dist, diameter

def tapse_calculation(
    coordinates_septum: np.ndarray,
    coordinates_fw: np.ndarray,
    direction: np.ndarray,
    pixelsize: list,
):
    """
    Calculate TAPSE (Tricuspid Annular Plane Systolic Excursion) from the coordinates of the septum and free wall.
    The coordinates should be in the format (x, y) and the direction should be a unit vector.
    The pixelsize is a list containing the pixel size in mm for each dimension.
    """

    projection_septum = coordinates_septum @ direction
    print("projection_septum", projection_septum)
    projection_fw = coordinates_fw @ direction
    tapse_septum = projection_septum.max() - projection_septum.min()
    tapse_fw = projection_fw.max() - projection_fw.min()
    tapse = (tapse_septum + tapse_fw) / 2

    return tapse * pixelsize[0] #TODO: change this to adapt to the case where pixelsize has 2 different values

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